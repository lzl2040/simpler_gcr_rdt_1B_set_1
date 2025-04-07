import os
import fnmatch
import json
import yaml
import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import json
import torchvision
import importlib
import logging


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"

def convert_euler_to_rotation_matrix(euler):
    """
    Convert Euler angles (rpy) to rotation matrix (3x3).
    """
    quat = R.from_euler('xyz', euler).as_matrix()
    
    return quat

def compute_ortho6d_from_rotation_matrix(matrix):
    # The ortho6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return ortho6d

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix

def convert_rotation_matrix_to_euler(rotmat):
    """
    Convert rotation matrix (3x3) to Euler angles (rpy).
    """
    r = R.from_matrix(rotmat)
    euler = r.as_euler('xyz', degrees=False)
    
    return euler

def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = np.stack((i, j, k), axis=1)
    return out


STATE_VEC_IDX_MAPPING = {
    # [0, 10): right arm joint positions
    **{
        'arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    **{
        'right_arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    # [10, 15): right gripper joint positions
    **{
        'gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    'gripper_open': 10, # alias of right_gripper_joint_0_pos
    'right_gripper_open': 10,
    # [15, 25): right arm joint velocities
    **{
        'arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    **{
        'right_arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    # [25, 30): right gripper joint velocities
    **{
        'gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    'gripper_open_vel': 25, # alias of right_gripper_joint_0_vel
    'right_gripper_open_vel': 25,
    # [30, 33): right end effector positions
    'eef_pos_x': 30,
    'right_eef_pos_x': 30,
    'eef_pos_y': 31,
    'right_eef_pos_y': 31,
    'eef_pos_z': 32,
    'right_eef_pos_z': 32,
    # [33, 39): right end effector 6D pose
    'eef_angle_0': 33,
    'right_eef_angle_0': 33,
    'eef_angle_1': 34,
    'right_eef_angle_1': 34,
    'eef_angle_2': 35,
    'right_eef_angle_2': 35,
    'eef_angle_3': 36,
    'right_eef_angle_3': 36,
    'eef_angle_4': 37,
    'right_eef_angle_4': 37,
    'eef_angle_5': 38,
    'right_eef_angle_5': 38,
    # [39, 42): right end effector velocities
    'eef_vel_x': 39,
    'right_eef_vel_x': 39,
    'eef_vel_y': 40,
    'right_eef_vel_y': 40,
    'eef_vel_z': 41,
    'right_eef_vel_z': 41,
    # [42, 45): right end effector angular velocities
    'eef_angular_vel_roll': 42,
    'right_eef_angular_vel_roll': 42,
    'eef_angular_vel_pitch': 43,
    'right_eef_angular_vel_pitch': 43,
    'eef_angular_vel_yaw': 44,
    'right_eef_angular_vel_yaw': 44,
    # [45, 50): reserved 
    # [50, 60): left arm joint positions
    **{
        'left_arm_joint_{}_pos'.format(i): i + 50 for i in range(10)
    },
    # [60, 65): left gripper joint positions
    **{
        'left_gripper_joint_{}_pos'.format(i): i + 60 for i in range(5)
    },
    'left_gripper_open': 60, # alias of left_gripper_joint_0_pos
    # [65, 75): left arm joint velocities
    **{
        'left_arm_joint_{}_vel'.format(i): i + 65 for i in range(10)
    },
    # [75, 80): left gripper joint velocities
    **{
        'left_gripper_joint_{}_vel'.format(i): i + 75 for i in range(5)
    },
    'left_gripper_open_vel': 75, # alias of left_gripper_joint_0_vel
    # [80, 83): left end effector positions
    'left_eef_pos_x': 80,
    'left_eef_pos_y': 81,
    'left_eef_pos_z': 82,
    # [83, 89): left end effector 6D pose
    'left_eef_angle_0': 83,
    'left_eef_angle_1': 84,
    'left_eef_angle_2': 85,
    'left_eef_angle_3': 86,
    'left_eef_angle_4': 87,
    'left_eef_angle_5': 88,
    # [89, 92): left end effector velocities
    'left_eef_vel_x': 89,
    'left_eef_vel_y': 90,
    'left_eef_vel_z': 91,
    # [92, 95): left end effector angular velocities
    'left_eef_angular_vel_roll': 92,
    'left_eef_angular_vel_pitch': 93,
    'left_eef_angular_vel_yaw': 94,
    # [95, 100): reserved
    # [100, 102): base linear velocities
    'base_vel_x': 100,
    'base_vel_y': 101,
    # [102, 103): base angular velocities
    'base_angular_vel': 102,
    # [103, 128): reserved
}
STATE_VEC_LEN = 128


class HDF5VLADataset:
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        LEROBOT_DIR = "/Data/lerobot_data/simulated/libero_spatial_no_noops_lerobot"
        # HDF5_DIR = "/datassd_1T/dataset_cache/simpler_data"
        WEIGHT_FILE = "episode_sample_weights.npy"
        self.DATASET_NAME = "libero"
        self.emb_path = ""
        
        self.file_paths = []
        data_path = os.path.join(LEROBOT_DIR, "data")
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".parquet"):
                    self.file_paths.append(os.path.join(root, file))
        
        meta_path = os.path.join(LEROBOT_DIR, "meta")
        task_list_json = os.path.join(meta_path, "tasks.jsonl")
        self.task_dict_list = {}
        with open(task_list_json, "r", encoding="utf-8") as f:
            for line in f:
                task_dict = json.loads(line)
                task_id = task_dict["task_index"]
                task = task_dict["task"]
                self.task_dict_list[task_id] = task
        
        self.image_views = ["observation.images.image", " observation.images.wrist_image"]
        self.video_root = os.path.join(LEROBOT_DIR, "videos")
        self.backend = get_safe_default_codec()
        torchvision.set_video_backend(self.backend)

        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        if os.path.exists(os.path.join(LEROBOT_DIR, WEIGHT_FILE)):
            print("Loading episode sample weights from cache")
            # self.episode_sample_weights = np.load(os.path.join(HDF5_DIR, "episode_sample_weights.npy"))
            self.episode_sample_weights = np.load(os.path.join(LEROBOT_DIR, WEIGHT_FILE))
        else:
            print("Generating episode sample weights from raw data")
            episode_lens = []
            for file_path in tqdm(self.file_paths, desc="Generating weights from raw data"):
                valid, res = self.parese_parquet_file_state_only(file_path)
                _len = res['state'].shape[0] if valid else 0
                episode_lens.append(_len)
            self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    

    def get_ortho6d_from_euler_angle(self, states):
        new_states = []
        for i in range(len(states)):
            state = states[i:i+1]
            # print(state.shape)
            xyz = state[:, :3]
            euler = state[:,3:6]
            gripper = state[:, 6:7] # in fact, there are two gripper state, but we just select the first
            rot_mat = convert_euler_to_rotation_matrix(euler)
            orth6d = compute_ortho6d_from_rotation_matrix(rot_mat)
            new_state = np.concatenate((xyz, orth6d, gripper), axis=1)
            new_states.append(new_state[0])
        new_states = np.stack(new_states)
        return new_states
    
    def parese_parquet_file_state_only(self, file_path):
        data = pq.read_table(
            file_path,
            use_threads=True,  # 启用多线程
            memory_map=True    # 内存映射文件加速
        ).to_pandas()
        actions = np.stack(data["action"].values)
        states = np.stack(data["observation.state"].values)
        actions = self.get_ortho6d_from_euler_angle(actions)
        states = self.get_ortho6d_from_euler_angle(states)
        num_steps = len(actions)
        # [Optional] We drop too-short episode
        if num_steps < 30:
            return False, None

        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In our data: 3 translation, 6 rotation, 1 gripper
            UNI_STATE_INDICES = [ STATE_VEC_IDX_MAPPING['eef_pos_x']
                            ] + [ STATE_VEC_IDX_MAPPING['eef_pos_y']
                            ] + [ STATE_VEC_IDX_MAPPING['eef_pos_z']
                    ] + [
                        STATE_VEC_IDX_MAPPING[f"eef_angle_{i}"] for i in range(6)
                    ] + [
                        STATE_VEC_IDX_MAPPING["right_gripper_open"]
                    ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        
        states = fill_in_state(states)
        actions = fill_in_state(actions)
        
        return True, {
            "state": states,
            "action": actions
        }

    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                # file_path = np.random.choice(self.file_paths)
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_parquet_file(file_path) \
                if not state_only else self.parese_parquet_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def read_video(self, video_path):
        frames = []
        reader = torchvision.io.VideoReader(video_path, "video")
        for frame in reader:
            f_data = frame["data"].permute(1, 2, 0).numpy()
            frames.append(f_data)
        
        return np.array(frames)  # Convert to NumPy array

    def parse_parquet_file(self, file_path):
        # print("!24")
        chunk_name = file_path.split("/")[-2]
        episode_name = file_path.split("/")[-1].split(".")[0]
        primary_video_path = os.path.join(self.video_root, chunk_name, self.image_views[0], f"{episode_name}.mp4")
        # print(primary_video_path)
        wrist_video_path = os.path.join(self.video_root, chunk_name, self.image_views[1], f"{episode_name}.mp4")

        data = pq.read_table(
            file_path,
            use_threads=True,  # 启用多线程
            memory_map=True    # 内存映射文件加速
        ).to_pandas()
        actions = np.stack(data["action"].values)
        states = np.stack(data["observation.state"].values)
        task_ids = np.stack(data["task_index"].values)
        instruction = self.task_dict_list[task_ids[0]]
        actions = self.get_ortho6d_from_euler_angle(actions)
        states = self.get_ortho6d_from_euler_angle(states)
        frames = self.read_video(primary_video_path)
        # print(frames.shape)
        num_steps = len(actions)
        # [Optional] We drop too-short episode
        if num_steps < 30:
            return False, None
        
        # We randomly sample a timestep
        first_idx = 0
        step_id = np.random.randint(0, num_steps-1)

        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }

        state_norm = np.sqrt(np.mean(states**2, axis=0))
        state_std = np.std(states, axis=0)
        state_mean = np.mean(states, axis=0)
        
        CHUNK_SIZE = self.CHUNK_SIZE
        ACTUAL_CHUNK_SIZE = min(CHUNK_SIZE, num_steps - step_id)
        
        target_qpos = np.zeros((self.CHUNK_SIZE, 10)).astype(np.float32)
        for i in range(ACTUAL_CHUNK_SIZE):
            target_qpos[i] = actions[step_id + i]
        state = states[step_id : step_id + 1]


        def padding_state(value, actual, expected=0):
            # print(value.shape, actual, expected)
            if actual < expected:
                for i in range(1, expected - actual + 1 ):
                    value[-i] = value[actual - 1]
                    # value[-i] = zero_action
            return value
        
        target_qpos = padding_state(target_qpos, ACTUAL_CHUNK_SIZE, CHUNK_SIZE)
        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In our data: 3 translation, 6 rotation, 1 gripper
            UNI_STATE_INDICES = [ STATE_VEC_IDX_MAPPING['eef_pos_x']
                            ] + [ STATE_VEC_IDX_MAPPING['eef_pos_y']
                            ] + [ STATE_VEC_IDX_MAPPING['eef_pos_z']
                    ] + [
                        STATE_VEC_IDX_MAPPING[f"eef_angle_{i}"] for i in range(6)
                    ] + [
                        STATE_VEC_IDX_MAPPING["right_gripper_open"]
                    ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
            
        state_indicator = fill_in_state(np.ones_like(state[0]))
        state = fill_in_state(state)
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm = fill_in_state(state_norm)
        
        actions = fill_in_state(target_qpos)

        def unavailable_img():
            return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
        
        def parse_img(id:int):
            imgs = []

            if id > 0:
                for i in range(max(id - self.IMG_HISORY_SIZE + 1, 0), id + 1):
                    img = frames[i]
                    # img = img[:, 65:575, :]
                    imgs.append(img)
            else:
                img = frames[id]
                imgs.append(img)
            imgs = np.stack(imgs)
            
            if imgs.shape[0] < self.IMG_HISORY_SIZE:
                # Pad the image squence using the first image
                imgs = np.concatenate([
                    np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                    imgs
                    ], axis=0)
            
            return imgs
        
        cam_right_wrist = parse_img(step_id)
        valid_len = min(step_id - first_idx + 1, self.IMG_HISORY_SIZE)
        cam_right_wrist_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
        
        cam_left_wrist = unavailable_img()
        cam_left_wrist_mask = cam_right_wrist_mask.copy()

        cam_high = unavailable_img()
        cam_high_mask = cam_right_wrist_mask.copy()

        return True, {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_high_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_left_wrist_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask
        }

        

if __name__ == "__main__":
    ds = HDF5VLADataset()
    # ds = LerobotVLADataset()
    print(ds.episode_sample_weights)
    # weight_path = '/datassd_1T/dataset_cache/simpler_data/episode_sample_weights_test.npy'
    # episode_sample_weights = np.load(weight_path)
    
    # print(np.equal(ds.episode_sample_weights, episode_sample_weights))
    # print((ds.episode_sample_weights == episode_sample_weights).all())
    # diff_weights = ds.episode_sample_weights - episode_sample_weights
    # # absloute value of diff weights
    # diff_weights = np.abs(diff_weights)
    # print(np.max(diff_weights))
    # np.save(weight_path, ds.episode_sample_weights)
    # abnormal_shape =0
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        sample = ds.get_item(i)
        print(sample["state"].shape, sample["actions"].shape)
    #     print(sample['cam_right_wrist'].shape[0], sample['cam_right_wrist_mask'].shape[0], abnormal_shape)
    #     if (sample['cam_right_wrist'].shape[0] != 2) or (sample['cam_right_wrist_mask'].shape[0] != 2):
    #         abnormal_shape += 1
    # print(abnormal_shape)
        
