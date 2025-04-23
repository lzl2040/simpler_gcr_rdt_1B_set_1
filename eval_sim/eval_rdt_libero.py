from typing import Callable, List, Type
import sys
sys.path.append('/home/v-wangxiaofa/lzl/simpler_gcr_rdt_1B_set_1')
import numpy as np
import argparse
import yaml
from scripts.libero_model import create_model
import torch
from collections import deque
from PIL import Image
import cv2
import dataclasses
import logging
import pathlib
import tyro
import math
import tqdm
import collections
import imageio
import time
from scipy.spatial.transform import Rotation as R
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "visual/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    policy = get_model()
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        text_embed = policy.encode_instruction(task)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            # action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            policy.reset()
            obs_window = deque(maxlen=2)
            obs_window.append(None)
            # wrist_obs_window = deque(maxlen=2)
            # wrist_obs_window.append(None)

            done = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                # try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        if t == args.num_steps_wait:
                            # rotate 180 degrees to match train preprocessing
                            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            obs_window.append(img)
                            # wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            # wrist_obs_window.append(wrist_img)

                            proprio = np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            )
                        continue
                    image_arrs = []
                    # print("123")
                    for window_img in obs_window:
                        image_arrs.append(window_img)
                        image_arrs.append(None)
                        image_arrs.append(None)
                    # print("after")
                    images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]
                    # img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    # )
                    # wrist_img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    # )
                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    
                    proprio = get_ortho6d_from_euler_angle(proprio)
                    proprio = torch.from_numpy(proprio)
                    
                    actions = policy.step(proprio, images, text_embed) # chunk(20) 7

                    # actions = actions[::4, :]
                    actions = actions[:15]
                    for idx in range(actions.shape[0]):
                        action = actions[idx]
                        print(f"action->{action}")
                        obs, reward, done, info = env.step(action)
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        obs_window.append(img)
                        # wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        # wrist_obs_window.append(wrist_img)
                        
                        replay_images.append(img)
                        proprio = np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            )
                        if done:
                            break
                        t += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                # except Exception as e:
                #     logging.error(f"Caught exception: {e}")
                #     break

            task_episodes += 1
            total_episodes += 1

            if total_episodes % 5 == 0 or done:
                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, tip=""):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

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

def get_ortho6d_from_euler_angle(state):
    # print(state.shape)
    state = state.reshape(1, -1)
    xyz = state[:, :3]
    euler = state[:, 3:6]
    gripper = state[:, 6:7] # in fact, there are two gripper state, but we just select the first
    rot_mat = convert_euler_to_rotation_matrix(euler)
    orth6d = compute_ortho6d_from_rotation_matrix(rot_mat)
    new_state = np.concatenate((xyz, orth6d, gripper), axis=1)
    # new_state = new_state.reshape(-1)
    # print(new_state.shape)
    return new_state

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def get_model():
    config_path = 'configs/base.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    pretrained_text_encoder_name_or_path = "/Data/lzl/weights/rdt_param/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "/Data/lzl/weights/rdt_param/siglip-so400m-patch14-384"
    pretrained_path = "/Data/lzl/rdt-ft-simulated/0422-rdt-libero-all/checkpoint-90000/pytorch_model/mp_rank_00_model_states.pt"
    policy = create_model(
        args=config, 
        dtype=torch.bfloat16,
        pretrained=pretrained_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
    )
    return policy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)