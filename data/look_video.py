import torchvision
import importlib
import logging
import numpy as np
def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"
video_path = "/Data/lerobot_data/simulated/libero_spatial_no_noops_lerobot/videos/chunk-000/observation.images.image/episode_000414.mp4"

backend = get_safe_default_codec()
keyframes_only = False
torchvision.set_video_backend(backend)
if backend == "pyav":
    keyframes_only = True  # pyav doesnt support accuracte seek

# set a video stream reader
# TODO(rcadene): also load audio stream at the same time
reader = torchvision.io.VideoReader(video_path, "video")
loaded_frames = []
loaded_ts = []
for frame in reader:
    current_ts = frame["pts"]
    logging.info(f"frame loaded at timestamp={current_ts:.4f}")
    f_data = frame["data"].permute(1, 2, 0).numpy()
    # print(np.max(f_data))

    loaded_frames.append(frame["data"])
    loaded_ts.append(current_ts)
