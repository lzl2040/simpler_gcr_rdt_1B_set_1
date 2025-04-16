import os

import torch
import yaml
import sys
sys.path.append("/home/v-wangxiaofa/lzl/simpler_gcr_rdt_1B_set_1")

from models.multimodal_encoder.t5_encoder import T5Embedder
import json

GPU = 3
MODEL_PATH = "/Data/lzl/weights/rdt_param/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"

# Modify this to your task name and instruction
# TASK_NAME = "handover_pan"
lerobot_data_root = "/Data/lerobot_data/simulated/libero_goal_no_noops_lerobot"
task_path = os.path.join(lerobot_data_root, "meta", "tasks.jsonl")
SAVE_DIR = os.path.join(lerobot_data_root, "task_embeddings")
os.makedirs(SAVE_DIR, exist_ok=True)
INSTRUCTION = "Pick up the black marker on the right and put it into the packaging box on the left."

# Note: if your GPU VRAM is less than 24GB, 
# it is recommended to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    with open(task_path, "r") as f:
        for line in f:
            task = json.loads(line)
            print(task)
            task_index = task["task_index"]
            INSTRUCTION = task["task"]
            tokens = tokenizer(
                INSTRUCTION, return_tensors="pt",
                padding="longest",
                truncation=True
            )["input_ids"].to(device)

            tokens = tokens.view(1, -1)
            with torch.no_grad():
                pred = text_encoder(tokens).last_hidden_state.detach().cpu()
    
            save_path = os.path.join(SAVE_DIR, f"task_{task_index}.pt")
            # We save the embeddings in a dictionary format
            torch.save(pred, save_path)
            
            print(f'\"{INSTRUCTION}\" is encoded by \"{MODEL_PATH}\" into shape {pred.shape} and saved to \"{save_path}\"')


if __name__ == "__main__":
    main()
