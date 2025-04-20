import os
from ox4rl.vis.object_detection.space_vis import label_bb
from ox4rl.models.space.space import Space
import torch
from PIL import Image
import torchvision.transforms as transforms
from ox4rl.training.checkpointing.loading import load_model
from ox4rl.utils.load_config import get_config_v2
import argparse

# Load configuration and model
config_path = "../configs/my_atari_pong.yaml"
cfg = get_config_v2(config_path)
model, _, _, _, _ = load_model(cfg, mode="eval")

# Load the test image
image_path = "input/0000_0.png"
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    exit()
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)

# Call the label_bb function and save the labeled image
output_path = "output/0000_0_label_bb.png"
label_bb(model, image_tensor, path=output_path, device="cpu")
print(f"Labeled image with bounding boxes saved to: {output_path}")