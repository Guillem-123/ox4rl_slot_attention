import os
from ox4rl.vis.object_detection.space_vis import draw_image_bb
from ox4rl.models.space.space import Space
import torch
from PIL import Image
import torchvision.transforms as transforms
from ox4rl.dataset_creation.create_dataset_using_OCAtari import BaseDetectionModel, OCAtariDatasetCreator
import argparse
import ipdb
from ox4rl.training.checkpointing.loading import load_model
from ox4rl.utils.load_config import get_config_v2
from torch.utils.data import DataLoader

# Load configuration
config_path = "../configs/my_atari_pong.yaml"
cfg = get_config_v2(config_path)

# Load the model
model, _, _, _, _ = load_model(cfg, mode="eval")

# Load the dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the dataset for a specific game of ATARI Gym')
    parser.add_argument('-g', '--game', type=str, help='An atari game', default='Pong')
    parser.add_argument('--compute_root_images', default=False, action="store_true", help='instead compute the mode of the images')
    parser.add_argument('--no_color_hist', default=False, action="store_true", help='use the color_hist to filter')
    parser.add_argument('--render', default=False, action="store_true", help='renders the environment')
    parser.add_argument('-s', '--stacks', default=True, action="store_false", help='should render in correlated stacks of T (defined in code) frames')
    parser.add_argument('--bb', default=True, action="store_false", help='should compute bounding_boxes')
    parser.add_argument('-f', '--folder', type=str, choices=["train", "test", "validation"], required=True, help='folder to write to: train, test or validation')
    parser.add_argument('--vis', default=False, action="store_true", help='creates folder vis with visualizations which can be used for debugging')
    parser.add_argument('-m', '--model', type=str, choices=["SPACE", "SLOT"], required=True, help='Name of base detection model.')
    args = parser.parse_args()

# Throws an error
#  Traceback (most recent call last):
#   File "/Users/anshulchahar/Documents/GitHub/ox4rl/ox4rl/tests/draw_image_bb.test.py", line 35, in <module>
#     dataset_creator = OCAtariDatasetCreator(
#   File "/Users/anshulchahar/Documents/GitHub/ox4rl/ox4rl/dataset_creation/create_dataset_using_OCAtari.py", line 82, in __init__
#     self._agent, _, _ = self._configure_OCAtari_environment()
#   File "/Users/anshulchahar/Documents/GitHub/ox4rl/ox4rl/dataset_creation/create_dataset_using_OCAtari.py", line 356, in _configure_OCAtari_environment
#     0 if args.folder == "train" else 1 if args.folder == "validation" else 2,
# NameError: name 'args' is not defined
    dataset_creator = OCAtariDatasetCreator(
        directory=args.folder,
        game_str=args.game,
        compute_root_images=args.compute_root_images,
        no_color_hist=args.no_color_hist,
        render=args.render,
        stacks=args.stacks,
        compute_bounding_boxes=args.bb,
        vis=args.vis,
        base_detection_model=BaseDetectionModel[args.model]
    )
dataset = dataset_creator._create_dataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Fetch a batch of data
data_dict = next(iter(dataloader))
images = data_dict["imgs"]
motion_z_pres = data_dict.get("motion_z_pres", None) 
motion_z_where = data_dict.get("motion_z_where", None)  

# Call the draw_image_bb function
global_step = 1000
num_batch = 4  # Number of images to process, should match batch_size
output_images = draw_image_bb(model, cfg, dataset, global_step, num_batch)

# Save the output images
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

for i, img_tensor in enumerate(output_images):
    output_path = os.path.join(output_dir, f"batch_{i}_bb.png")
    img = transforms.ToPILImage()(img_tensor)  # Convert tensor to PIL image
    img.save(output_path)
    print(f"Image with bounding boxes saved to: {output_path}")