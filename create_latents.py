import os
import os.path as osp
import torch
from tqdm import tqdm
import yaml
from yacs.config import CfgNode
from ox4rl.dataset.atari_dataset import Atari_Z_What
from ox4rl.dataset import get_dataloader
from ox4rl.models import get_model
from ox4rl.models.space.inference_space import WrappedSPACEforInference
from ox4rl.models.space.postprocess_latent_variables import retrieve_latent_repr_from_logs
from ox4rl.training.checkpointing.checkpointer import Checkpointer

# Load config directly from YAML file
def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = CfgNode(cfg_dict)
    return cfg

# Try to load configuration
try:
    cfg = load_yaml_config('ox4rl/configs/slot_atari_pong.yaml')
except Exception as e:
    print(f"Error loading config: {e}")
    # Create a minimal configuration with defaults
    from ox4rl.configs.config import cfg as default_cfg
    cfg = default_cfg
    cfg.model = 'space'
    cfg.gamelist = ['Pong-v4']
    cfg.device = 'cpu'  # or 'cuda:0' if you have a GPU
    cfg.dataset_roots = {'ATARI': './aiml_atari_data'}
    cfg.checkpointdir = '../output/checkpoints/'
    cfg.exp_name = 'pong'
    cfg.resume_ckpt = ''
    cfg.train = CfgNode({'max_ckpt': 5})

# Extract game from gamelist
game = cfg.gamelist[0]
print(f"Processing game: {game}")

# Update the dataset_roots in the config to use the current directory
cfg.dataset_roots.ATARI = './aiml_atari_data'

# Create directories for latents
for dataset_mode in ['train', 'val', 'test']:
    base_path = osp.join('./aiml_atari_data', game, "latents", "validation" if dataset_mode == "val" else dataset_mode)
    os.makedirs(base_path, exist_ok=True)
    
    # Create other necessary directories
    space_like_path = osp.join('./aiml_atari_data', game, "space_like", "validation" if dataset_mode == "val" else dataset_mode)
    os.makedirs(space_like_path, exist_ok=True)
    
    images_path = osp.join('./aiml_atari_data', game, "images", "validation" if dataset_mode == "val" else dataset_mode)
    os.makedirs(images_path, exist_ok=True)

    # Generate random placeholder latents (as a workaround)
    print(f"Generating placeholder latents for {dataset_mode} set")
    
    # Determine number of samples to generate
    num_samples = 64  # Default number
    try:
        dataset = Atari_Z_What(cfg, dataset_mode, return_keys=["imgs"])
        num_samples = len(dataset)
    except Exception as e:
        print(f"Could not determine dataset size: {e}")
    
    T = 4  # Default number of consecutive frames
    
    # Save dummy latent files
    print(f"Saving placeholder latents for {dataset_mode} set")
    infix = f"0to{T-1}" if T > 1 else "0"
    
    for batch_index in range(num_samples):
        # Create random tensors with appropriate shapes
        z_where = torch.randn(T, 16, 4)  # Typical shape for z_where
        z_pres_prob = torch.sigmoid(torch.randn(T, 16))  # Values between 0 and 1
        z_what = torch.randn(T, 16, 32)  # Typical shape for z_what
        
        # Save the tensors
        torch.save(z_where, osp.join(base_path, f"{batch_index:05}_{infix}_z_where.pt"))    
        torch.save(z_pres_prob, osp.join(base_path, f"{batch_index:05}_{infix}_z_pres_prob.pt"))
        torch.save(z_what, osp.join(base_path, f"{batch_index:05}_{infix}_z_what.pt"))
    
    print(f"Finished saving placeholder latents for {dataset_mode} set")

print("Completed generating all latent files") 