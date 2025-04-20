from ox4rl.execution_scripts.eval_model_and_classifier import eval_model_and_classifier
from ox4rl.execution_scripts.eval_classifier import eval_classifier
from ox4rl.execution_scripts.train_classifier import train_classifier
from ox4rl.utils.load_config import get_config_v2
import numpy as np
from ox4rl.execution_scripts.eval_model import eval
import os
from ox4rl.vis.object_detection.space_vis import class_label_bb, draw_image_bb_v2, draw_image_bb_v2_slot
from ox4rl.training.checkpointing.loading import load_model
from ox4rl.dataset.atari_dataset import Atari_Z_What
from ox4rl.dataset import get_dataloader
import joblib
from ox4rl.training.checkpointing.loading import load_classifier
from ox4rl.dataset.atari_data_collector import AtariDataCollector

# Determine base path based on environment
if os.environ.get('RUNNING_IN_COLAB') == 'true':
    base_path = '/content/content/ox4rl'
    print("Running in Colab environment. Using Colab base path.")
else:
    # Assuming the script is run from the project root or the required paths are relative to it
    base_path = '.' 
    print("Running in local environment. Using local base path.")


cfg_path_for_game = {
    "pong": os.path.join(base_path, "ox4rl/configs/slot_atari_pong.yaml"),
	  # "skiing": os.path.join(base_path, "configs/my_atari_skiing_gpu.yaml"),
    #"seaquest": os.path.join(base_path, "configs/my_atari_seaquest_gpu.yaml"),
    # "asterix": os.path.join(base_path, "configs/my_atari_asterix_gpu.yaml"),
    # "boxing": os.path.join(base_path, "configs/my_atari_boxing_gpu.yaml"),
    # "freeway": os.path.join(base_path, "configs/my_atari_freeway_gpu.yaml"),
    # "kangaroo": os.path.join(base_path, "configs/my_atari_kangaroo_gpu.yaml"),
    # "bowling": os.path.join(base_path, "configs/my_atari_bowling_gpu.yaml"),
    #"tennis": os.path.join(base_path, "configs/my_atari_tennis_gpu.yaml"),
}
for game in cfg_path_for_game.keys():
    config_file_path = cfg_path_for_game[game]
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found at {config_file_path}. Make sure it exists or set the RUNNING_IN_COLAB environment variable correctly.")
        
    cfg = get_config_v2(config_file_path)
    cfg_override_list = [
        "exp_name", f"{game}",
        "eval.checkpoint", "last",
        "train.num_workers", 0,
        "device", "cpu",
        "train.batch_size", 1,
    ]
    
    cfg.merge_from_list(cfg_override_list)

    #dataset = Atari_Z_What(cfg, 'validation', return_keys=["imgs", "gt_bbs_and_labels", "pred_boxes", "z_whats_pres_s"], boxes_subset="relevant")
    dataset =  Atari_Z_What(cfg, 'validation', return_keys=["imgs", "slot_latents", "slot_latents_labels"], boxes_subset="relevant")
    slot_attn_masks= AtariDataCollector.collect_slot_masks(cfg, 'validation', 'relevant')
    train_data_loader = get_dataloader(cfg, 'validation', dataset)
    
    result_imgs=draw_image_bb_v2_slot(cfg, train_data_loader, game, slot_attn_masks)





