import os

from ox4rl.models.space.postprocess_latent_variables import retrieve_latent_repr_from_logs
from ox4rl.training.checkpointing.checkpointer import Checkpointer

import os.path as osp
from ox4rl.utils.load_config import get_config_v2
from ox4rl.models import get_model
import os
import torch
from torch._prims_common import check
from tqdm import tqdm
from ox4rl.dataset import get_dataloader
from ox4rl.dataset.atari_dataset import Atari_Z_What
from ox4rl.models.slot.model import SlotAttentionAutoEncoder

def create_latent_dataset(cfg, dataset_mode, model=None):
    game = cfg.gamelist[0]

    # if model is None:
    #     model = get_model(cfg)
    #     model = model.to(cfg.device)
    #     checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    #     ckpt_path = cfg.resume_ckpt
    #     print(ckpt_path)
    #     checkpointer.load_last(cfg.resume, ckpt_path, model, None, None, cfg.device)

    # use trained model given as parameter or load based on config
    device = torch.device(cfg.device)
    model = SlotAttentionAutoEncoder(cfg.resolution, cfg.arch_slot.num_slots, cfg.arch_slot.hid_dim, output_channel=3, device=device.type)

    # Determine checkpoint path based on environment
    if os.environ.get('RUNNING_IN_COLAB') == 'true':
        checkpoint_path = '/content/content/ox4rl/epoch_10000_final.ckpt'
        print("Running in Colab environment. Using Colab checkpoint path.")
    else:
        # Assuming the checkpoint is in the project root directory when not in Colab
        checkpoint_path = 'epoch_10000_final.ckpt'
        print("Running in local environment. Using local checkpoint path.")

    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Make sure it exists or set the RUNNING_IN_COLAB environment variable correctly.")

    checkpoint=torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(dataset_mode)
    # Convert 'val' to 'validation' to match Atari_Z_What class behavior
    dataset_mode_path = 'validation' if dataset_mode == 'val' else dataset_mode
    base_path = osp.join(cfg.dataset_roots.ATARI, game, "latents_slot", dataset_mode_path)
    os.makedirs(base_path, exist_ok=True)

    create_latent_dataset_with_more_options(cfg, dataset_mode_path, model, base_path)

def create_latent_dataset_with_more_options(cfg, dataset_mode, model, base_path, number_of_data_points = None):
    game = cfg.gamelist[0]
    dataset = Atari_Z_What(cfg, dataset_mode, return_keys = ["imgs"])
    dataloader = get_dataloader(cfg, dataset_mode, dataset, no_shuffle_overwrite=True)

    # Convert 'val' to 'validation' to match Atari_Z_What class behavior
    dataset_mode_path = 'validation' if dataset_mode == 'val' else dataset_mode
    base_path = osp.join(cfg.dataset_roots.ATARI, game, "latents_slot", dataset_mode_path)
    os.makedirs(base_path, exist_ok=True)

    B = dataloader.batch_size
    T = dataset.T

    print("start inference and saving")
    for i, data_dict in enumerate(tqdm(dataloader)):
        if number_of_data_points is not None and i >= number_of_data_points:
            break

        image_tensor = data_dict["imgs"]
        image_tensor = image_tensor.to(cfg.device)
        with torch.no_grad():
            recon_combined, masks, _, slots = model(image_tensor)

        # combine first two dimensions
        masks = masks.view(-1, *masks.shape[2:])
        slots = slots.view(-1, *slots.shape[2:])
        infix = f"0to{T-1}"
        for b in range(B):
            batch_index = i * B + b
            index = b * T
            torch.save(masks[index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_attn_masks.pt"))    
            torch.save(slots[index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_slot_repr.pt"))
    print("Finished inference and saving")

    # print("start inference")
    # for i, data_dict in enumerate(tqdm(dataloader)):
    #     if number_of_data_points is not None and i >= number_of_data_points:
    #         break

    #     image_tensor = data_dict["imgs"]
    #     image_tensor = image_tensor.to(cfg.device)
    #     with torch.no_grad():
    #         #_ , space_log = model(image_tensor, global_step=global_step)
    #         space_log = model(image_tensor)
    #     # (B*T, N, 4), (B*T, N,), (B*T, N), (B*T, N, 32)
    #     z_where, _, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)

    #     z_where = z_where.to(cfg.device)
    #     z_pres_prob = z_pres_prob.to(cfg.device)
    #     z_what = z_what.to(cfg.device)

    #     z_wheres.append(z_where)
    #     z_pres_probs.append(z_pres_prob)
    #     z_whats.append(z_what)
    # print("finished inference")

    # print("start saving")
    # infix = f"0to{T-1}" if T > 1 else "0"
    # for i in range(len(z_wheres)):
    #     for b in range(B):
    #         batch_index = i * B + b
    #         index = b * T
    #         torch.save(z_wheres[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_where.pt"))    
    #         torch.save(z_pres_probs[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_pres_prob.pt"))
    #         torch.save(z_whats[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_what.pt"))
    # print("Finished saving")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )
    parser.add_argument(
        '--dataset_mode',
        type=str,
        default='test',
        choices=['train', 'validation', 'test'],
        help='Dataset mode: train, validation, or test'
    )
    args = parser.parse_args()
    config_path = args.config_file
    dataset_mode = args.dataset_mode
    cfg = get_config_v2(config_path)
    create_latent_dataset(cfg, dataset_mode)