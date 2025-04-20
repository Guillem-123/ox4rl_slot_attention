from ox4rl.utils.load_config import get_config_v2
import os.path as osp

# First load the default slot_atari_pong.yaml config
default_config_path = osp.join(osp.dirname(osp.dirname(__file__)), "configs", "slot_atari_pong.yaml")
cfg = get_config_v2(default_config_path)

from ox4rl.dataset import get_dataset, get_dataloader, Atari_Z_What

import math
from torch.nn.utils import clip_grad_norm_
from ox4rl.models.slot.utils import object_consistency_loss as oc_loss, motion_loss as motion_loss, \
    exponential_decay, plot_masks, temporal_loss
from ox4rl.models.slot.utils import adjusted_rand_index as ARI
from ox4rl.models.slot.utils import compute_ari, compute_mi_atari
from ox4rl.models.slot.model import SlotAttentionAutoEncoder

import os
import random
import argparse
import datetime
import numpy as np
import scipy
import time
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.functional as F
from tqdm import tqdm
from rtpt import RTPT
from torch.utils.tensorboard import SummaryWriter

from skimage.measure import block_reduce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_save_paths(opt, cfg):
    gamename = opt.config_file.split("_")[-1].replace(".yaml", "")
    opt.sample_dir = os.path.join(opt.model_dir, gamename, f"{opt.exp_name}_{opt.seed}", 'samples')
    opt.events_dir = os.path.join(opt.model_dir, gamename, f"{opt.exp_name}_{opt.seed}", 'events')
    opt.model_dir = os.path.join(opt.model_dir, gamename, f"{opt.exp_name}_{opt.seed}", 'ckpts')
    if not os.path.exists(opt.sample_dir):
        os.makedirs(opt.sample_dir)
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)
    if not os.path.exists(opt.events_dir):
        os.makedirs(opt.events_dir)


def update_cfg(opt, cfg):
    cfg.data_path = opt.data_path
    cfg.train.weight_mask = opt.weight_mask
    cfg.train.weight_oc = opt.weight_oc
    cfg.train.weight_temporal = opt.weight_temporal
    cfg.seed = opt.seed
    cfg.alt_oc = opt.alt_oc
    cfg.alt_oc_2 = opt.alt_oc_2


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)


def plot_masks_training(image, recon_combined, masks, masks_gt, step, cfg, writer=None):
    """
    Plots masks during training for visualization in TensorBoard.
    """
    batch_size = image.size(0)
    
    # Convert tensors to numpy for visualization
    image_np = image.detach().cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
    recon_np = recon_combined.detach().cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
    masks_np = masks.detach().cpu().permute(0, 2, 1, 3, 4).numpy().squeeze(2)
    
    # Process ground truth masks if available
    # masks_gt is a nested list with structure [[tensor, tensor, ...], [tensor, tensor, ...], ...]
    masks_gt_np = None
    if masks_gt is not None and len(masks_gt) > 0 and len(masks_gt[0]) > 0:
        # Just take the first ground truth mask from the first batch sample for visualization
        masks_gt_np = masks_gt[0][0].detach().cpu().numpy()
    
    # Draw only the first sample for simplicity
    n_slots = masks_np.shape[1]
    fig, axes = plt.subplots(1, 3 + n_slots, figsize=(3 + n_slots * 2, 4))
    
    # Original image
    axes[0].imshow(image_np[0])
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(recon_np[0])
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    # Ground truth mask if available
    if masks_gt_np is not None:
        axes[2].imshow(masks_gt_np, cmap='viridis')
        axes[2].set_title('GT Mask')
        axes[2].axis('off')
    else:
        axes[2].axis('off')
        axes[2].set_title('No GT Mask')
        
    # Individual masks
    for i in range(n_slots):
        axes[i+3].imshow(masks_np[0, i], cmap='viridis')
        axes[i+3].set_title(f'Slot {i+1}')
        axes[i+3].axis('off')
    
    plt.tight_layout()
    
    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_figure('training/masks', fig, step)
    
    plt.close(fig)


def main():
    opt = parser.parse_args()
    update_save_paths(opt, cfg)
    conf = cfg.merge_from_file(opt.config_file)
    update_cfg(opt, cfg)
    print(cfg.alt_oc)
    print(cfg.alt_oc_2)
    set_seed(cfg.seed)
    writer = SummaryWriter(opt.events_dir)

    dataset_subset_mode = "relevant"

    data_path = opt.data_path
    # import ipdb;ipdb.set_trace()
    dataset = Atari_Z_What(cfg, 'train', boxes_subset = dataset_subset_mode, return_keys=["imgs", "motion_slot"])
    train_dataloader = get_dataloader(cfg, 'train', dataset)
    val_set = Atari_Z_What(cfg, 'val', boxes_subset = dataset_subset_mode, return_keys=["imgs", "motion_slot", "motion_gt_labels"])

    model = SlotAttentionAutoEncoder(
        cfg.resolution, cfg.arch_slot.num_slots, cfg.arch_slot.hid_dim, output_channel=3, device=device.type).to(device)

    criterion = nn.MSELoss()

    params = [{'params': model.parameters()}]

    optimizer = optim.Adam(params, lr=cfg.train.solver.slot.lr)

    # Create RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name='SAMOC', max_iterations=cfg.train.max_epochs)

    val_dataloader = get_dataloader(cfg, 'val', val_set)
    cfg.dataset = 'ATARI-Mask-labels'
    val_dataloader_labels = get_dataloader(cfg, 'val', val_set)

    # Start the RTPT tracking
    rtpt.start()

    # coupled loss scheduler
    # l_scheduler = exponential_decay(1, 0.01, opt.num_epochs)
    # l_scheduler = 1. - np.arange(cfg.train.max_epochs/2 + 1) / cfg.train.max_epochs/2
    first_half = list(1. - np.arange(int(cfg.train.max_epochs / 2) + 1) / int(cfg.train.max_epochs / 2))
    second_half = list(np.zeros(int(cfg.train.max_epochs / 2)))
    l_scheduler = np.array(first_half + second_half)

    start = time.time()
    step = 0
    print('Model build finished!')
    model.eval()

    plot_masks(val_set, model, -1, cfg, opt, writer)
    print("Computing initial train ARI and MI")
    ari_m, ari_std = compute_ari(val_dataloader, model, cfg.train.batch_size, cfg.arch_slot.num_slots)
    print(f"INITIAL : Test ARI: {ari_m}")
    writer.add_scalar("ari_m", ari_m, -1)
    mi_score = compute_mi_atari(val_dataloader_labels, model, cfg)
    print(f"INITIAL : Test MI: {mi_score}")
    writer.add_scalar("test_mi", mi_score, -1)

    for epoch in range(cfg.train.max_epochs + 1):
        model.train()

        total_loss = 0
        R_loss = 0
        M_loss = 0
        OC_loss = 0
        T_loss = 0
        total_ari = 0
        print(f"Epoch {epoch}/{cfg.train.max_epochs + 1}")

        for data_dict in tqdm(train_dataloader, dynamic_ncols=True):
            step += 1

            img_stacks = data_dict["imgs"]
            masks_gt= data_dict["motion_slot"]  
            n_imgs = len(masks_gt[0])

            # if step < cfg.train.warmup_steps:
            #     learning_rate = cfg.train.solver.slot.lr * (step / cfg.train.warmup_steps)
            # else:
            learning_rate = cfg.train.solver.slot.lr

            # learning_rate = learning_rate * (cfg.train.decay_rate ** (
            #         step / cfg.train.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate

            image = img_stacks.to(device)
            masks_gt = [[masks_gt[i][j].to(device) for j in range(len(masks_gt[i]))]
                        for i in range(cfg.train.batch_size)
                        ]
            

            recon_combined, masks, _, slots = model(image)
            recon_combined = recon_combined.view(
                cfg.train.batch_size, 4, 3, cfg.resolution[0], cfg.resolution[1])

            # -------------------------------------------------------#
            # reconstruction loss
            loss = criterion(recon_combined, image)
            # -------------------------------------------------------#
            # motion loss
            loss_mask = motion_loss(masks, masks_gt, n_imgs=n_imgs, cfg=cfg)
            # -------------------------------------------------------#
            # object consistency loss
            loss_oc = oc_loss(slots, masks, n_imgs=n_imgs, cfg=cfg)
            if cfg.alt_oc_2:
                loss_oc *= -1
            # -------------------------------------------------------#
            # temporal persitency loss from Bao et al.
            loss_temporal = temporal_loss(slots)
            # -------------------------------------------------------#

            # (l_scheduler[epoch] * cfg.train.weight_mask) * loss_mask + \
            whole_loss = loss + \
                         (cfg.train.weight_mask) * loss_mask + \
                         ((1 - l_scheduler[epoch]) * cfg.train.weight_oc) * loss_oc + \
                         cfg.train.weight_temporal * loss_temporal

            optimizer.zero_grad()
            whole_loss.backward(retain_graph=True)
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()#whole_loss.item()
            R_loss += loss.item()
            M_loss += loss_mask.item()
            OC_loss += loss_oc.item()
            T_loss += loss_temporal.item()
            # total_ari += ARI(masks_gt, masks)

            if step % cfg.train.print_every == 0:
                print(f"Step: {step}, Loss: {loss.item():.6f}, "
                      f"Loss_mask: {loss_mask.item():.6f}, "
                      f"Loss_oc: {loss_oc.item():.6f}, "
                      f"Loss_temporal: {loss_temporal.item():.6f}, "
                      f"LR: {learning_rate:.6f}")
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_mask', loss_mask.item(), step)
                writer.add_scalar('train/loss_oc', loss_oc.item(), step)
                writer.add_scalar('train/loss_temporal', loss_temporal.item(), step)
                writer.add_scalar('train/lr', learning_rate, step)
                # plot_masks_training(image, recon_combined, masks, masks_gt, step, cfg, writer)

            del recon_combined, masks, image, loss, whole_loss, masks_gt, loss_mask, loss_oc, loss_temporal

        total_loss /= len(train_dataloader)
        R_loss /= len(train_dataloader)
        M_loss /= len(train_dataloader)
        OC_loss /= len(train_dataloader)
        T_loss /= len(train_dataloader)
        # total_ari /= len(train_dataloader)
        rtpt.step()
        dtime = datetime.timedelta(seconds=time.time() - start)
        print(f"Epoch: {epoch}, Loss: {total_loss}, Recon Loss: {R_loss}, Loss mask: {M_loss}, " + 
            f"Loss OC: {OC_loss}, Loss Temporal: {T_loss}, Time: {dtime}")


        writer.add_scalar("total_loss", total_loss, epoch)
        writer.add_scalar("recon_loss", R_loss, epoch)
        writer.add_scalar("mask_loss", M_loss, epoch)
        writer.add_scalar("oc_loss", OC_loss, epoch)
        writer.add_scalar("t_loss", T_loss, epoch)

        if epoch and not epoch % 10:
            model.eval()
            print("Computing test ARI and MI")
            ari_m, ari_std = compute_ari(val_dataloader, model, cfg.train.batch_size, cfg.arch_slot.num_slots)
            writer.add_scalar("ari_m", ari_m, epoch)
            print(f"INTERMEDIATE: Test ARI: {ari_m}")
            mi_score = compute_mi_atari(val_dataloader_labels, model, cfg)
            writer.add_scalar("test_mi", mi_score, epoch)
            print(f"INTERMEDIATE : Test MI: {mi_score}")
            torch.save({
                'model_state_dict': model.state_dict(),
            }, os.path.join(opt.model_dir, 'epoch_{}.ckpt'.format(epoch))
            )
            print("Saved ckpt in ", os.path.join(opt.model_dir, 'epoch_{}.ckpt'.format(epoch)))
            plot_masks(val_set, model, epoch, cfg, opt, writer)

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(opt.model_dir, 'epoch_{}_final.ckpt'.format(epoch))
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument
    # basic configurations
    parser.add_argument('--config_file', help='path to config file', required=True,
                        type=str)
    parser.add_argument('--task', default='train', type=str)
    parser.add_argument('--model_dir', default='./runs/',
                        type=str, help='where to save models')
    parser.add_argument('--exp_name', default='', type=str,
                        help='experiment name, used for model saving/plotting/wand ect')
    # parser.add_argument('--proj_name', default='my-project',
    #                     type=str, help='wandb project name')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data_path', default='/mnt/fsx/pd_v2',
                        type=str, help='path of KITTI dataset')
    parser.add_argument('--alt_oc',  action='store_true', default=False,
                        help='alternative oc loss idea')
    parser.add_argument('--alt_oc_2',  action='store_true', default=False,
                        help='alternative oc loss idea')
    # model parameters
    parser.add_argument('--weight_mask', default=1.0,
                        type=float, help='weight for the mask loss')
    parser.add_argument('--weight_oc', default=1.0,
                        type=float, help='weight for the object consistency loss')
    parser.add_argument('--weight_temporal', default=1.0,
                        type=float, help='weight for the temporal loss')
    parser.add_argument('--wandb', default=False, type=bool)
    parser.add_argument('--entity', default='zpbao', type=str, help='wandb name')

    torch.autograd.set_detect_anomaly(True)
    main()
