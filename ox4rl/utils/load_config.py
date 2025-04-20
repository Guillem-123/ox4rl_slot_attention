import os
import argparse
from argparse import ArgumentParser

from ox4rl.configs.config import cfg



def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        metavar='TASK',
        help='What to do. See engine'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )

    parser.add_argument(
        '--resume_ckpt',
        help='Provide a checkpoint to restart training from',
        default=''
    )

    parser.add_argument(
        "--eval_ckpt",
        help='Provide a chekpoint to restart evaluation from',
        default=''
    )

    parser.add_argument(
        '--arch-type',
        help='architecture type',
        choices=['baseline', '+m', '+moc'],
        default= "+moc",
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )
    # example usage for opts: python main.py moc_cfg.area_object_weight 0.0 moc_cfg.motion_input False

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Use config file name as the default experiment name
    if cfg.exp_name == '':
        if args.config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')
    
    # set moc_cfg parameters for moc loss (only relevant for training)
    if args.arch_type == 'baseline':
        cfg.moc_cfg.area_object_weight = 0.0
        cfg.moc_cfg.motion_weight = 0.0
    elif args.arch_type == "+m": #
        cfg.moc_cfg.area_object_weight = 0.0
    elif args.arch_type == "+moc": #
        cfg.moc_cfg.area_object_weight = 10.0
    else:
        print(f"arch_type {args.arch_type} must be one of ['baseline', '+m', '+moc']")
        raise NotImplementedError

    if args.resume_ckpt != '':
        cfg.resume_ckpt = args.resume_ckpt

    import torch
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(cfg.seed)

    return cfg, args.task


def get_config_v2(config_path):
    cfg.merge_from_file(config_path)
    return cfg

def get_config_for_game(game):
    cfg_path_for_game = {
        "pong": "configs/my_atari_pong_gpu.yaml",
        "boxing": "configs/my_atari_boxing_gpu.yaml",
        "skiing": "configs/my_atari_skiing_gpu.yaml",
    }
    cfg = get_config_v2(cfg_path_for_game[game])
    return cfg