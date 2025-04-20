from torch import nn
import os.path as osp
from ox4rl.models import get_model
from ox4rl.training.optimizers import get_optimizers
from ox4rl.training.checkpointing.checkpointer import Checkpointer

import joblib
import pandas as pd

def print_info(cfg):
    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    print('Resume checkpoint (only potentially relevant for training):', cfg.resume_ckpt)
    print('Evaluation checkpoint (not relevant for the training script):', cfg.eval.eval_ckpt)
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)
    suffix = cfg.gamelist[0]
    print(f"Using Game {suffix}")


def load_model(cfg, mode):
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,)
    checkpoint = None

    optimizer_fg, optimizer_bg = None, None
    if mode == "eval":
        if cfg.eval.eval_ckpt: 
            checkpoint = checkpointer.load(cfg.eval.eval_ckpt, model, None, None, cfg.device)
        elif cfg.eval.checkpoint == 'last':
            checkpoint = checkpointer.load_last(model, None, None, cfg.device)
        elif cfg.eval.checkpoint == 'best':
            checkpoint = checkpointer.load_best(cfg.eval.metric, model, None, None, cfg.device)
    elif mode == "train":
        optimizer_fg, optimizer_bg = get_optimizers(cfg, model)
        if cfg.resume:
            print(f"Loading checkpoint: {cfg.resume_ckpt if cfg.resume_ckpt else 'last checkpoint'}")
            if cfg.resume_ckpt:
                checkpoint = checkpointer.load(cfg.resume_ckpt, model, optimizer_fg, optimizer_bg, cfg.device)
            else:
                checkpoint = checkpointer.load_last(model, optimizer_fg, optimizer_bg, cfg.device)
        else:
            print("No checkpoint loaded. Training from scratch.")
            checkpoint = None
    else:
        raise ValueError(f"Invalid mode for loading model: {mode}")

    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)

    return model, optimizer_fg, optimizer_bg, checkpointer, checkpoint

def load_classifier(folder_path , clf_name="kmeans", data_subset_mode="relevant"):
    classifier_path = f"{folder_path}/z_what-classifier_{data_subset_mode}_{clf_name}.joblib.pkl"
    classifier = joblib.load(classifier_path)

    centroid_labels_dict = None
    if clf_name == "kmeans":
        centroid_labels_path = f"{folder_path}/z_what-classifier_{data_subset_mode}_{clf_name}_centroid_labels.csv"
        centroid_labels = pd.read_csv(centroid_labels_path, header=None, index_col=0)
        centroid_labels_dict = centroid_labels.iloc[:,0].to_dict()

    return classifier, centroid_labels_dict