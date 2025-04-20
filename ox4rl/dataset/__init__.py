from .atari_dataset import Atari_Z_What
from torch.utils.data import DataLoader
from .atari_labels import label_list_for


__all__ = ['get_dataset', 'get_dataloader', 'get_label_list']


def get_dataset(cfg, dataset_mode):
    assert dataset_mode in ['train', 'validation', 'test']
    if cfg.dataset == 'ATARI':
        return Atari_Z_What(cfg, dataset_mode)  # , return_keys=["z_whats_pres_s", "gt_labels_for_pred_boxes", "pred_boxes", "gt_bbs_and_labels"]
    #elif cfg.dataset == 'OBJ3D_SMALL':
    #    return Obj3D(cfg.dataset_roots.OBJ3D_SMALL, dataset_mode)
    #elif cfg.dataset == 'OBJ3D_LARGE':
    #    return Obj3D(cfg.dataset_roots.OBJ3D_LARGE, dataset_mode)


def get_dataloader(cfg, dataset_mode, dataset, no_shuffle_overwrite = False):
    assert dataset_mode in ['train', 'validation', 'test']

    # get batch size and num_workers from config, value specified for train is always used
    batch_size = getattr(cfg, 'train').batch_size
    num_workers = getattr(cfg, 'train').num_workers

    shuffle = True if dataset_mode == 'train' else False
    if no_shuffle_overwrite:
        shuffle = False

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


def get_label_list(cfg):
    game = cfg.gamelist[0]
    return label_list_for(game)
