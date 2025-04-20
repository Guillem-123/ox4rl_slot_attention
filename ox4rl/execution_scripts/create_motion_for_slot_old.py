
from skimage.measure import block_reduce
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import label
from ox4rl.dataset.atari_dataset import Atari_Z_What
from ox4rl.dataset import get_dataloader
from ox4rl.utils.load_config import get_config_v2
import argparse
import os



def segment_connected_areas(tensor):

    CONSTANT_NUMBER_OF_FEATURES = 10
    # Ensure tensor is a binary tensor (0s and 1s)
    tensor = tensor.astype(bool)
    
    # Label connected components
    labeled_array, num_features = label(tensor)
    
    # Create an array to store the segmented tensors
    segmented_tensors = np.zeros((CONSTANT_NUMBER_OF_FEATURES, *tensor.shape), dtype=int)
    
    # For each connected component, extract the region
    for i in range(1, max(CONSTANT_NUMBER_OF_FEATURES, num_features) + 1):
        # Create a mask for the current connected component
        mask = (labeled_array == i)
        
        # Extract the region as a 32x32 tensor
        segmented_tensors[i-1] = mask.astype(int)
    
    return segmented_tensors

def main(cfg, dataset_mode):
    dataset = Atari_Z_What(cfg, dataset_mode, return_keys=["imgs", "motion_slot"])
    train_dataloader = get_dataloader(cfg, dataset_mode, dataset)
    for (index, data_dict) in tqdm(enumerate(train_dataloader)):
        masks_gt_prel= data_dict["motion"]
        masks_gt_prel = block_reduce(masks_gt_prel.numpy(), block_size=(1, 1, 4, 4), func=np.max)
        # separate masks_gt into individual masks
        masks_gt = np.zeros((masks_gt_prel.shape[0], masks_gt_prel.shape[1], 10, masks_gt_prel.shape[2], masks_gt_prel.shape[3]))
        for i in range(masks_gt.shape[0]):
            for j in range(masks_gt.shape[1]):
                masks_gt[i][j] = segment_connected_areas(masks_gt_prel[i][j])
        masks_gt = torch.from_numpy(masks_gt).float()
        n_imgs = len(masks_gt[0])
        # save as pt file, save each motion mask as a separate file (masks_gt.shape = torch.Size([1, 4, 10, 32, 32])
        motion_path = dataset.motion_path
        for i in range(n_imgs):
            # index with 5 digits, i with 1 digit
            file_name = f"{index:05}_{i}_slot.pt"
            base_path = motion_path
            torch.save(masks_gt[0, i], os.path.join(base_path, file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='path to config file', required=True,
                        type=str)
    parser.add_argument('--dataset_mode', help='train or val', required=True,
                        type=str)
    config_path = parser.parse_args().config_file
    cfg = get_config_v2(config_path)
    dataset_mode = parser.parse_args().dataset_mode
    main(cfg, dataset_mode)