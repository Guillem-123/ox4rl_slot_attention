import torch
from ox4rl.dataset.atari_dataset import Atari_Z_What
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from ox4rl.dataset.bbox_filter import BBoxFilter, filter_z_whats_with_valid_boxes
import ipdb

class AtariDataCollector:

    return_keys_with_list_return_value = ["z_whats_pres_s", "gt_labels_for_pred_boxes", "pred_boxes", "gt_bbs_and_labels"]
    return_keys_with_list_return_value_slot = ["slot_latents", "slot_latents_labels"]


    def __init__(self):
        pass
    
    @staticmethod
    def collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys, only_collect_first_image_of_consecutive_frames):    
        atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, data_subset_mode, return_keys=return_keys)
        atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
        result_dict = defaultdict(list)
        T = atari_z_what_dataset.T

        if only_collect_first_image_of_consecutive_frames:
            T = 1
        for batch in atari_z_what_dataloader:
            for return_key in return_keys:
                if return_key in AtariDataCollector.return_keys_with_list_return_value:
                    value_for_current_return_key = batch[return_key]
                    tmp_value = [value_for_current_return_key[i][0] for i in range(T)] #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
                    result_dict[return_key].extend(tmp_value)
                else:
                    result_dict[return_key].extend(batch[return_key][0][0:T]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension

        return result_dict
    
    #convencience methods
    @staticmethod
    def collect_z_what_data_reshaped(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):        
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["z_whats_pres_s", "gt_labels_for_pred_boxes"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        z_whats = torch.cat(result_dict["z_whats_pres_s"], dim=0)
        labels = torch.cat(result_dict["gt_labels_for_pred_boxes"], dim=0)
        labels = labels.squeeze(-1)
        return z_whats, labels
    
    @staticmethod
    def collect_z_what_data(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["z_whats_pres_s"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        return result_dict["z_whats_pres_s"], result_dict["gt_labels_for_pred_boxes"]
    
    @staticmethod
    def collect_images(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["imgs"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        images = torch.stack(result_dict["imgs"], dim=0)
        return images
    
    @staticmethod
    def collect_pred_boxes(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["pred_boxes"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        pred_boxes = result_dict["pred_boxes"]
        return pred_boxes

    @staticmethod
    def collect_bbox_data(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["pred_boxes", "gt_bbs_and_labels"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        pred_boxes = [np.array(pred_boxes) for pred_boxes in result_dict["pred_boxes"]]
        gt_boxes = [np.array(gt_boxes) for gt_boxes in result_dict["gt_bbs_and_labels"]]
        return gt_boxes, pred_boxes

    @staticmethod
    def collect_z_what_data_with_filtering(cfg, dataset_mode, data_subset_mode,
                                           only_collect_first_image_of_consecutive_frames=False):
        """
        Collect z_what data while filtering based on bounding box overlap quality.
        """
        # Get ground truth and predicted boxes
        gt_boxes, pred_boxes = AtariDataCollector.collect_bbox_data(cfg, dataset_mode, data_subset_mode,
                                                      only_collect_first_image_of_consecutive_frames)

        # Get original z_what features and labels
        z_whats, labels = AtariDataCollector.collect_z_what_data(cfg, dataset_mode, data_subset_mode,
                                                                 only_collect_first_image_of_consecutive_frames)

        # Initialize and apply filtering
        bbox_filter = BBoxFilter(iou_threshold=cfg.get('bbox_filter_threshold', 0.5))
        valid_indices, filtered_boxes, matched_gt_info = bbox_filter.filter_predictions(pred_boxes, gt_boxes)

        # Filter z_what features to only include those with good localization
        filtered_z_whats = filter_z_whats_with_valid_boxes(z_whats, valid_indices)

        # Extract labels from matched ground truth info
        filtered_labels = [int(label_info[1].split()[0]) for label_info in matched_gt_info]

        return filtered_z_whats, torch.tensor(filtered_labels)
    
    @staticmethod
    def collect_slot_attention_data(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        """
        Collects slot attention data (slot latents + labels) for classification.

        Args:
        - cfg: Configuration object.
        - dataset_mode: 'train' or 'test'.
        - data_subset_mode: 'all' or 'relevant' labels.
        - only_collect_first_image_of_consecutive_frames (bool): If True, collects only the first frame per sequence.

        Returns:
        - slot_latents (torch.Tensor): Tensor of slot representations.
        - slot_latents_labels (torch.Tensor): Corresponding labels.
        """

        # bug probably here!!!!
        # TODO fix this

        return_keys = ["imgs", "slot_latents", "slot_latents_labels"]
        atari_slot_labels_dataset = Atari_Z_What(cfg, dataset_mode, data_subset_mode, return_keys=return_keys)
        atari_slot_labels_dataloader = DataLoader(atari_slot_labels_dataset, batch_size=1, shuffle=False, num_workers=0)

        result_dict = defaultdict(list)
        T = atari_slot_labels_dataset.T

        if only_collect_first_image_of_consecutive_frames:
            T = 1  # Use only the first frame

        for batch in atari_slot_labels_dataloader:
            for key in return_keys:
                if key in AtariDataCollector.return_keys_with_list_return_value_slot:
                    print(key)
                   
                    value_for_current_key = batch[key] # list of 4 tensors for slot_latents and lis of 1 tensor for labels
                    #ipdb.set_trace()
                    if key == "slot_latents":
                        tmp_value = [value_for_current_key[i][0] for i in range(T)]  # Take only `T` time steps
                    else:
                        tmp_value = [value_for_current_key[0][0] for i in range(T)]  # Take only `T` time steps
                   
                    result_dict[key].extend(tmp_value)
                   
                else:
                    result_dict[key].extend(batch[key][0][0:T])  # Ensure time step consistency
        

        # default dict has 3 keys imgs, slot_latents and slot_latents_labels with len 256, 256, 256
        # so now we have two tensors with 
        # size 256, 10 for latents_labels
        # and size 256, 10, 64 for latents
        print(f"Before Fix: slot_latents = {len(result_dict['slot_latents'])}, slot_latents_labels.shape = {len(result_dict['slot_latents_labels'])}")
        #ipdb.set_trace()

        slot_latents = torch.cat(result_dict["slot_latents"], dim=0)  # Shape [T, num_slots, latent_dim] but we get 640, 64
        slot_latents_labels = torch.cat(result_dict["slot_latents_labels"], dim=0)  # Shape [T, num_slots]
        slot_latents_labels = slot_latents_labels.unsqueeze(-1)  # Ensure [T, num_slots, 1] but we get 640, 1

        print(f"✅ After Fix: slot_latents.shape = {slot_latents.shape}, slot_latents_labels.shape = {slot_latents_labels.shape}")

        #ipdb.set_trace()
        # check with copilot and ipdb

     
        return slot_latents.cpu().numpy(), slot_latents_labels.cpu().numpy()
    @staticmethod
    def collect_slot_masks(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
      """
      Collects slot attention masks for all sequences.

      Args:
      - cfg: Configuration object.
      - dataset_mode: 'train' or 'test'.
      - data_subset_mode: 'all' or 'relevant' labels.
      - only_collect_first_image_of_consecutive_frames (bool): If True, collects only the first frame per sequence.

      Returns:
      - slot_attn_masks (torch.Tensor): Tensor of slot attention masks [num_sequences, T, num_slots, H, W].
      """
      
      atari_dataset = Atari_Z_What(cfg, dataset_mode, data_subset_mode, return_keys=["slot_attn_masks"])
      num_sequences = len(atari_dataset)  # Total sequences in dataset

      base_path = atari_dataset.latents_path  # Path where masks are stored
      all_slot_attn_masks = []

      for stack_idx in range(num_sequences):
          slot_attn_masks = atari_dataset.read_tensor_of_complete_T_dim(stack_idx, base_path, "attn_masks")

          if only_collect_first_image_of_consecutive_frames:
              slot_attn_masks = slot_attn_masks[:1]  # Keep only first time step

          all_slot_attn_masks.append(slot_attn_masks)

      # Stack all sequences into a single tensor
      all_slot_attn_masks = torch.stack(all_slot_attn_masks)

      print(f"✅ slot_attn_masks.shape = {all_slot_attn_masks.shape}")  # Expected: [num_sequences, T, num_slots, H, W]

      return all_slot_attn_masks

                    

