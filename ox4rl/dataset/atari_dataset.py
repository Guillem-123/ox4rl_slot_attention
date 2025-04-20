import os
import ipdb
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
import pandas as pd
from torchvision import transforms
from ox4rl.utils.bbox_matching import get_label_of_best_matching_gt_bbox
from ox4rl.dataset.atari_labels import label_list_for, no_label_str, filter_relevant_boxes_masks
from ox4rl.models.space.postprocess_latent_variables import convert_to_boxes
from ox4rl.utils.bbox_matching import compute_misalignment, compute_center_negative_distance

class Atari_Z_What(Dataset):
    def __init__(self, cfg, dataset_mode, boxes_subset="all", return_keys=None, nr_consecutive_frames=4):
        self.game = cfg.gamelist[0]

        assert dataset_mode in ['train', 'validation', 'test'], f'Invalid dataset mode "{dataset_mode}"'
        self.dataset_mode = dataset_mode

        self.T = nr_consecutive_frames

        # folder paths
        img_folder = "space_like"
        self.image_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, img_folder)
        self.bb_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, "bb")
        self.motion_base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], cfg.moc_cfg.motion_kind)
        self.latents_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, "latents_slot")

        self.image_fn_count = len([True for img in os.listdir(self.image_path) if img.endswith(".png")])

        max_num_of_different_samples_for_dataset_mode = {
            "train": cfg.dataset_size_cfg.max_num_of_different_samples_for_dataset_mode.train,
            "validation": cfg.dataset_size_cfg.max_num_of_different_samples_for_dataset_mode.validation, 
            "test": cfg.dataset_size_cfg.max_num_of_different_samples_for_dataset_mode.test
        }

        self.max_num_samples = max_num_of_different_samples_for_dataset_mode[dataset_mode]
        self.len = min(self.image_fn_count // self.T, self.max_num_samples)
        if self.len < self.max_num_samples:
            print(f"Warning: available number of samples for {dataset_mode} is {self.len} but max_num_samples is {self.max_num_samples}.")
        elif self.len > self.max_num_samples:
            print(f"Warning: max_num_samples is {self.max_num_samples} but available number of samples for {dataset_mode} is {self.len}. -> not all samples will be used.")

        self.boxes_subset = boxes_subset # "all", "relevant"
        self.return_keys = return_keys

        self.img_size = cfg.space_cfg.img_shape[0]
        print(f"Expected latents folder path: {self.latents_base_path}")

    @property
    def bb_path(self):
        return osp.join(self.bb_base_path, self.dataset_mode)

    @property
    def image_path(self):
        return osp.join(self.image_base_path, self.dataset_mode)

    @property
    def latents_path(self):
        return osp.join(self.latents_base_path, self.dataset_mode)

    @property
    def motion_path(self):
        return osp.join(self.motion_base_path, self.dataset_mode)

    def __getitem__(self, stack_idx): # (T, ...) where T is number of consecutive frames and ... represents the actual dimensions of the data
        def __getitem__(self, stack_idx):
            """
            Retrieves a data sample from the dataset based on the provided index.
            Args:
                stack_idx (int): Index of the data sample to retrieve.
            Returns:
                dict: A dictionary containing the requested data keys and their corresponding values. The keys and values depend on the `return_keys` attribute of the dataset instance. Possible keys include:
                    - "imgs": A tensor of stacked images.
                    - "motion_slot": A tensor of motion slots.
                    - "motion_gt_labels": Ground truth labels for motion slots.
                    - "slot_latents": Latent representations of slots.
                    - "slot_latents_labels": Labels for slot latents.
                    - "motion": A tensor representing motion.
                    - "motion_z_pres": A tensor representing motion presence.
                    - "motion_z_where": A tensor representing motion location.
                    - "z_whats": Latent representations of objects.
                    - "z_pres_probs": Probabilities of object presence.
                    - "z_wheres": Locations of objects.
                    - "gt_bbs_and_labels": Ground truth bounding boxes and labels.
                    - "gt_labels_for_pred_boxes": Ground truth labels for predicted boxes.
                    - "pred_boxes": Predicted bounding boxes.
                    - "z_whats_pres_s": Latent representations of present objects.
            Raises:
                FileNotFoundError: If the latent file for the given index is missing.
            """

        base_path = self.image_path
        imgs = torch.stack([transforms.ToTensor()(self.read_img(stack_idx, i, base_path)) for i in range(self.T)])

        if self.return_keys == ["imgs"]:
            return {"imgs": imgs}

        base_path = self.motion_path

        if set(self.return_keys) == set(["imgs", "motion"]):
            motion = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix=f'{self.img_size}') for i in range(self.T)])
            motion = (motion > motion.mean() * 0.1).float() # Why???
            return {"imgs": imgs, "motion": motion}
        
        if set(self.return_keys) == set(["imgs", "motion_slot"]):
            motion_slot = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix="slot") for i in range(self.T)])
            return {"imgs": imgs, "motion_slot": motion_slot}

        if set(self.return_keys) == set(["imgs", "motion_slot", "motion_gt_labels"]):
            # used for eval
            motion_slot = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix="slot") for i in range(self.T)])
            base_path = self.bb_path
            gt_bbs_and_labels = [self.read_csv(stack_idx, i, base_path) for i in range(self.T)] # can't be stacked because of varying number of objects

            if self.boxes_subset == "relevant":  # remove non-moving gt boxes
                gt_bbs_and_labels = [gt_bbs_and_labels[i][gt_bbs_and_labels[i][:, 4] == 1] for i in range(self.T)]
            def transform_2d_mask_into_bb_coordinates(mask):
                x, y = np.where(mask == 1)
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                # normalize to [0, 1]
                x_min, x_max = x_min / mask.shape[0], x_max / mask.shape[0]
                y_min, y_max = y_min / mask.shape[1], y_max / mask.shape[1]
                if x_min == x_max: #TODO check 0.25, I chose it arbitrarily
                    x_min -= 0.25/mask.shape[0]
                    x_max += 0.25/mask.shape[0] 
                if y_min == y_max:
                    y_min -= 0.25/mask.shape[1]
                    y_max += 0.25/mask.shape[1]
                return [y_min, y_max, x_min, x_max]
            gt_labels_for_motion_masks = []
            for i in range(self.T):
                motion_slot_i_bbs = []
                for mask in motion_slot[i]:
                    if mask.sum() == 0:
                        break  # only empty masks left
                    bbox = transform_2d_mask_into_bb_coordinates(mask)
                    motion_slot_i_bbs.append(bbox)
                if isinstance(gt_bbs_and_labels[i], torch.Tensor) and gt_bbs_and_labels[i].dtype == torch.float64:
                    gt_bbs_and_labels[i] = gt_bbs_and_labels[i].to(torch.float)
                labels = get_label_of_best_matching_gt_bbox(torch.Tensor(gt_bbs_and_labels[i]), torch.Tensor(motion_slot_i_bbs), matching_method=compute_center_negative_distance).to(torch.int).tolist()
                labels = labels + [0] * (len(motion_slot[i]) - len(labels)) # append 0s to labels to match the length of motion_slot_i_bbs
                gt_labels_for_motion_masks.append(labels)
            gt_labels_for_motion_masks = torch.tensor(gt_labels_for_motion_masks)
            return {"imgs": imgs, "motion_slot": motion_slot, "motion_gt_labels": gt_labels_for_motion_masks}


            # raise NotImplementedError("This part of the code is not implemented yet.")
            # # ROUGH IDEA:
           
        
        if self.return_keys == ["imgs", "slot_latents", "slot_latents_labels"]:
            base_path = self.latents_path
            slot_latents = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "slot_repr")
            slot_attn_masks = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "attn_masks")
            # determine labels for slot_latents based on slot_attn_masks and gt_bbs_and_labels
            base_path = self.bb_path
            gt_bbs_and_labels = [self.read_csv(stack_idx, i, base_path) for i in range(self.T)] # can't be stacked because of varying number of object
            if self.boxes_subset == "relevant":  # remove non-moving gt boxes
                gt_bbs_and_labels = [gt_bbs_and_labels[i][gt_bbs_and_labels[i][:, 4] == 1] for i in range(self.T)]
            THRESHOLD = 0.1 # arbitrary threshold TODO TUNE
            def transform_2d_mask_into_bb_coordinates(mask):
                x, y = np.where(mask.cpu() >= THRESHOLD)
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                # normalize to [0, 1]
                x_min, x_max = x_min / mask.shape[0], x_max / mask.shape[0]
                y_min, y_max = y_min / mask.shape[1], y_max / mask.shape[1]
                if x_min == x_max: #TODO check 0.25, I chose it arbitrarily
                    x_min -= 0.25/mask.shape[0]
                    x_max += 0.25/mask.shape[0]
                if y_min == y_max:
                    y_min -= 0.25/mask.shape[1]
                    y_max += 0.25/mask.shape[1]
                return [y_min, y_max, x_min, x_max]
            gt_labels_for_latents = []
            collected_latents = []
            for i in range(self.T):
                slot_attn_masks_i_bbs = []
                latents_with_objects = []
                latents_wo_objects = []
                for idx, mask in enumerate(slot_attn_masks[i]):
                    if mask.max() < THRESHOLD:
                        latents_wo_objects.append(slot_latents[i][idx])
                        continue
                    bbox = transform_2d_mask_into_bb_coordinates(mask)
                    slot_attn_masks_i_bbs.append(bbox)
                    latents_with_objects.append(slot_latents[i][idx])
                if isinstance(gt_bbs_and_labels[i], torch.Tensor) and gt_bbs_and_labels[i].dtype == torch.float64:
                    gt_bbs_and_labels[i] = gt_bbs_and_labels[i].to(torch.float)
                labels_for_latents_with_objects = get_label_of_best_matching_gt_bbox(torch.Tensor(gt_bbs_and_labels[i]), torch.Tensor(slot_attn_masks_i_bbs), matching_method=compute_center_negative_distance).to(torch.int).tolist()
                labels = [0] * len(latents_wo_objects) + labels_for_latents_with_objects
                latents = latents_wo_objects + latents_with_objects
                gt_labels_for_latents.append(labels)
               
                #collected_latents.append(latents)
                if len(latents) > 0:
                    collected_latents.append(torch.stack(latents))  # Convert list to tensor
                else:
                    print(f"⚠️ Warning: No latents found at frame {i}, adding empty tensor.")
                    collected_latents.append(torch.empty((0, slot_latents.shape[-1])))  # Avoid stacking errors

            gt_labels_for_latents = torch.tensor(gt_labels_for_latents)
            # collected_latents = torch.stack(collected_latents)
            return {"imgs": imgs, "slot_latents": collected_latents, "slot_latents_labels": gt_labels_for_latents}

        motion = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix=f'{self.img_size}') for i in range(self.T)])
        motion = (motion > motion.mean() * 0.1).float() # Why???
        motion_z_pres = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix="z_pres") for i in range(self.T)])
        motion_z_where = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix="z_where") for i in range(self.T)])
        
        # early return if only imgs, motion, motion_z_pres, motion_z_where are requested
        if set(self.return_keys) == set(["imgs", "motion", "motion_z_pres", "motion_z_where"]):
            return {
                "imgs": imgs,
                "motion": motion,
                "motion_z_pres": motion_z_pres,
                "motion_z_where": motion_z_where
            }

        base_path = self.latents_path
        
        z_whats = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "slot_repr")
        z_pres_probs = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "attn_masks")
        z_wheres = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "z_where")
        z_pres_s = z_pres_probs > 0.5

        # z_whats = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "z_what")
        # z_pres_probs = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "z_pres_prob")
        # z_wheres = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "z_where")
        # z_pres_s = z_pres_probs > 0.5

        base_path = self.bb_path
        gt_bbs_and_labels = [self.read_csv(stack_idx, i, base_path) for i in range(self.T)] # can't be stacked because of varying number of objects

        ## boxes_subset == "moving" is deprecated
        # if self.boxes_subset == "moving": # modify labels of static objects to "no_label"
        #    for i in range(self.T):
        #        mask = gt_bbs_and_labels[i][:, 4] == 0 # static objects
        #        index_for_no_label = label_list_for(self.game).index(no_label_str)
        #        gt_bbs_and_labels[i][mask, 5] = index_for_no_label

        if self.boxes_subset == "relevant":  # remove non-moving gt boxes
            gt_bbs_and_labels = [gt_bbs_and_labels[i][gt_bbs_and_labels[i][:, 4] == 1] for i in range(self.T)]

        pred_boxes = convert_to_boxes(z_wheres, z_pres_s, z_pres_probs, with_conf=True) # list of arrays of shape (N, 4) where N is number of objects in that frame

        z_whats_pres_s = []
        for i in range(self.T):
            z_whats_pres_s.append(z_whats[i][z_pres_s[i]])

        gt_labels_for_pred_boxes = []
        for i in range(self.T):
            if isinstance(gt_bbs_and_labels[i], torch.Tensor) and gt_bbs_and_labels[i].dtype == torch.float64:
                gt_bbs_and_labels[i] = gt_bbs_and_labels[i].to(torch.float)
            gt_labels_for_pred_boxes.append(get_label_of_best_matching_gt_bbox(torch.Tensor(gt_bbs_and_labels[i]), torch.Tensor(pred_boxes[i])).reshape(-1, 1).to(torch.int)) 

        if self.boxes_subset == "relevant":
            masks = filter_relevant_boxes_masks(self.game, pred_boxes, None)
            for i in range(self.T):
                pred_boxes[i] = pred_boxes[i][masks[i]]
                gt_labels_for_pred_boxes[i] = gt_labels_for_pred_boxes[i][masks[i]]
                z_whats_pres_s[i] = z_whats_pres_s[i][masks[i]]

        data = {
            "imgs": imgs,
            "motion": motion,
            "motion_z_pres": motion_z_pres,
            "motion_z_where": motion_z_where,
            "z_whats": z_whats,
            "z_pres_probs": z_pres_probs,
            "z_wheres": z_wheres,
            "gt_bbs_and_labels": gt_bbs_and_labels,
            "gt_labels_for_pred_boxes": gt_labels_for_pred_boxes,
            "pred_boxes": pred_boxes,
            "z_whats_pres_s": z_whats_pres_s
        }
        if self.return_keys is not None:
            return {k: data[k] for k in self.return_keys}
        else:
            return data

    def __len__(self):
        return self.len

    def read_img(self, stack_idx, i, base_path):
        path = os.path.join(base_path, f'{stack_idx:05}_{i}.png')
        return np.array(Image.open(path).convert('RGB'))

    def read_tensor(self, stack_idx, i, base_path, postfix=None):
        path = os.path.join(base_path,
                            f'{stack_idx:05}_{i}_{postfix}.pt'
                            if postfix else f'{stack_idx:05}_{i}.pt')
        return torch.load(path)

    def read_tensor_of_complete_T_dim(self, stack_idx, base_path, postfix=None):
        infix = f"0to{self.T-1}" if self.T > 1 else "0"
        path = os.path.join(base_path,
                            f'{stack_idx:05}_{infix}_{postfix}.pt'
                            if postfix else f'{stack_idx:05}_{infix}.pt')

        # if not os.path.exists(base_path):
        #     print(f"Latents folder {base_path} not found. Creating it now...")
        #     os.makedirs(base_path, exist_ok=True)

        # if not os.path.exists(path):
        #     raise FileNotFoundError(f"Expected file not found: {path}")

        return torch.load(path)

    def read_csv(self, stack_idx, i, base_path):
        path = os.path.join(base_path, f"{stack_idx:05}_{i}.csv")

        label_list = label_list_for(self.game)
        
        df = pd.read_csv(path, header=None)
        df = df[df[5].isin(label_list)]
        df[4] = df[4].apply(lambda x: 1 if x.lower() == "m" else 0)
        df[5] = df[5].apply(lambda x: label_list.index(x))
        # convert to tensor
        return torch.tensor(df.values)
