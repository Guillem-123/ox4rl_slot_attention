
from ox4rl.dataset.atari_dataset import Atari_Z_What
from ox4rl.dataset import get_dataloader
from ox4rl.utils.load_config import get_config_v2
import os

import torch
from torchvision.utils import draw_bounding_boxes as draw_bb
from PIL import Image
import numpy as np

from ox4rl.dataset.atari_labels import relevant_area_borders


# THE FOLLOWING CODE NEEDS TO BE ADDED TO THE __getitem__ METHOD OF Atari_Z_What (in ox4rl/dataset/atari_dataset.py) for this script to work:
# if set(self.return_keys) == set(["imgs", "gt_bbs_and_labels"]):
#     base_path = self.bb_path
#     gt_bbs_and_labels = [self.read_csv(stack_idx, i, base_path) for i in range(self.T)]
#     if self.boxes_subset == "relevant":  # remove non-moving gt boxes
#         gt_bbs_and_labels = [gt_bbs_and_labels[i][gt_bbs_and_labels[i][:, 4] == 1] for i in range(self.T)]
#     # turn into tensor
#     for i in range(self.T):
#         if isinstance(gt_bbs_and_labels[i], torch.Tensor) and gt_bbs_and_labels[i].dtype == torch.float64:
#             gt_bbs_and_labels[i] = gt_bbs_and_labels[i].to(torch.float)
#     return {"imgs": imgs, "gt_bbs_and_labels": gt_bbs_and_labels}





from ox4rl.dataset.atari_labels import filter_relevant_boxes_masks
def draw_bboxes_inyminymaxxminxmax(dataloader, game):
    for i, data_dict in enumerate(dataloader):
        imgs = data_dict["imgs"] # B, T, 3, H, W
        gt_bbs_and_labels = data_dict["gt_bbs_and_labels"] # T, B, N, 5
        # assume batch size is 1
        imgs = imgs[0]
        gt_bbs_and_labels = [gt_bbs_and_labels[t][0] for t in range(len(gt_bbs_and_labels))]
        result_imgs = []
        for t in range(len(imgs)):
            img = imgs[t]
            bbs = gt_bbs_and_labels[t]

            # remove bbox that correspond to NoObject: all x_min, y_min, x_max, y_max are 0
            bbs = bbs[[not (bb[0] == 0 and bb[1] == 0 and bb[2] == 0 and bb[3] == 0) for bb in bbs]]

            # scale the bbs to 128x128
            bbs_masks = filter_relevant_boxes_masks(game, [bbs], None)[0]
            # check whether objects were removed
            if bbs_masks.sum() < bbs.shape[0]:
                print(f"Objects were removed in {game} at time step {t}.")
                # reverse the mask
                bbs_masks = ~bbs_masks
            else:
                continue
            bbs = bbs[bbs_masks]
            bbs = bbs * 128
            for bb in bbs:
                y_min, y_max, x_min, x_max = bb[0], bb[1], bb[2], bb[3]
                # Boxes need to be in (xmin, ymin, xmax, ymax)
                if y_min > y_max:
                    y_min, y_max = y_max, y_min
                try:
                    img = draw_bb(img, torch.Tensor([[x_min, y_min, x_max, y_max]]), colors=["red"])
                except:
                    import ipdb; ipdb.set_trace()
            result_imgs.append(img)
        # save the images to via Image from PIL
        for t in range(len(result_imgs)):
            # Convert tensor to NumPy array
            img_array = result_imgs[t].numpy()
            # Scale values to [0, 255] and convert to uint8
            img_array = (img_array * 255).astype(np.uint8)
            # Convert to HWC format for PIL
            img_array = np.transpose(img_array, (1, 2, 0))
            # Save the image
            os.makedirs(f"removed_objects/{game}", exist_ok=True)
            Image.fromarray(img_array).save(f"removed_objects/{game}/{game}_{i}_{t}.png")


def save_tensor_img_as_png(tensor, path):
    # Convert tensor to NumPy array
    img_array = tensor.numpy()
    # Scale values to [0, 255] and convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    # Convert to HWC format for PIL
    img_array = np.transpose(img_array, (1, 2, 0))
    # Save the image
    Image.fromarray(img_array).save(path)

# strategy: store the maximum x_max, y_max and minimum x_min, y_min for each game when only relevant gt_bbs_and_labels are considered
def get_min_max(dataloader, game):
    max_x_max, max_y_max = 0.0, 0.0
    min_x_min, min_y_min = 1.0, 1.0
    for i, data_dict in enumerate(dataloader):
        gt_bbs_and_labels = data_dict["gt_bbs_and_labels"] # T, B, N, 5
        # assume batch size is 1
        gt_bbs_and_labels = [gt_bbs_and_labels[t][0] for t in range(len(gt_bbs_and_labels))]
        for t in range(len(gt_bbs_and_labels)):
            bbs = gt_bbs_and_labels[t]
            for bb in bbs:
                y_min, y_max, x_min, x_max = bb[0], bb[1], bb[2], bb[3]

                # filter out NoObject boxes: can be identified by all x_min, y_min, x_max, y_max being 0
                if y_min == 0 and y_max == 0 and x_min == 0 and x_max == 0:
                    continue

                # error handling because some games are buggy (in particular Skiing)
                if y_min > y_max:
                    y_min, y_max = y_max, y_min
                if x_min > x_max:
                    x_min, x_max = x_max, x_min

                if x_max > max_x_max:
                    max_x_max = x_max
                if y_max > max_y_max:
                    max_y_max = y_max
                if x_min < min_x_min:
                    min_x_min = x_min
                if y_min < min_y_min:
                    min_y_min = y_min
    return min_x_min, min_y_min, max_x_max, max_y_max


# visualize the images with bboxes
games = ["Skiing", "Asterix", "Tennis", "Seaquest", "Kangaroo", "Freeway", "Bowling", "Pong", "Boxing"]
# games = ["Kangaroo"]


for game in games:
    cfg_path = f"configs/my_atari_pong_v2.yaml"
    cfg = get_config_v2(cfg_path)
    # update gamelist and exp_name
    cfg.gamelist = [f"ALE/{game}-v5"]
    cfg.exp_name = game.lower()
    dataset = Atari_Z_What(cfg, 'validation', return_keys=["imgs", "gt_bbs_and_labels"], boxes_subset="all")
    # manually set length of dataset to 2
    # dataset.len = 4
    train_data_loader = get_dataloader(cfg, 'validation', dataset)
    draw_bboxes_inyminymaxxminxmax(train_data_loader, game)


# results = {}

# for game in games:
#     cfg_path = f"configs/my_atari_pong_v2.yaml"
#     cfg = get_config_v2(cfg_path)
#     # update gamelist and exp_name
#     cfg.gamelist = [f"ALE/{game}-v5"]
#     cfg.exp_name = game.lower()
#     dataset = Atari_Z_What(cfg, 'validation', return_keys=["imgs", "gt_bbs_and_labels"], boxes_subset="relevant")
#     # manually set length of dataset to 2
#     train_data_loader = get_dataloader(cfg, 'validation', dataset)
#     #res_tuple = get_min_max(train_data_loader, cfg.exp_name)
    
#     #results[game] = res_tuple

#     res_tuple = relevant_area_borders[game.lower()]

#     # visualize on the first image of the dataset
#     # get first image from the dataloader
#     first_img = next(iter(train_data_loader))["imgs"][0][0]
#     # scale the result tuple to the image size (128x128)
#     res_tuple = [res_tuple[0] * 128, res_tuple[1] * 128, res_tuple[2] * 128, res_tuple[3] * 128]
#     first_img = draw_bb(first_img, torch.Tensor([[res_tuple[0], res_tuple[1], res_tuple[2], res_tuple[3]]]), colors=["red"])
#     save_tensor_img_as_png(first_img, f"{game}.png")



# # print results:
# for game in games:
#     print(f"{game}")
#     print(f"min_x_min: {results[game][0]}")
#     print(f"min_y_min: {results[game][1]}")
#     print(f"max_x_max: {results[game][2]}")
#     print(f"max_y_max: {results[game][3]}")
#     print("\n")



