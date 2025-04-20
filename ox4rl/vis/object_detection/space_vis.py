from torch.utils.tensorboard import SummaryWriter
import imageio
import numpy as np
import torch
import matplotlib

from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader
from torchvision.utils import draw_bounding_boxes as draw_bb
from PIL import Image , ImageDraw, ImageFont
from ox4rl.vis.utils.color_box import gbox
import PIL
import math

from ox4rl.models.space.postprocess_latent_variables import convert_to_boxes
from ox4rl.models.space.space_utils import inverse_spatial_transform
from ox4rl.utils.data_reading import read_boxes

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ipdb
import pandas as pd
import os
from ox4rl.dataset.atari_dataset import Atari_Z_What
from ox4rl.dataset import get_dataloader

def bbox_in_one(x, z_pres, z_where, gbox=gbox):
    z_where_scale = z_where[..., :2]
    z_where_shift = z_where[..., 2:]
    B, _, *img_shape = x.size()
    B, N, _ = z_pres.size()
    z_pres = z_pres.reshape(-1, 1, 1, 1)
    z_scale = z_where_scale.reshape(-1, 2)
    z_shift = z_where_shift.reshape(-1, 2)
    # argmax_cluster = argmax_cluster.view(-1, 1, 1, 1)
    # kbox = boxes[argmax_cluster.view(-1)]
    bbox = inverse_spatial_transform(z_pres * gbox,  # + (1 - z_pres) * rbox,
                                     torch.cat((z_scale, z_shift), dim=1),
                                     torch.Size([B * N, 3, *img_shape]))
    bbox = (bbox.reshape(B, N, 3, *img_shape).sum(dim=1).clamp(0.0, 1.0) + x).clamp(0.0, 1.0)
    return bbox

def colored_bbox_in_one_image(x, z_pres, z_where_scale, z_where_shift, gbox=gbox):
    B, _, *img_shape = x.size()
    B, N, _ = z_pres.size()
    z_pres = z_pres.view(-1, 1, 1, 1)
    z_scale = z_where_scale.reshape(-1, 2)
    z_shift = z_where_shift.reshape(-1, 2)
    # argmax_cluster = argmax_cluster.view(-1, 1, 1, 1)
    # kbox = boxes[argmax_cluster.view(-1)]
    bbox = inverse_spatial_transform(z_pres * gbox,  # + (1 - z_pres) * rbox,
                                      torch.cat((z_scale, z_shift), dim=1),
                                      torch.Size([B * N, 3, *img_shape]))

    bbox = bbox.view(B, N, 3, *img_shape).sum(dim=1).clamp(0.0, 1.0)
    bbox = (bbox + x).clamp(0.0, 1.0)
    return bbox

def grid_mult_img(grid, imgs, target_shape, scaling=4):
    grid = grid.reshape(target_shape)
    to_G = imgs.shape[2] // grid.shape[2]
    grid = grid.repeat_interleave(to_G, dim=2).repeat_interleave(to_G, dim=3)
    vis_imgs = (imgs + 0.3) / 1.3
    grid = vis_imgs * (scaling * grid + 1) / 4  # Intended oversaturation for visualization
    return make_grid(grid, 4, normalize=False, pad_value=1)


# def grid_z_where_vis(z_where, imgs, z_pres, scaling=4):
#     vis_imgs = (imgs + 0.3) / 1.3
#     boxes_batch = convert_to_boxes(z_where, z_pres.squeeze(), z_pres.squeeze(), with_conf=False)
#     grid = torch.zeros_like(vis_imgs)[:, 0:1]
#     for i, boxes in enumerate(boxes_batch):
#         for box in boxes:
#             y_min, y_max, x_min, x_max = [min(127, int(round(e * 128))) for e in box]
#             grid[i][0][y_min:y_max, x_min:x_max] = 1
#     grid = imgs * (scaling * grid + 4) / 9
#     return make_grid(grid, 4, normalize=False, pad_value=1)

    # @torch.no_grad()
    # def show_vis(self, model, dataset, indices, path, device):
    #     print(f"Data: {dataset}")
    #     print(f"Indices: {indices}")
    #     dataset = Subset(dataset, indices)
    #     print(f"Data: {dataset}")
    #     dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
    #     data = next(iter(dataloader))
    #     for key in ['z_whats_pres_s', 'gt_labels_for_pred_boxes', 'pred_boxes', 'gt_bbs_and_labels']:
    #         print(key)
    #         tmp = data.get(key)
    #         print(len(tmp))
    #         for tensor in tmp:
    #             print(tensor.size())

    #     data = data.get('z_whats_pres_s')
    #     print(len(data))
    #     print(data[0].size())
    #     # for i in range(len(data)):  # TODO: check whether this is necessary
    #     #     data[i] = data[i].to(device)
    #     #print(*data)
    #     loss, log = model(torch.tensor(data), 100000000)
    #     for key, value in log.items():
    #         if isinstance(value, torch.Tensor):
    #             log[key] = value.detach().cpu()

    #     # (B, 3, H, W)
    #     fg_box = bbox_in_one(
    #         log['fg'], log['z_pres'], log['z_scale'], log['z_shift']
    #     )
    #     # (B, 1, 3, H, W)
    #     imgs = log['imgs'][:, None]
    #     fg = log['fg'][:, None]
    #     recon = log['y'][:, None]
    #     fg_box = fg_box[:, None]
    #     bg = log['bg'][:, None]
    #     # (B, K, 3, H, W)
    #     comps = log['comps']
    #     # (B, K, 3, H, W)
    #     masks = log['masks'].expand_as(comps)
    #     masked_comps = comps * masks
    #     alpha_map = log['alpha_map'][:, None].expand_as(imgs)
    #     grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
    #     nrow = grid.size(1)
    #     B, N, _, H, W = grid.size()
    #     grid = grid.view(B * N, 3, H, W)

    #     # (3, H, W)
    #     grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)

    #     # (H, W, 3)
    #     image = torch.clamp(grid_image, 0.0, 1.0)
    #     image = image.permute(1, 2, 0).numpy()
    #     image = (image * 255).astype(np.uint8)
    #     imageio.imwrite(path, image)

@torch.no_grad()
def show_bb(model, image, path, device):
    import torch
    image = image.to(device)
    loss, log = model(image, 100000000)
    for key, value in log.items():
        if isinstance(value, torch.Tensor):
            log[key] = value.detach().cpu()

    # (B, 3, H, W)
    fg_box = colored_bbox_in_one_image(
        log['fg'], log['z_pres'], log['z_scale'], log['z_shift']
    )
    # (B, 1, 3, H, W)
    imgs = log['imgs'][:, None]
    fg = log['fg'][:, None]
    recon = log['y'][:, None]
    fg_box = fg_box[:, None]
    bg = log['bg'][:, None]
    # (B, K, 3, H, W)
    comps = log['comps']
    # (B, K, 3, H, W)
    masks = log['masks'].expand_as(comps)
    masked_comps = comps * masks
    alpha_map = log['alpha_map'][:, None].expand_as(imgs)
    grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
    plt.imshow(fg_box[0][0].permute(1, 2, 0))
    plt.savefig(path)


# Times 10 to prevent index out of bound.
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * 10
def draw_image_bb(model, cfg, dataset, global_step, num_batch):
    indices = np.random.choice(len(dataset), size=num_batch, replace=False)
    dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
    data_dict = next(iter(dataloader))
    data, motion_z_pres, motion_z_where = data_dict["imgs"], data_dict["motion_z_pres"], data_dict["motion_z_where"]
    data = data.to(cfg.device)
    motion_z_pres = motion_z_pres.to(cfg.device)
    motion_z_where = motion_z_where.to(cfg.device)
    loss, log = model(data, global_step)
    bb_path = f"{cfg.dataset_roots.ATARI}/{cfg.gamelist[0]}/train/bb"
    rgb_folder_src = f"{cfg.dataset_roots.ATARI.replace('space_like', 'rgb')}/{cfg.gamelist[0]}/train"
    boxes_gt, boxes_gt_moving, _ = read_boxes(bb_path, indices=indices)
    boxes_pred = []
    z_where, z_pres_prob = log['z_where'][2:num_batch * 4:4], log['z_pres_prob'][2:num_batch * 4:4]
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu().squeeze()
    z_pres = z_pres_prob > 0.5
    boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
    boxes_pred.extend(boxes_batch)
    result = []
    for idx, pred, gt, gt_m in zip(indices, boxes_pred[2::4], boxes_gt[2::4], boxes_gt_moving[2::4]):
        pil_img = Image.open(f'{rgb_folder_src}/{idx:05}_2.png', ).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        image = np.array(pil_img)
        torch_img = torch.from_numpy(image).permute(2, 1, 0)
        pred_tensor = torch.FloatTensor(pred) * 128
        pred_tensor = torch.index_select(pred_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        gt_tensor = torch.FloatTensor(gt) * 128
        gt_tensor = torch.index_select(gt_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        gt_m_tensor = torch.FloatTensor(gt_m) * 128
        gt_m_tensor = torch.index_select(gt_m_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        bb_img = draw_bb(torch_img, gt_tensor, colors=["red"] * len(gt_tensor))
        bb_img = draw_bb(bb_img, gt_m_tensor, colors=["blue"] * len(gt_m_tensor))
        bb_img = draw_bb(bb_img, pred_tensor, colors=["green"] * len(pred_tensor))
        bb_img = bb_img.permute(0, 2, 1)
        result.append(bb_img)
    return torch.stack(result)

def transform_bbox_format(bboxes):
    """
    Transform from (y_min, y_max, x_min, x_max) to (x_min, y_min, x_max, y_max)

    Args:
        bboxes: Tensor of shape (N, M) where N is the number of bounding boxes in format and the first M >= 4 and the first 4 are in format (y_min, y_max, x_min, x_max)

    Returns:
        Tensor of shape (N, M) where N is the number of bounding boxes in format and the first M >= 4 and the first 4 are in
        format (x_min, y_min, x_max, y_max)
    """
    new_format_bboxes = np.array(bboxes)
    new_format_bboxes[:, [0, 1, 2, 3]] = new_format_bboxes[:, [2, 0, 3, 1]]
    return new_format_bboxes


# Times 10 to prevent index out of bound.
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * 10
def draw_image_bb_v2(cfg, dataloader, game, max_num_of_img_sequences=10):
    # dataloader that wraps Atari_Z_What dataset with return_keys=["imgs", "
    for i, data_dict in enumerate(dataloader):
        if i >= max_num_of_img_sequences:
            break

        # assume batch size is 1
        imgs = data_dict["imgs"][0] # B, T, C, H, W -> T, C, H, W
        pred_boxes = [data_dict["pred_boxes"][t][0] for t in range(len(data_dict["pred_boxes"]))] # T, B, N, 4 -> T, N, 4
        gt_bbs_and_labels = [data_dict["gt_bbs_and_labels"][t][0] for t in range(len(data_dict["gt_bbs_and_labels"]))] # T, B, N, 6 -> T, N, 6
        #gt_labels_for_pred_boxes = [data_dict["gt_labels_for_pred_boxes"][t][0] for t in range(len(data_dict["gt_labels_for_pred_boxes"]))] # T, B, N, 1 -> T, N, 1



        result_imgs = []
        # the data stores boxes in format y_min, y_max, x_min, x_max
        for t in range(len(imgs)):
            img = imgs[t]
            pred_boxes_t = transform_bbox_format(pred_boxes[t])
            gt_bbs_and_labels_t = transform_bbox_format(gt_bbs_and_labels[t])
            # skiing game has bug that y_min, y_max and x_min, x_max are swapped, respectively. Check if this is the case and correct it
            if game.lower() == "skiing": # TODO: move this check closer to the data loading part
                # check for each object whether y_min > y_max and x_min > x_max; Note that pred_boxes has format (x_min, y_min, x_max, y_max)
                for j in range(len(gt_bbs_and_labels_t)):
                    if gt_bbs_and_labels_t[j][0] > gt_bbs_and_labels_t[j][2]:
                        # swap 0 and 2 using a temporary variable
                        print("Alert: Skiing ground truth bounding box has x_min > x_max")
                        temp = gt_bbs_and_labels_t[j][0]
                        gt_bbs_and_labels_t[j][0] = gt_bbs_and_labels_t[j][2]
                        gt_bbs_and_labels_t[j][2] = temp
                    if gt_bbs_and_labels_t[j][1] > gt_bbs_and_labels_t[j][3]:
                        # swap 1 and 3
                        print("Alert: Skiing ground truth bounding box has y_min > y_max")
                        temp = gt_bbs_and_labels_t[j][1]
                        gt_bbs_and_labels_t[j][1] = gt_bbs_and_labels_t[j][3]
                        gt_bbs_and_labels_t[j][3] = temp

            img = draw_bb(img, torch.Tensor(pred_boxes_t[:, :4]*128), colors=["green"] * len(pred_boxes_t))
            img = draw_bb(img, torch.Tensor(gt_bbs_and_labels_t[:, :4]*128), colors=["red"] * len(gt_bbs_and_labels_t))
            result_imgs.append(img)

        for t in range(len(result_imgs)):
            # Convert tensor to NumPy array
            img_array = result_imgs[t].numpy()
            # Scale values to [0, 255] and convert to uint8
            img_array = (img_array * 255).astype(np.uint8)
            # Convert to HWC format for PIL
            img_array = np.transpose(img_array, (1, 2, 0))
            # Save the image
            os.makedirs(f"visualizations/{game}", exist_ok=True)
            Image.fromarray(img_array).save(f"visualizations/{game}/{game}_{i}_{t}.png")



# Times 10 to prevent index out of bound.
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * 10
def draw_image_bb_v2_slot(cfg, dataloader, game, masks, max_num_of_img_sequences=10):
  """
  method to generate the bounding boxes for slot attention
  """
    # dataloader that wraps Atari_Z_What dataset with return_keys=["imgs", "
  for i, data_dict in enumerate(dataloader):
        if i >= max_num_of_img_sequences:
            break

        # assume batch size is 1
        imgs = data_dict["imgs"][0] # B, T, C, H, W -> T, C, H, W
        for t in range(len(imgs)):
            # Convert tensor to NumPy array
            img_array = imgs[t].numpy()
            # Scale values to [0, 255] and convert to uint8
            img_array = (img_array * 255).astype(np.uint8)
            # Convert to HWC format for PIL
            img_array = np.transpose(img_array, (1, 2, 0))
            # Save the image
            os.makedirs(f"visualizations/slot/original_imgs/{game}", exist_ok=True)
            Image.fromarray(img_array).save(f"visualizations/slot//original_imgs/{game}/{game}_{i}_{t}.png")
  
       
        #pred_boxes = [data_dict["pred_boxes"][t][0] for t in range(len(data_dict["pred_boxes"]))] # T, B, N, 4 -> T, N, 4
        #gt_bbs_and_labels = [data_dict["gt_bbs_and_labels"][t][0] for t in range(len(data_dict["gt_bbs_and_labels"]))] # T, B, N, 6 -> T, N, 6
        #gt_labels_for_pred_boxes = [data_dict["gt_labels_for_pred_boxes"][t][0] for t in range(len(data_dict["gt_labels_for_pred_boxes"]))] # T, B, N, 1 -> T, N, 1



        result_imgs = []
        # the data stores boxes in format y_min, y_max, x_min, x_max
        for t in range(len(imgs)):
            img = imgs[t]
            slot_masks=masks[i]
            

    
            
            #pred_boxes_t = transform_bbox_format(pred_boxes[t])
            #gt_bbs_and_labels_t = transform_bbox_format(gt_bbs_and_labels[t])
            # skiing game has bug that y_min, y_max and x_min, x_max are swapped, respectively. Check if this is the case and correct it
            if game.lower() == "skiing": # TODO: move this check closer to the data loading part
                # check for each object whether y_min > y_max and x_min > x_max; Note that pred_boxes has format (x_min, y_min, x_max, y_max)
                for j in range(len(gt_bbs_and_labels_t)):
                    if gt_bbs_and_labels_t[j][0] > gt_bbs_and_labels_t[j][2]:
                        # swap 0 and 2 using a temporary variable
                        print("Alert: Skiing ground truth bounding box has x_min > x_max")
                        temp = gt_bbs_and_labels_t[j][0]
                        gt_bbs_and_labels_t[j][0] = gt_bbs_and_labels_t[j][2]
                        gt_bbs_and_labels_t[j][2] = temp
                    if gt_bbs_and_labels_t[j][1] > gt_bbs_and_labels_t[j][3]:
                        # swap 1 and 3
                        print("Alert: Skiing ground truth bounding box has y_min > y_max")
                        temp = gt_bbs_and_labels_t[j][1]
                        gt_bbs_and_labels_t[j][1] = gt_bbs_and_labels_t[j][3]
                        gt_bbs_and_labels_t[j][3] = temp

            # Extract bounding boxes from slot attention masks
            slot_bboxes = extract_bboxes_from_masks(slot_masks[t])  # Compute slot attention bboxes
            

            img = draw_bb(img, torch.Tensor(slot_bboxes[:, :4] * 128), colors=["yellow"] * len(slot_bboxes))  # Slot attention bboxes
            
            #img = draw_bb(img, torch.Tensor(gt_bbs_and_labels_t[:, :4] * 128), colors=["red"] * len(gt_bbs_and_labels_t))  # Ground truth bboxes

            #img = draw_bb(img, torch.Tensor(pred_boxes_t[:, :4]*128), colors=["green"] * len(pred_boxes_t))
            #img = draw_bb(img, torch.Tensor(gt_bbs_and_labels_t[:, :4]*128), colors=["red"] * len(gt_bbs_and_labels_t))
            result_imgs.append(img)
        for t in range(len(result_imgs)):
            # Convert tensor to NumPy array
            img_array = result_imgs[t].numpy()
            # Scale values to [0, 255] and convert to uint8
            img_array = (img_array * 255).astype(np.uint8)
            # Convert to HWC format for PIL
            img_array = np.transpose(img_array, (1, 2, 0))
            # Save the image
            os.makedirs(f"visualizations/slot/{game}", exist_ok=True)
            Image.fromarray(img_array).save(f"visualizations/slot/{game}/{game}_{i}_{t}.png")

def extract_bboxes_from_masks(mask):
    """
    Extracts bounding boxes from slot attention masks.
    Assumes input mask is of shape (num_slots, H, W).
    """
    num_slots, H, W = mask.shape
    bboxes = []
    
    for i in range(num_slots):
        binary_mask = mask[i] > 0.5  # Threshold to binarize the mask
        if binary_mask.sum() == 0:
            continue
        
        y_indices, x_indices = torch.where(binary_mask)
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        
        bboxes.append([x_min / W, y_min / H, x_max / W, y_max / H])  # Normalize coordinates
    
    return torch.tensor(bboxes)

# (existing imports and functions, including colored_bbox_in_one_image, etc.)

@torch.no_grad()
def label_bb(model, image, path, device):
    """
    Runs the model on the image, draws the predicted bounding boxes and a text label,
    and saves the resulting image to `path`.
    """
    image = image.to(device)
    loss, log = model(image, 100000000)
    for key, value in log.items():
        if isinstance(value, torch.Tensor):
            log[key] = value.detach().cpu()
    
    # Get the bounding boxes overlayed image (same as in show_bb)
    fg_box = colored_bbox_in_one_image(
        log['fg'], log['z_pres'], log['z_scale'], log['z_shift']
    )
    
    # Get the first image from fg_box (assuming batch size 1)
    # Convert to a PIL image for drawing
    pil_img = Image.fromarray((fg_box[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    # Use PIL ImageDraw to add a text label.
    draw = ImageDraw.Draw(pil_img)
    # Use the default font; for custom fonts, specify the font file
    font = ImageFont.load_default()
    
    # (For demonstration, we simply label the image with "Label" at position (10, 10).
    # In a real case you may loop over multiple box coordinates.)
    draw.text((10, 10), "Label", fill="red", font=font)
    
    pil_img.save(path)

@torch.no_grad()
def class_label_bb(model, image, path, device, classifier=None, game_name="Pong"):
    """
    Runs the model on the image, draws bounding boxes colored according to
    predicted classes with label text, and saves the resulting image to `path`.
    
    Args:
        model: The model to process the image
        image: Input image tensor
        path: Path to save the output image
        device: Device to run the model on
        classifier: Optional classifier to predict object classes from z_what
        game_name: The name of the game (to get the correct labels)
    """
    from ox4rl.dataset.atari_labels import label_list_for
    
    # Get the actual label names for this game
    game_labels = label_list_for(game_name)
    
    image = image.to(device)
    loss, log = model(image, 100000000)
    
    # Detach tensors in log to CPU
    for key, value in log.items():
        if isinstance(value, torch.Tensor):
            log[key] = value.detach().cpu()
    
    # Extract latent variables
    z_where = log.get('z_where', None)
    z_pres_prob = log.get('z_pres_prob', None)
    z_what = log.get('z_what', None)  # For classification if available
    
    if z_where is None or z_pres_prob is None:
        # Fall back to drawing colored bounding boxes without labels
        fg_box = colored_bbox_in_one_image(
            log['fg'], log['z_pres'], log['z_scale'], log['z_shift']
        )
        plt.imshow(fg_box[0].permute(1, 2, 0))
        plt.title("No pred_boxes available")
        plt.savefig(path)
        return
    
    # Fix the shape mismatch by reshaping z_pres to match z_where's shape
    z_pres = (z_pres_prob > 0.5).squeeze(-1)  # Remove the last dimension if it exists
    if len(z_pres.shape) == 2 and z_pres.shape[1] == 1:
        z_pres = z_pres.squeeze(1)  # Remove the second dimension if it's 1
        
    # Check if z_where needs similar treatment
    if len(z_where.shape) == 3 and z_where.shape[2] == 4:
        # All good, z_where has the right shape
        pass
    elif len(z_where.shape) == 2 and z_where.shape[1] == 4:
        # Add batch dimension if needed
        z_where = z_where.unsqueeze(0)
        
    z_pres_prob = z_pres_prob.squeeze(-1)  # Remove the last dimension if it exists
    if len(z_pres_prob.shape) == 2 and z_pres_prob.shape[1] == 1:
        z_pres_prob = z_pres_prob.squeeze(1)  # Remove the second dimension if it's 1

    # Now convert to boxes
    boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
    
    # If we have a classifier and z_what, predict classes
    if classifier is not None and z_what is not None:
        # Flatten z_what if needed (depends on your model's output format)
        # This might need adjustment based on your z_what shape
        flattened_z_what = z_what.reshape(-1, z_what.shape[-1])
        # Filter by z_pres
        active_z_what = flattened_z_what[z_pres.flatten()]
        
        # Predict labels
        if len(active_z_what) > 0:
            active_z_what_np = active_z_what.cpu().numpy().astype('float64')
            pred_labels = classifier.predict(active_z_what_np)
        else:
            pred_labels = []
    else:
        # Assign a default class (0) if no classifier
        pred_labels = [0] * len(boxes_batch[0])
    
    # Create PIL image from input image
    pil_img = Image.fromarray((image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    # Define colors for different classes
    colors_list = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 
                   'lime', 'pink', 'teal', 'lavender', 'brown', 'maroon', 'navy']
    
    # Define colors for different classes - map to actual labels
    unique_labels = set([int(l) for l in pred_labels])
    color_map = {label: colors_list[i % len(colors_list)] 
                for i, label in enumerate(unique_labels)}
    
    # Draw each box with its class label
    for i, box in enumerate(boxes_batch[0]):
        if i < len(pred_labels):
            class_id = int(pred_labels[i])
            
            # Get the actual label name from game_labels
            # Note: Ensure class_id is within range of game_labels
            if 0 <= class_id < len(game_labels):
                label_name = game_labels[class_id]
            else:
                label_name = f"Unknown ({class_id})"
                
            color = color_map.get(class_id, 'white')
            
            # Convert box to pixel coordinates
            y_min, y_max, x_min, x_max = [int(coord * 128) for coord in box[:4]]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            
            # Add label with actual object name
            draw.text((x_min, max(y_min-10, 0)), label_name, fill=color)
    
    # Save the image
    pil_img.save(path)
