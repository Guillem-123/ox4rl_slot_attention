from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision.utils import make_grid
import math
from ox4rl.vis.object_detection.space_vis import grid_mult_img, bbox_in_one



class Logger:

    @staticmethod
    @torch.no_grad()
    def log(model, writer: SummaryWriter, log, global_step, mode, cfg, dataset, num_batch=8): #TODO remove unnecessary params

        writer.add_scalar(f'{mode}/total_loss', log['elbo_loss'].item() + log['moc_loss'].item(), global_step=global_step)

        Logger.log_moc_losses(writer, log, global_step, mode, cfg)

        Logger.log_space_losses(writer, log, global_step, mode)

        Logger.log_visual_results(writer, log, global_step, mode, num_batch)

    @staticmethod  
    @torch.no_grad()
    def log_moc_losses(writer, log, global_step, mode, cfg):
        # log number of objects detected
        writer.add_scalar(f'{mode}/objects_detected', log['objects_detected'].item(), global_step=global_step) #TODO: doesnt work because objects_detected is of type long and not float

        # log development of parameters for balancing motion and object contuinity loss
        writer.add_scalar(f'{mode}/m_scheduling_weight', log['m_scheduling_weight'].item(), global_step=global_step)
        writer.add_scalar(f'{mode}/oc_scheduling_weight', log['oc_scheduling_weight'].item(), global_step=global_step)

        # high level losses
        writer.add_scalar(f'{mode}/moc_loss', log['moc_loss'].item(), global_step=global_step)
        writer.add_scalar(f'{mode}/oc_loss', log['oc_loss'].item(), global_step=global_step)
        writer.add_scalar(f'{mode}/motion_loss', log['motion_loss'].item(), global_step=global_step)

        # individual losses
        writer.add_scalar(f'{mode}/z_what_loss_objects', log['z_what_loss_objects'].item(), global_step=global_step)
        writer.add_scalar(f'{mode}/flow_loss_z_pres', log['flow_loss_z_pres'].item(), global_step=global_step)
        writer.add_scalar(f'{mode}/flow_loss_z_where', log['flow_loss_z_where'].item(), global_step=global_step)
        writer.add_scalar(f'{mode}/flow_loss_alpha_map', log['flow_loss_alpha_map'].item(), global_step=global_step)
        
        # scaled losses
        writer.add_scalar(f'{mode}/flow_loss_z_pres_scaled', log['flow_loss_z_pres'].item() * cfg.moc_cfg.motion_loss_weight_z_pres, global_step=global_step)
        writer.add_scalar(f'{mode}/flow_loss_z_where_scaled', log['flow_loss_z_where'].item() * cfg.moc_cfg.motion_loss_weight_z_where, global_step=global_step)
        writer.add_scalar(f'{mode}/flow_loss_alpha_map_scaled', log['flow_loss_alpha_map'].item() * cfg.moc_cfg.motion_loss_weight_alpha, global_step=global_step)
        writer.add_scalar(f'{mode}/motion_loss_scaled', log['motion_loss'].item() * log['m_scheduling_weight'].item() * cfg.moc_cfg.motion_weight, global_step=global_step)
        writer.add_scalar(f'{mode}/oc_loss_scaled', log['oc_loss'].item() * log['oc_scheduling_weight'].item() * cfg.moc_cfg.area_object_weight, global_step=global_step)


    
    @staticmethod
    @torch.no_grad()
    def log_space_losses(writer, log, global_step, mode):
        # Remark: in original code, the mean is used, but we use the sum here. Also, only a subset of the batch is used.
        count = torch.sum(log['z_pres']).item()
        writer.add_scalar(f'{mode}/count', count, global_step=global_step)
        writer.add_scalar(f'{mode}/elbo_loss', log['elbo_loss'].item(), global_step=global_step)
        writer.add_scalar(f'{mode}/mse', torch.sum(log['mse']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/log_like', torch.sum(log['log_like']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/What_KL', torch.sum(log['kl_z_what']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Where_KL', torch.sum(log['kl_z_where']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Pres_KL', torch.sum(log['kl_z_pres']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Depth_KL', torch.sum(log['kl_z_depth']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Bg_KL', torch.sum(log['kl_bg']).item(), global_step=global_step)
    
    @staticmethod
    @torch.no_grad()
    def log_visual_results(writer, log, global_step, mode, num_batch):
        # FYI: For visualization only use some images of each stack in the batch
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
                if isinstance(log[key], torch.Tensor) and log[key].ndim > 0:
                    log[key] = log[key][2:num_batch * 4:4] # step size 4 because by default T=4 (i.e. 4 consecutive images are stacked and they are very similar)
        log_img = dict(log)

        ### Visualization of the images ###
        # (B, 3, H, W) TR: Changed to z_pres_prob, why sample?
        fg_box = bbox_in_one(
            log_img['fg'], log_img['z_pres_prob'], log_img['z_where']
        )
        # (B, 1, 3, H, W)
        imgs = log_img['imgs'][:, None]
        fg = log_img['fg'][:, None]
        recon = log_img['y'][:, None]
        fg_box = fg_box[:, None]
        bg = log_img['bg'][:, None]
        # (B, K, 3, H, W)
        comps = log_img['comps']
        # (B, K, 3, H, W)
        masks = log_img['masks'].expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log_img['alpha_map'][:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.reshape(B * N, 3, H, W)

        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/0-separations', grid_image, global_step)

        grid_image = make_grid(log_img['imgs'], 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/1-dataset_image', grid_image, global_step)

        grid_image = make_grid(log_img['y'], 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/2-model_reconstruction_overall', grid_image, global_step)

        grid_image = make_grid(log_img['bg'], 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/3-model_background', grid_image, global_step)

        B = log_img['motion_z_pres'].shape[0]
        G = int(math.sqrt(log_img['motion_z_pres'].shape[1]))
        motion_z_pres_shape = (B, 1, G, G)
        writer.add_image(f'{mode}/4-dataset_motion',
                         grid_mult_img(log_img['motion_z_pres'], log_img['imgs'], motion_z_pres_shape),
                         global_step)

        grid_image = make_grid(log_img['motion'], 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/4-1-dataset_gt_alpha_map', grid_image, global_step)

        reshaped_motion = log_img['motion_z_pres'].reshape(motion_z_pres_shape)
        writer.add_image(f'{mode}/4-2-dataset_motion_z_pres', make_grid(reshaped_motion, 4, normalize=True, pad_value=1),
                         global_step)

        # sigmoid of z_pres_logits, which is reshjaped z_pres_pure ((B, 1, G, G) - > (B, G*G, 1))
        z_pres_grid = grid_mult_img(log_img['z_pres_prob'], log_img['imgs'], motion_z_pres_shape)
        writer.add_image(f'{mode}/5-model_z_pres_prob', z_pres_grid, global_step)

        # writer.add_image(f'{mode}/7-z_where', grid_z_where_vis(log_img['z_where'], log_img['imgs'], log_img['motion_z_pres']),
        #                 global_step)

        # gg_z_pres = log_img['z_pres_prob_pure'].reshape(log_img['motion_z_pres'].shape) > 0.5
        # writer.add_image(f'{mode}/8-z_where_pure', grid_z_where_vis(log_img['z_where_pure'], log_img['imgs'], gg_z_pres),
        #                                                            global_step)

        alpha_map = make_grid(log_img['alpha_map'], 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/9-model_alpha_map', alpha_map, global_step)
        # bb_image = draw_image_bb(model, cfg, dataset, global_step, num_batch)
        # grid_image = make_grid(bb_image, 4, normalize=False, pad_value=1)

        # importance_map = make_grid(log_img['importance_map_full_res_norm'], 4, normalize=False, pad_value=1)
        # writer.add_image(f'{mode}/10-model_importance_map', importance_map, global_step)