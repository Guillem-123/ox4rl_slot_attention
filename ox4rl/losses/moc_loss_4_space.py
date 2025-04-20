import torch
import torch.nn as nn
from ox4rl.configs.space_cfg import space_cfg
from ox4rl.configs.moc_cfg import moc_cfg
from ox4rl.losses.moc_loss_scheduling import Linear_MOC_Loss_Scheduler, Dynamic_MOC_Loss_Scheduler

class MOCLoss():
    def __init__(self):
        self.zero = nn.Parameter(torch.tensor(0.0))
        self.area_object_weight = moc_cfg.area_object_weight
        self.scheduler = Dynamic_MOC_Loss_Scheduler() if moc_cfg.dynamic_scheduling else Linear_MOC_Loss_Scheduler()
  
    def compute_loss(self, motion, motion_z_pres, motion_z_where, logs, global_step):
        """
        Inference.
        With time-dimension for consistency
        :param x: (B, 3, H, W)
        :param motion: (B, H, W)
        :param motion_z_pres: z_pres hint from motion
        :param motion_z_where: z_where hint from motion
        :param global_step: global training step
        :return:
            loss: a scalar. Note it will be better to return (B,)
            log: a dictionary for visualization
        """
        if len(motion.shape) == 4: # (B, T, H, W)
            B, T, H, W = motion.shape
            motion = motion.reshape(T * B, 1, H, W)
            motion_z_pres = motion_z_pres.reshape(T * B, space_cfg.G * space_cfg.G, 1)
            motion_z_where = motion_z_where.reshape(T * B, space_cfg.G * space_cfg.G, 4)
        else: # (B, H, W)
            B, H, W = motion.shape
            motion = motion.reshape(B, 1, H, W)
            motion_z_pres = motion_z_pres.reshape(B, space_cfg.G * space_cfg.G, 1)
            motion_z_where = motion_z_where.reshape(B, space_cfg.G * space_cfg.G, 4)

        tc_log = {
            'motion': motion,
            'motion_z_pres': motion_z_pres,
            'motion_z_where': motion_z_where,
        }
        logs.update(tc_log)

        # Object Continuity Loss Components
        z_what_loss_objects, objects_detected = z_what_consistency_objects(logs) if moc_cfg.area_object_weight > 1e-3 else (self.zero, self.zero)

        # Motion Loss Components
        flow_loss_z_pres = compute_flow_loss_z_pres(logs) if moc_cfg.motion_loss_weight_z_pres > 1e-3 else self.zero
        flow_loss_z_where = compute_flow_loss_z_where(logs) if moc_cfg.motion_loss_weight_z_where > 1e-3 else self.zero
        flow_loss_alpha_map = compute_flow_loss_alpha(logs) if moc_cfg.motion_loss_weight_alpha > 1e-3 else self.zero

        # weigh object continuity loss components
        oc_loss = z_what_loss_objects

        # weigh motion loss components
        motion_loss = flow_loss_z_pres * moc_cfg.motion_loss_weight_z_pres \
            + flow_loss_alpha_map * moc_cfg.motion_loss_weight_alpha \
            + flow_loss_z_where * moc_cfg.motion_loss_weight_z_where

        # weight to balance image reconstruction and object localization improvements (NOT MENTIONED IN PAPER)
        # (only relevant in the beginning as contribution of motion loss decreases to 0 due to scheduling)
        lambda_m = moc_cfg.motion_weight

        # weight to balance image reconstruction and encoding improvements
        lambda_oc = moc_cfg.area_object_weight

        # compute scheduling-based relative weights for motion loss and object continuity loss
        lambda_align = self.scheduler.get_value(logs=logs, global_step=global_step)
        lambda_align = torch.tensor(lambda_align).to(motion.device)
        m_scheduling_weight = (1 - lambda_align)
        oc_scheduling_weight = lambda_align
        
        moc_loss = m_scheduling_weight * lambda_m * motion_loss + \
                     oc_scheduling_weight * lambda_oc * oc_loss

        tc_log = {
            'objects_detected': objects_detected,

            'z_what_loss_objects': z_what_loss_objects,
            'oc_loss': oc_loss,
            'flow_loss_z_pres': flow_loss_z_pres,
            'flow_loss_z_where': flow_loss_z_where,
            'flow_loss_alpha_map': flow_loss_alpha_map,
            'motion_loss': motion_loss,
            'moc_loss': moc_loss,

            'm_scheduling_weight': m_scheduling_weight,
            'oc_scheduling_weight': oc_scheduling_weight,
        }

        logs.update(tc_log)
        return moc_loss, logs

def compute_flow_loss_alpha(responses):
    alpha_map = responses['alpha_map']
    alpha_map_gt = responses['motion']
    return nn.functional.mse_loss(alpha_map, alpha_map_gt, reduction='sum')

def compute_flow_loss_z_pres(responses):
    z_pres = responses['z_pres_prob_pure']
    motion_z_pres = responses['motion_z_pres']
    return nn.functional.mse_loss(z_pres, motion_z_pres.reshape(z_pres.shape), reduction='sum')

def compute_flow_loss_z_where(responses):
    motion_z_pres = responses['motion_z_pres'].squeeze()
    pred_z_pres = responses['z_pres_prob_pure'].reshape(motion_z_pres.shape)
    z_where = responses['z_where_pure']
    motion_z_where = responses['motion_z_where']
    return nn.functional.mse_loss(z_where[motion_z_pres > 0.5], motion_z_where[motion_z_pres > 0.5], reduction='sum') + \
           nn.functional.mse_loss(z_where[pred_z_pres > 0.5], motion_z_where[pred_z_pres > 0.5], reduction='sum')

# Remark: the original code of the MOC paper contains further versions that try to approximate the contrastive loss, but they were not used in the paper (see https://github.com/k4ntz/MOC/blob/master/src/model/space/time_consistency.py)
# Contrastive loss for z_what
def z_what_consistency_objects(responses):
    # B = Batch size, T = Sequence length, G = Grid size, D = z_what dimension
    cos = nn.CosineSimilarity(dim=1)

    z_whats = responses['z_what']
    # (B*T, G*G, D)
    _, GG, D = z_whats.shape
    # (B, T, G*G, D)
    z_whats = z_whats.reshape(-1, space_cfg.T, GG, D)
    B, T = z_whats.shape[:2]

    # (T, B, G, G)
    z_pres = responses['z_pres_prob'].reshape(B, T, GG, 1).reshape(B, T, space_cfg.G, space_cfg.G).transpose(0, 1)
    # (T, B, G, G, D)
    z_whats = z_whats.reshape(B, T, space_cfg.G, space_cfg.G, D).transpose(0, 1)
    # (T, B, G, G, 4)
    z_where = responses['z_where'].reshape(B, T, space_cfg.G, space_cfg.G, -1).transpose(0, 1)

    # (T, B, G+2, G+2)
    z_pres_same_padding = torch.nn.functional.pad(z_pres, (1, 1, 1, 1), mode='circular')
    # (T, B, G+2, G+2, D)
    z_what_same_padding = torch.nn.functional.pad(z_whats, (0, 0, 1, 1, 1, 1), mode='circular') # Nils: (0,0) adds no padding for the last dimension
    # (T, B, G+2, G+2, 4)
    z_where_same_padding = torch.nn.functional.pad(z_where, (0, 0, 1, 1, 1, 1), mode='circular')

    # (#detected objects, 4) where z_pres_idx[0] are the "coordinates" of the objects in the z_pres tensor structure (i.e. the indices of the objects)
    z_pres_idx = (z_pres[:-1] > moc_cfg.object_threshold).nonzero(as_tuple=False) # -1 because we need to compare objects in consecutive frames
    # idx: (4,)
    object_consistency_loss = torch.tensor(0.0).to(z_whats.device)
    for idx in z_pres_idx:
        # get 3x3 area around idx (note that due to padding, idx is shifted by 1)
        # (3, 3)
        z_pres_area = z_pres_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
        # (3, 3, D)
        z_what_area = z_what_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
        # (3, 3, 4)
        z_where_area = z_where_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]

        # Tuple((#hits,) (#hits,))
        z_what_idx = (z_pres_area > moc_cfg.object_threshold).nonzero(as_tuple=True)

        # (1, D)
        z_what_prior = z_whats[idx.tensor_split(4)]
        # (1, 4)
        z_where_prior = z_where[idx.tensor_split(4)]

        # (#hits, D)
        z_whats_now = z_what_area[z_what_idx]
        # (#hits, 4)
        z_where_now = z_where_area[z_what_idx]
        if z_whats_now.nelement() == 0:
            continue

        if moc_cfg.cosine_sim:
            # (#hits,)
            z_sim = cos(z_what_prior, z_whats_now)
        else:
            z_means = nn.functional.mse_loss(z_what_prior, z_whats_now, reduction='none')
            # (#hits,) in (0,1]
            z_sim = 1 / (torch.mean(z_means, dim=1) + 1)

        if z_whats_now.shape[0] > 1:
            similarity_max_idx = z_sim.argmax()
            if moc_cfg.agree_sim:
                # if mse and cosine similarity do not agree on the most similar object, continue
                pos_dif_min = nn.functional.mse_loss(z_where_prior.expand_as(z_where_now), z_where_now,
                                                     reduction='none').sum(dim=-1).argmin()
                if pos_dif_min != similarity_max_idx:
                    continue
            
            object_consistency_loss += -moc_cfg.z_cos_match_weight * z_sim[similarity_max_idx] + torch.sum(z_sim)
        else:
            object_consistency_loss += -moc_cfg.z_cos_match_weight * torch.max(z_sim) + torch.sum(z_sim)
        # Remark: z_cos_match_weight - 1 corresponds to beta in the paper
        # Remark: if there is only one object in z_whats_now, the loss implicitly assumes that it is the same as the object in z_what_prior

    return object_consistency_loss, torch.tensor(len(z_pres_idx)).to(z_whats.device)



