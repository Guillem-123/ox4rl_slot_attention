import torch

from ox4rl.configs.moc_cfg import moc_cfg
from ox4rl.configs.space_cfg import space_cfg

class MOC_Loss_Scheduler:
    def get_value(self, **kwargs):
        raise NotImplementedError

class Linear_MOC_Loss_Scheduler(MOC_Loss_Scheduler):

    def get_value(self, **kwargs):
        global_step = kwargs['global_step']
        flow_scaling = max(0, 1 - (global_step - moc_cfg.motion_cooling_start_step) / moc_cfg.motion_cooling_end_step)
        area_object_scaling = 1 - flow_scaling
        lambda_align = area_object_scaling
        return lambda_align
    
class Dynamic_MOC_Loss_Scheduler(MOC_Loss_Scheduler):
    def __init__(self):
        self.count_difference_variance = RunningVariance()

    def get_value(self, **kwargs):
        logs = kwargs['logs']
        object_count_accurate = self.object_count_accurate_scaling(logs)
        area_object_scaling = moc_cfg.dynamic_steepness ** (-object_count_accurate)
        lambda_align = area_object_scaling
        return lambda_align
   
    def object_count_accurate_scaling(self, responses):
        # (T, B, G, G, 5)
        z_whats = responses['z_what']
        _, GG, D = z_whats.shape
        # (B, T, G*G, 1)
        z_whats = z_whats.reshape(-1, space_cfg.T, GG, D)
        B, T = z_whats.shape[:2]
        z_where_pure = responses['z_where_pure'].reshape(B, T, space_cfg.G, space_cfg.G, -1).transpose(0, 1)
        z_pres_pure = responses['z_pres_prob_pure'].reshape(B, T, GG, 1).reshape(B, T, space_cfg.G, space_cfg.G).transpose(0, 1)
        motion_z_where = responses['motion_z_where'].reshape(B, T, space_cfg.G, space_cfg.G, -1).transpose(0, 1)
        motion_z_pres = responses['motion_z_pres'].reshape(B, T, GG, 1).reshape(B, T, space_cfg.G, space_cfg.G).transpose(0, 1)
        frame_marker = (torch.arange(T * B, device=z_pres_pure.device) * 10).reshape(T, B)[..., None, None].expand(T, B, space_cfg.G, space_cfg.G)[..., None]
        z_where_pad = torch.cat([z_where_pure, frame_marker], dim=4)
        motion_pad = torch.cat([motion_z_where, frame_marker], dim=4)
        # (#hits, 5)
        objects = z_where_pad[z_pres_pure > moc_cfg.object_threshold]
        motion_objects = motion_pad[motion_z_pres > moc_cfg.object_threshold]
        if objects.nelement() == 0 or motion_objects.nelement() == 0:
            return 1000

        # (#objects, #motion_objects)
        dist = torch.cdist(motion_objects, objects, p=1)
        # (#hits, 1)
        values, _ = torch.topk(dist, 1, largest=False)
        motion_z_pres = responses['motion_z_pres'].mean().detach().item()
        pred_z_pres = responses['z_pres_prob_pure'].mean().detach().item()
        zero = 0

        motion_objects_found = values.mean().detach().item()
        # print(f'{motion_objects_found=}')
        value = (max(zero, (motion_objects_found - moc_cfg.z_where_offset) * moc_cfg.motion_object_found_lambda) +
                 max(zero, pred_z_pres - motion_z_pres * moc_cfg.motion_underestimating))
        self.count_difference_variance += pred_z_pres - motion_z_pres
        return (value + self.count_difference_variance.value() * moc_cfg.use_variance)\
               * responses['motion_z_pres'][0].nelement()
    
class RunningVariance:
    def __init__(self, n=moc_cfg.variance_steps):
        self.values = []
        self.n = n

    def __add__(self, other):
        if not self.values:
            self.values.append(other)
        self.values.append(other)
        self.values = self.values[-self.n:]
        return self

    def value(self):
        return torch.var(torch.tensor(self.values), unbiased=True)


