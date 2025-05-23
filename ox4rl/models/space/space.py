import torch
import torch.nn as nn
from ox4rl.configs.space_cfg import space_cfg
from .fg import SpaceFg
from .bg import SpaceBg
from collections import defaultdict


class Space(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.fg_module = SpaceFg()
        self.bg_module = SpaceBg()
        self.arch_type = "space"

    # @profile
    def forward(self, x, global_step=100000000):
        """
        Inference.
        With time-dimension for consistency
        :param x: (B, 3, H, W) # being 128 x 128
        :param global_step: global training step
        :return:
            loss: a scalar. Note it will be better to return (B,)
            log: a dictionary for visualization
        """

        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        bg_likelihood, bg, kl_bg, log_bg = self.bg_module(x, global_step)

        # Foreground extraction
        fg_likelihood, fg, alpha_map, kl_fg, log_fg = self.fg_module(x, global_step)

        # Fix alpha trick
        if global_step and global_step < space_cfg.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, space_cfg.fix_alpha_value)

        # Compute final mixture likelihood
        # (B, 3, H, W)
        fg_likelihood = (fg_likelihood + (alpha_map + 1e-5).log())
        bg_likelihood = (bg_likelihood + (1 - alpha_map + 1e-5).log())
        # (B, 2, 3, H, W)
        log_like = torch.stack((fg_likelihood, bg_likelihood), dim=1)
        # (B, 3, H, W)
        log_like = torch.logsumexp(log_like, dim=1)
        # (B,)
        log_like = log_like.flatten(start_dim=1).sum(1)

        # Take mean as reconstruction
        y = alpha_map * fg + (1.0 - alpha_map) * bg

        # Elbo
        elbo = log_like - kl_bg - kl_fg

        # Sum over batch
        loss = -elbo.sum()
        x = x[:, :3]
        log = {
            'imgs': x,
            'y': y,
            # (B,)
            'mse': ((y - x) ** 2).flatten(start_dim=1).sum(dim=1),
            'log_like': log_like,
            'elbo_loss': loss
        }
        log.update(log_fg)
        log.update(log_bg)

        return loss, log

    def scene_description(self, x, z_classifier, only_z_what=True, size=True, **kwargs):
        """
        Computes a dictionary where each entity is located
        :param x: (3, H, W)
        :param z_classifier: a classifier mapping encodings (z_where, z_depth and z_what concatenated)
            to labels. Has to be able to do z_classifier.predict(list of z_encodings) to return a list of labels
        :param only_z_what: predicts labels only on active (z_pres) z_what encodings, but disregards z_where / z_depth
        :param size: if True, also returns the size encoding of the objects
        :return: dict[label, list of (int, int)], all positions of the object in question
        """
        _, log = self.forward(x.unsqueeze(dim=0))
        z_where, z_pres_prob, z_what, z_depth = log['z_where'], log['z_pres_prob'], log['z_what'], log['z_depth']
        z_where, z_what, z_depth = z_where.squeeze().detach().cpu(), z_what.squeeze().detach().cpu(), z_depth.squeeze().detach().cpu()
        z_pres_prob = z_pres_prob.squeeze().detach().cpu()
        z_pres = z_pres_prob > 0.5 # TODO: fix code for case when z_pres_prob <= 0.5 everywhere
        if size:
            pos = z_where[z_pres]
        else:
            pos = z_where[z_pres][:,:2]
        if only_z_what:
            z_encs = z_what[z_pres]
        else:
            z_encs = torch.concat((z_where[z_pres], z_depth[z_pres], z_what[z_pres]), dim=1)
        if z_classifier is None:
            return z_encs, pos
        labels = z_classifier.predict(z_encs, **kwargs)
        result = defaultdict(list)
        for label, p in zip(labels, pos):
            result[label].append(tuple(coordinate.item() for coordinate in p))
        return result

    def __repr__(self):
        return f"{self.arch_type} unsupervised object detection model"

    def state_dict(self, destination=None, *args, **kwargs):
        _state_dict = super().state_dict(destination, *args, **kwargs)
        _state_dict["arch_type"] = self.arch_type
        return _state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        if "arch_type" in state_dict.keys():
            self.arch_type = state_dict.pop("arch_type")
        return super().load_state_dict(state_dict, *args, **kwargs)
