import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from .resnet import resnet18
from .ConvGRU import *
from math import ceil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("DEVICE: ", device)

'''
Slot Attention module
differences from the original repo:
1. learnable slot initializtaion
2. pad for the first frame
Inputs --> [batch_size, number_of_frame, sequence_length, hid_dim]
outputs --> slots[batch_size, number_of_frame, num_of_slots, hid_dim]; attn_masks [batch_size, number_of_frame, num_of_slots, sequence_length]
'''


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = torch.randn(1, 1, dim).to(device)
        self.slots_sigma = torch.randn(1, 1, dim).to(device)

        self.slots_sigma = self.slots_sigma.absolute()

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        mu = self.slots_mu.expand(1, self.num_slots, -1)
        sigma = self.slots_sigma.expand(1, self.num_slots, -1)
        slots = torch.normal(mu, sigma)

        slots = slots.contiguous()
        self.register_buffer("slots", slots)

    def get_attention(self, slots, inputs):
        slots_prev = slots
        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots=None):
        b, nf,  n, d = inputs.shape
        slots_out = []
        attns = []
        slots = self.slots.expand(b, -1, -1)
        # pre-attention for the first frame
        slots, _ = self.get_attention(slots, inputs[:, 0, :, :])
        for f in range(nf):
            cur_slots, cur_attn = self.get_attention(slots, inputs[:, f, :, :])
            slots_out.append(cur_slots)
            attns.append(cur_attn)
            slots = cur_slots
        slots_out = torch.stack([slot for slot in slots_out])
        slots_out = slots_out.permute(1, 0, 2, 3)
        attns = torch.stack([attn for attn in attns])
        attns = attns.permute(1, 0, 2, 3)
        return slots_out, attns

    def infer(self, inputs, num_slots=None, slot_state=None):
        b, nf,  n, d = inputs.shape
        slots_out = []
        attns = []
        slots = self.slots.expand(b, -1, -1)
        if slot_state is None:
            slots, _ = self.get_attention(slots, inputs[:, 0, :, :])
        else:
            slots = slot_state
        for f in range(nf):
            cur_slots, cur_attn = self.get_attention(slots, inputs[:, f, :, :])
            slots_out.append(cur_slots)
            attns.append(cur_attn)
            slots = cur_slots
        slots_out = torch.stack([slot for slot in slots_out])
        slots_out = slots_out.permute(1, 0, 2, 3)
        attns = torch.stack([attn for attn in attns])
        attns = attns.permute(1, 0, 2, 3)
        return slots_out, attns, slots


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1).to(device)


"""Adds soft positional embedding with learnable projection."""


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


'''
encoder
input: image [bs, num_frames, n_channels, H, W]
outputs: features [bs*num_frames, ceil(H*W // 16),hid_dim], 16 is the (downsample ratio)**2 
'''


class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim, device='cuda'):
        super().__init__()
        self.resolution = resolution
        self.hid_dim = hid_dim
        self.resnet = resnet18(pretrained=False)
        dtype = torch.FloatTensor
        if device == 'cuda':
            dtype = torch.cuda.FloatTensor
        self.convGRU = ConvGRU(input_size=(ceil(resolution[0]/4), ceil(resolution[1]/4)),
                               input_dim=512,
                               hidden_dim=hid_dim,
                               kernel_size=(3, 3),
                               num_layers=2,
                               dtype=dtype,
                               batch_first=True,
                               bias=True,
                               return_all_layers=False)
        self.encoder_pos = SoftPositionEmbed(
            hid_dim, (ceil(resolution[0]/4), ceil(resolution[1]/4)))

    def forward(self, x):
        bs, n_frames, N, H, W = x.shape
        x = x.view(-1, N, H, W)
        x = self.resnet(x)
        x = x.view(bs, n_frames, x.shape[1], x.shape[2], x.shape[3])
        x, _ = self.convGRU(x)
        x = x[0]
        x = x.view(bs*n_frames, -1, x.shape[3], x.shape[4])
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x

    def infer(self, x, h=None):
        bs, n_frames, N, H, W = x.shape
        x = x.view(-1, N, H, W)
        x = self.resnet(x)
        x = x.view(bs, n_frames, x.shape[1], x.shape[2], x.shape[3])
        x, _, h_out = self.convGRU.infer(x, h)
        x = x[0]
        x = x.view(bs*n_frames, -1, x.shape[3], x.shape[4])
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x, h_out


class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution, output_channel):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            hid_dim, hid_dim, 4, stride=(2, 2), padding=1)
        self.conv2 = nn.ConvTranspose2d(
            hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv3 = nn.ConvTranspose2d(
            hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv4 = nn.ConvTranspose2d(
            hid_dim, hid_dim, 4, stride=(2, 2), padding=1)
        self.conv5 = nn.ConvTranspose2d(
            hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(
            hid_dim, output_channel, 3, stride=(1, 1), padding=1)
        self.resolution = resolution
        self.output_channel = output_channel

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:, :, :self.resolution[0], :self.resolution[1]]
        return x


"""Slot Attention-based auto-encoder for object discovery."""


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, hid_dim, output_channel=3, device='cuda'):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        hid_dim: dimension for ConvGRU and slot attention
        dresolution: downsampled resolution --> resnet 18 downsampled with a factor of 4
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.dresolution = (ceil(resolution[0]/4), ceil(resolution[1]/4))
        self.num_slots = num_slots
        self.output_channel = output_channel
        self.device = device

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim, device=self.device)
        self.decoder = Decoder(
            self.hid_dim, self.resolution, self.output_channel)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            eps=1e-8,
            hidden_dim=128)
        self.LN = nn.LayerNorm(
            [self.dresolution[0] * self.dresolution[1], hid_dim])

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].
        bs, n_frames, dim, him, wim = image.shape
        x = self.encoder_cnn(image)
        x = self.LN(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x shape: [bs * n_frames, width*height // 16, hiden_dim].
        _, H, W = x.shape
        x = x.view(bs, n_frames, H, W)

        # Slot Attention module.
        slots_ori, attn_masks = self.slot_attention(x)
        # reshape and broadcaast attention masks
        attn_masks = attn_masks.reshape(bs*n_frames, self.num_slots, -1)
        attn_masks = attn_masks.view(
            attn_masks.shape[0], attn_masks.shape[1], self.dresolution[0], self.dresolution[1])
        attn_masks = attn_masks.unsqueeze(-1)

        attn_masks = attn_masks.reshape(bs*n_frames, self.num_slots, -1)
        attn_masks = attn_masks.view(
            attn_masks.shape[0], attn_masks.shape[1], self.dresolution[0], self.dresolution[1])
        attn_masks = attn_masks.unsqueeze(-1)

        slots = slots_ori.reshape(bs*n_frames, self.num_slots, -1)
        slots = slots.unsqueeze(2).unsqueeze(3)
        slots_combine = slots * attn_masks
        slots_combine = slots_combine.sum(dim=1)
        slots_combine = slots_combine.permute(0, 3, 1, 2)

        # `slots_combine` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        recons = self.decoder(slots_combine)
        # `recons` has shape: [bs, n_frames, 3, H//4, W // 4]

        return recons, attn_masks.view(bs, n_frames, self.num_slots, attn_masks.shape[2], attn_masks.shape[3]), \
               slots_combine, slots_ori


class MLP_Classifier(nn.Module):
    """
    The classifier of the set prediction architecture of Locatello et al. 2020
    """
    def __init__(self, in_channels, out_channels):
        """
        Builds the classifier for the set prediction architecture.
        :param in_channels: Integer, input channel dimensions
        :param out_channels: Integer, output channel dimensions
        """
        super(MLP_Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
