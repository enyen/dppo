"""
source: https://github.com/YanjieZe/Improved-3D-Diffusion-Policy/blob/main/Improved-3D-Diffusion-Policy/diffusion_policy_3d/model/vision_3d/multi_stage_pointnet.py
"""
import torch
import torch.nn as nn


class PointEncoder(nn.Module):
    def __init__(self, in_shape=(), hidden_dim=128, embed_dim=128, num_lyr=4):
        super().__init__()
        in_dim = in_shape[0]
        self.num_point = in_shape[1]

        self.act = nn.ReLU()
        self.conv_in = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.lyrs, self.glyrs = nn.ModuleList(), nn.ModuleList()
        for i in range(num_lyr):
            self.lyrs.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))
            self.glyrs.append(nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(hidden_dim * num_lyr, embed_dim, kernel_size=1)

    def forward(self, x):
        """
        :param x: torch.tensor [b, d_in, l_in]
        :return: torch.tensor [b, d_out]
        """
        # sampling points
        len_in = x.shape[2]
        if len_in != self.num_point:
            x = self.sampling_uniform(x)

        x = self.act(self.conv_in(x))
        xs = []
        for (lyr, glyr) in zip(self.layers, self.glayers):
            x = self.act(lyr(x))
            gx = x.max(dim=-1, keepdim=True).values
            gx = torch.cat([x, gx.expand_as(x)], dim=1)
            x = self.act(glyr(gx))
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.conv_out(x)
        x = x.max(dim=-1).values
        return x

    def sampling_uniform(self, x):
        b, d, l = x.shape
        if l < self.num_point:
            pad = torch.zeros(b, d, self.num_point - l).to(x.device)
            x = torch.cat([x, pad], dim=2)
        idx = torch.randperm(x.shape[2])[:self.num_point]
        return x[..., idx]
