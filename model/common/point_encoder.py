import torch
import einops
import torch.nn as nn


class PointEncoder(nn.Module):
    def __init__(self, in_shape=(), pnt_cond_steps=1,
                 hidden_dim=64, embed_dim=64, num_lyr=4):
        super().__init__()
        in_dim = in_shape[1]
        self.num_point = in_shape[0]
        self.pnt_cond_steps = pnt_cond_steps

        self.activ = nn.ReLU()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.lyrs, self.glyrs = nn.ModuleList(), nn.ModuleList()
        for i in range(num_lyr):
            self.lyrs.append(nn.Linear(hidden_dim, hidden_dim))
            self.glyrs.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.proj_out = nn.Linear(hidden_dim * num_lyr, embed_dim // pnt_cond_steps)

    def forward(self, x):
        """
        :param x: torch.tensor [b, t, l_in, d_in]
        :return: torch.tensor [b, d_out]
        """
        b, t = x.shape[:2]
        assert t == self.pnt_cond_steps
        x = einops.rearrange(x, 'b t l d -> (b t) l d')

        # sampling points
        if x.shape[2] != self.num_point:
            x = self.sampling_uniform(x)

        x = self.activ(self.proj_in(x))
        xs = []
        for (lyr, glyr) in zip(self.lyrs, self.glyrs):
            x = self.activ(lyr(x))
            gx = x.max(dim=1, keepdim=True).values
            gx = torch.cat([x, gx.expand_as(x)], dim=2)
            x = self.activ(glyr(gx))
            xs.append(x)
        x = torch.cat(xs, dim=2)
        x = self.proj_out(x)
        x = x.max(dim=1).values
        x = einops.rearrange(x, '(b t) d -> b (d t)', b=b)
        return x

    def sampling_uniform(self, x):
        b, l, d = x.shape
        if l < self.num_point:
            pad = torch.zeros(b, self.num_point - l, d).to(x.device)
            x = torch.cat([x, pad], dim=1)
        idx = torch.randperm(x.shape[1])[:self.num_point]
        return x[:, idx]


if __name__ == '__main__':
    enc = PointEncoder(in_shape=(4096, 3), pnt_cond_steps=1, hidden_dim=64, embed_dim=64, num_lyr=4)
    print(enc)
    print('param:', sum([p.data.nelement() for p in enc.parameters()]))
    x = torch.randn((1, 1, 4097, 3))
    y = enc(x)
    print('input:', x.shape, ', output:', y.shape)
    x = torch.randn((1, 1, 4095, 3))
    y = enc(x)
    print('input:', x.shape, ', output:', y.shape)
