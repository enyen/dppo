import torch
import torch.nn as nn
from einops import rearrange, reduce
from pytorch3d.ops import sample_farthest_points, knn_points
from dppo.model.common.mlp import ResidualMLP


class PointEncoder(nn.Module):
    """
    https://arxiv.org/pdf/2410.10803v1
    """
    def __init__(self, in_dim=3, pnt_cond_steps=1, augment_pnt=0.01,
                 hidden_dim=(16, 32, 64, 128), embed_dim=128, dropout=0):
        super().__init__()
        self.augment_pnt =augment_pnt
        self.pnt_cond_steps = pnt_cond_steps

        self.lyrs, self.glyrs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(hidden_dim)):
            in_dim_ = hidden_dim[i - 1] if i else in_dim
            self.lyrs.append(nn.Sequential(
                nn.Linear(in_dim_, hidden_dim[i]),
                nn.ReLU(),
            ))
            self.glyrs.append(nn.Sequential(
                nn.Linear(hidden_dim[i] * 2, hidden_dim[i]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        assert embed_dim % pnt_cond_steps == 0, 'embed_dim need to be divisible by pnt_cond_steps'
        self.proj_out = nn.Linear(sum(hidden_dim), embed_dim // pnt_cond_steps)

    def forward(self, x):
        """
        :param x: torch.tensor [b, t, l_in, d_in]
        :return: torch.tensor [b, d_out]
        """
        b, t, l, _ = x.shape
        assert t == self.pnt_cond_steps
        x = rearrange(x, 'b t l d -> (b t) l d')
        x = process_point(x, self.augment_pnt)

        xs = []
        for (lyr, glyr) in zip(self.lyrs, self.glyrs):
            x = lyr(x)
            gx = x.max(dim=1, keepdim=True).values
            gx = torch.cat([x, gx.expand_as(x)], dim=2)
            x = glyr(gx)
            xs.append(x)
        x = self.proj_out(torch.cat(xs, dim=2))
        x = x.max(dim=1).values
        x = rearrange(x, '(b t) d -> b (d t)', b=b, t=t)
        return x


class PointEncoderSA(nn.Module):
    """
    https://arxiv.org/pdf/2202.06407
    """
    def __init__(self, in_dim=3, pnt_cond_steps=1, augment_pnt=0.01,
                 hidden_dim=(16, 32, 64), embed_dim=64, dropout=(0, 0), num_head=4,
                 mul_que=0.125, mul_neb=1.25):
        super().__init__()
        self.augment_pnt = augment_pnt
        self.pnt_cond_steps = pnt_cond_steps
        self.mul_que = mul_que
        self.num_neb = int(mul_neb / mul_que)

        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU())

        self.lyrs = nn.ModuleList()
        for i in range(len(hidden_dim)):
            lyr = nn.ModuleDict({
                'sa': nn.Sequential(
                    SelfAttention(hidden_dim[i], num_head, dropout[0]),
                    FefoAttention(hidden_dim[i], hidden_dim[i] * 3, dropout[1])),
                'up': ResidualMLP([hidden_dim[i]] + [hidden_dim[i] * 2] * 4, use_layernorm=True)})
            self.lyrs.append(lyr)

        assert embed_dim % pnt_cond_steps == 0, 'embed_dim need to be divisible by pnt_cond_steps'
        self.proj_out = nn.Linear(hidden_dim[-1] * 2, embed_dim // pnt_cond_steps)

    def forward(self, x):
        """
        :param x: torch.tensor [b, t, l_in, d_in]
        :return: torch.tensor [b, d_out]
        """
        b, t, l, _ = x.shape
        assert t == self.pnt_cond_steps
        x = rearrange(x, 'b t l d -> (b t) l d')
        x = process_point(x, self.augment_pnt)

        # project in
        x = self.proj_in(x)  # [b l d]

        # SA-CNN
        for lyr in self.lyrs:
            # sample que, gather neb
            num_que = int(x.shape[1] * self.mul_que)
            x = sample_gather(x, num_que, self.num_neb)
            # SA
            x = rearrange(x, 'b q k d -> (b q) k d')
            x = lyr['sa'](x)
            x = rearrange(x, '(b q) k d -> b q k d', b=b * t, q=num_que)
            x = reduce(x, 'b q k d -> b q d', 'max')
            x = lyr['up'](x)

        # project out
        x = self.proj_out(x)
        x = reduce(x, 'b q d -> b d', 'max')
        x = rearrange(x, '(b t) d -> b (d t)', b=b, t=t)
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, layer_norm_eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt):
        identity = tgt.clone()
        tgt = self.norm(tgt)
        tgt = self.attn(tgt, tgt, tgt, need_weights=False)[0]
        tgt = self.drop(tgt)
        return tgt + identity


class FefoAttention(nn.Module):
    def __init__(self, d_model, d_fefo, dropout, layer_norm_eps=1e-5):
        super().__init__()
        self.fefo = nn.Sequential(
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            nn.Linear(d_model, d_fefo),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fefo, d_model),
            nn.Dropout(dropout))

    def forward(self, x):
        return x + self.fefo(x)


def process_point(pnt, noise=0.):
    """
    sample valid points and add noise
    :param pnt: torch.tensor [b, l, d]
    :param noise: float, noise level
    :return: torch.tensor [b, l', d]
    """
    # valid point
    msk = pnt.sum(dim=-1) != 0  # bool [b, l]
    l = msk.sum(dim=-1).min()  # min([b])
    pnt = pnt[:, :l]

    # noise
    if noise > 0:
        pnt = pnt + torch.zeros_like(pnt).uniform_(-noise, noise)
    return pnt


def sample_gather(pts, num_que, num_neb):
    """
    :param pts: torch.ndarray [b, l, d]
    :param num_que: int q
    :param num_neb: int k
    :return: neb[b, q, k, d]
    """
    # sampling query [b, q, d]
    que, _ = sample_farthest_points(pts, K=num_que, random_start_point=True)

    # grouping neb [b, q, k, d]
    neb = knn_points(que, pts, K=num_neb, return_nn=True, return_sorted=False).knn

    return neb


if __name__ == '__main__':
    enc = PointEncoder(pnt_cond_steps=1, hidden_dim=(16, 32), embed_dim=32)
    # enc = PointEncoderSA(pnt_cond_steps=1, hidden_dim=(16, 32), embed_dim=32)
    print(enc)
    print('param:', sum([p.data.nelement() for p in enc.parameters()]))
    x = torch.randn((1, 1, 2048, 3))  # [b, t, l, d]
    print('input:', x.shape)
    y = enc(x)
    print('output:', y.shape)
