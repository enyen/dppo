import torch
import torch.nn as nn
from einops import rearrange, reduce
from pytorch3d.ops import sample_farthest_points, ball_query, knn_points


class PointEncoder(nn.Module):
    """
    https://arxiv.org/pdf/2410.10803v1
    """
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
        x = rearrange(x, 'b t l d -> (b t) l d')

        # sampling points
        if x.shape[2] != self.num_point:
            x = sampling_uniform(x, self.num_point)

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
        x = rearrange(x, '(b t) d -> b (d t)', b=b, t=t)
        return x


class PointEncoderSA(nn.Module):
    """
    https://arxiv.org/pdf/2202.06407
    """
    def __init__(self, in_shape=(), pnt_cond_steps=1,
                 hidden_dim=64, embed_dim=64, num_lyr=3, dropout=(0, 0), num_head=4,
                 mul_que=0.0625, mul_neb=1.25):
        super().__init__()
        in_dim = in_shape[1]
        self.num_point = in_shape[0]
        self.pnt_cond_steps = pnt_cond_steps
        self.mul_que = mul_que
        self.num_neb = int(mul_neb / mul_que)

        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU())

        self.lyrs = nn.ModuleList()
        for i in range(num_lyr):
            self.lyrs.append(nn.Sequential(
                SelfAttention(hidden_dim, num_head, dropout[0]),
                FefoAttention(hidden_dim, hidden_dim * 2, dropout[1])))

        self.proj_out = nn.Linear(hidden_dim, embed_dim // pnt_cond_steps)

    def forward(self, x):
        """
        :param x: torch.tensor [b, t, l_in, d_in]
        :return: torch.tensor [b, d_out]
        """
        b, t = x.shape[:2]
        assert t == self.pnt_cond_steps
        x = rearrange(x, 'b t l d -> (b t) l d')

        # sampling points
        if x.shape[2] != self.num_point:
            x = sampling_uniform(x, self.num_point)

        # project in
        x = self.proj_in(x)  # [b l d]

        # SA-CNN
        for lyr in self.lyrs:
            # sample que, gather neb
            num_que = int(x.shape[1] * self.mul_que)
            x = sample_gather(x, num_que, self.num_neb)
            # SA
            x = rearrange(x, 'b q k d -> (b q) k d')
            x = lyr(x)
            x = rearrange(x, '(b q) k d -> b q k d', b=b, q=num_que)
            x = reduce(x, 'b q k d -> b q d', 'max')

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
        nn.attention.enable_mem_efficient_sdp(False)

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


def sampling_uniform(x, num_point):
    b, l, d = x.shape
    if l < num_point:
        pad = torch.zeros(b, num_point - l, d).to(x.device)
        x = torch.cat([x, pad], dim=1)
    idx = torch.randperm(x.shape[1])[:num_point]
    return x[:, idx]


def sample_gather(pts, num_que, num_neb):
    """
    :param pts: torch.ndarray [b, l, d]
    :param num_que: int q
    :param num_neb: int k
    :return: neb[b, q, k, d]
    """
    # sampling query [b, q, d]
    que, _ = sample_farthest_points(pts, K=num_que)

    # grouping neb [b, q, k, d]
    neb = knn_points(que, pts, K=num_neb, return_nn=True, return_sorted=False).knn

    return neb


if __name__ == '__main__':
    # enc = PointEncoder(in_shape=(4096, 3), pnt_cond_steps=1, hidden_dim=64, embed_dim=64, num_lyr=4)
    enc = PointEncoderSA(in_shape=(4096, 3), pnt_cond_steps=1, hidden_dim=64, embed_dim=64, num_lyr=3)
    print(enc)
    print('param:', sum([p.data.nelement() for p in enc.parameters()]))
    x = torch.randn((1, 1, 4097, 3))
    y = enc(x)
    print('input:', x.shape, ', output:', y.shape)
    x = torch.randn((1, 1, 4095, 3))
    y = enc(x)
    print('input:', x.shape, ', output:', y.shape)
