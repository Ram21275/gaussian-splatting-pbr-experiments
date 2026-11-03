"""Single generative trunk shared by A0/A1/A2: anchor-token transformer with
self-attention over 1024 aligned anchors, camera-aware cross-attention to cached frozen
DINOv2 tokens of the 3 input views (+ Plucker ray embedding), adaLN time conditioning.
Only the output-parameterisation/objective differs between variants."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .latent import C, fibonacci_sphere
from .scene import INPUT_CAMS, cam_tensors, pixel_rays


def fourier(x, n_freq=8):
    f = 2.0 ** torch.arange(n_freq, device=x.device) * math.pi
    a = x[..., None] * f
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


def timestep_embed(t, dim=256):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = (t[:, None] * 1000.0) * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


def plucker_patch_rays(grid=16):
    """Plucker coordinates (d, o x d) for each DINO patch centre of the 3 input cameras + cls(0).
    Returns (3, 1+grid*grid, 6)."""
    w2c, K = cam_tensors(INPUT_CAMS)
    o, d, _ = pixel_rays(w2c, K, res=128)
    d = d.reshape(3, 128, 128, 3)
    step = 128 / grid
    idx = (torch.arange(grid) * step + step / 2).long()
    d = d[:, idx][:, :, idx].reshape(3, grid * grid, 3)
    m = torch.cross(o[:, None].expand_as(d), d, dim=-1)
    pl = torch.cat([d, m], -1)
    return torch.cat([torch.zeros(3, 1, 6), pl], 1)


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.ada = nn.Linear(d, 6 * d)  # shift/scale/gate for SA and MLP
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, temb):
        s1, g1, b1, s2, g2, b2 = self.ada(temb)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1.tanh().add(1) * self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.ca(h, c, c, need_weights=False)[0]
        h = self.n3(x) * (1 + s2) + b2
        return x + g2.tanh().add(1) * self.mlp(h)


class Trunk(nn.Module):
    def __init__(self, d=256, depth=4, heads=4, feat_dim=384, n_anchor=1024, out_heads=1):
        super().__init__()
        self.register_buffer("dirs", fibonacci_sphere(n_anchor))
        self.register_buffer("plk", plucker_patch_rays())
        self.inp = nn.Linear(C, d)
        self.pos = nn.Sequential(nn.Linear(3 + 3 * 2 * 8, d), nn.GELU(), nn.Linear(d, d))
        self.tmlp = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d))
        self.cproj = nn.Linear(feat_dim, d)
        self.cpl = nn.Sequential(nn.Linear(6 * 17, d), nn.GELU(), nn.Linear(d, d))
        self.view_emb = nn.Parameter(torch.zeros(3, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.nout = nn.LayerNorm(d)
        # out_heads: 1 (velocity / x0) ; B1/B2 later use 2 (velocity + score) on the same trunk
        self.out = nn.Linear(d, C * out_heads)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def cond(self, feats):
        """feats (B,3,257,384) cached frozen DINOv2 tokens -> (B,3*257,d)"""
        B = feats.shape[0]
        c = self.cproj(feats.float()) + self.cpl(self._pl()) + self.view_emb
        return c.reshape(B, -1, c.shape[-1])

    def _pl(self):
        p = self.plk  # (3,257,6) -> raw + 8-band sin features (first 96) = 102 dims
        f = 2.0 ** torch.arange(8, device=p.device) * math.pi
        a = p[..., None] * f
        return torch.cat([p, a.sin().flatten(-2)[..., :6 * 16]], -1)  # 6 + 96 = 102 = 6*17

    def forward(self, z, t, feats):
        B = z.shape[0]
        c = self.cond(feats)
        temb = self.tmlp(timestep_embed(t))
        x = self.inp(z) + self.pos(fourier(self.dirs, 8))[None]
        for blk in self.blocks:
            x = blk(x, c, temb)
        return self.out(self.nout(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
