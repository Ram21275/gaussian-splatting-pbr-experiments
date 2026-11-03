import math
import torch
from torch import nn
from .scene import anchors


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width); self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))
    def forward(self, x, context):
        q = self.norm1(x)
        x = x+self.self_attn(q, q, q, need_weights=False)[0]
        x = x+self.cross_attn(self.norm2(x), context, context, need_weights=False)[0]
        return x+self.ff(self.norm3(x))


class Core(nn.Module):
    """One token trunk and main/noise heads; heads exist in EVERY comparator."""
    def __init__(self, cfg):
        super().__init__()
        w = cfg["width"]
        self.input = nn.Linear(16, w)
        self.context = nn.Linear(207, w)  # 3*8*8 RGB + 4*3 camera + 3 light
        self.time = nn.Sequential(nn.Linear(16, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([Block(w, cfg["heads"]) for _ in range(cfg["depth"])])
        self.norm = nn.LayerNorm(w)
        self.main = nn.Linear(w, 13)
        self.noise = nn.Linear(w, 13)
    def forward(self, x, t, context):
        # x B,N,13; context B,V,207; t B.
        base = anchors(x.shape[1], x.device).expand(x.shape[0], -1, -1)
        freq = torch.exp(torch.arange(8, device=x.device)*(-math.log(10000)/7))
        phase = t[:, None]*freq[None]*1000
        te = self.time(torch.cat([phase.sin(), phase.cos()], -1))
        h = self.input(torch.cat([x, base], -1))+te[:, None]
        c = self.context(context)
        for block in self.blocks: h = block(h, c)
        h = self.norm(h)
        return self.main(h), self.noise(h)
