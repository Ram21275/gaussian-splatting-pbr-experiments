"""Aligned anchor latent and the shared deterministic decoder D.

Anchor identity: anchor i <-> fixed Fibonacci-sphere direction u_i (same for every object).
Initial target: outermost surface hit of the ray from 2*u_i towards the origin (body primitive
contains the origin, so every ray hits). A short full-view teacher fit then refines all channels
while keeping index i (and a soft tie to its ray-hit position), so z1[i] is token-aligned across objects.

Raw per-anchor channels (C=13):
  0:3 xyz | 3:6 normal (unnormalised) | 6 log scale | 7 opacity logit |
  8:11 albedo logit | 11 roughness logit | 12 metallic logit
Standardised latent z = (raw - mu) / sd with mu/sd from TRAINING objects only.
D(z): unstandardise -> normalise normal, exp scale, sigmoid opacity/materials.
"""
import math
import torch
import torch.nn.functional as F
from .scene import ObjectSDF

C = 13
N_ANCHORS = 1024
LOGIT_CLAMP = 4.0


def fibonacci_sphere(n=N_ANCHORS):
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return torch.stack([torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi), torch.cos(phi)], -1).float()


def _logit(x):
    x = x.clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x)).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)


@torch.no_grad()
def init_raw_latent(obj, device, n=N_ANCHORS):
    sdf = ObjectSDF(obj, device)
    u = fibonacci_sphere(n).to(device)
    o = 2.0 * u
    hit, t = sdf.raymarch(o, -u, tmin=0.0, tmax=2.0, steps=256)
    X = o - t[:, None] * u
    nrm = sdf.normal(X)
    alb, ro, me = sdf.material(X)
    # scale from mean distance to 3 nearest neighbours
    D = torch.cdist(X, X); D.fill_diagonal_(1e9)
    s = D.topk(3, largest=False).values.mean(-1) * 0.7
    raw = torch.cat([X, nrm, s.log()[:, None], torch.full((n, 1), 3.0, device=device),
                     _logit(alb), _logit(ro), _logit(me)], -1)
    return raw, hit


def decode_raw(raw):
    """raw (...,C) -> gaussian dict"""
    return dict(xyz=raw[..., 0:3], normal=F.normalize(raw[..., 3:6], dim=-1),
                scale=raw[..., 6].clamp(-7, 0).exp(), opacity=torch.sigmoid(raw[..., 7]),
                albedo=torch.sigmoid(raw[..., 8:11]), rough=torch.sigmoid(raw[..., 11:12]),
                metal=torch.sigmoid(raw[..., 12:13]))


class Standardizer(torch.nn.Module):
    def __init__(self, mu=None, sd=None):
        super().__init__()
        self.register_buffer("mu", torch.zeros(C) if mu is None else mu)
        self.register_buffer("sd", torch.ones(C) if sd is None else sd)

    @classmethod
    def fit(cls, raws):  # raws (M,N,C) training objects only
        x = raws.reshape(-1, C)
        return cls(x.mean(0), x.std(0).clamp(min=1e-3))

    def fwd(self, raw):
        return (raw - self.mu) / self.sd

    def inv(self, z):
        return z * self.sd + self.mu


def D(z, stdz):
    """The shared frozen decoder: standardised latent -> Gaussian/PBR parameters."""
    return decode_raw(stdz.inv(z))
