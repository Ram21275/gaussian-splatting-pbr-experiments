"""Pure-PyTorch differentiable 3D Gaussian rasterizer (dense, depth-sorted alpha compositing)
with deferred PBR shading. Representation: 3DGS fallback with flattened isotropic-disk
covariance  Sigma = s^2 (I - n n^T) + (eps*s)^2 n n^T  (always positive definite).

Dense evaluation: every pixel x every Gaussian, O(B*V*HW*N). Fine at 128px / 1024 Gaussians.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .scene import RES, pixel_rays, shade, to_srgb, LIGHTS

FLAT_EPS = 0.1
DILATE = 0.3  # px^2 low-pass, as in 3DGS
VIEW_CHUNK = 4  # views composited at once (gradient checkpointed -> bounded memory)


def _composite(u, v, z, ia, ib, ic, op, fe, res):
    """per-view sorted Gaussians (M,N) -> feature image (M,HW,C), depth (M,HW), acc (M,HW)"""
    ys, xs = torch.meshgrid(torch.arange(res, device=u.device), torch.arange(res, device=u.device), indexing="ij")
    px = xs.reshape(-1).float() + 0.5; py = ys.reshape(-1).float() + 0.5
    dx = px[None, :, None] - u[:, None, :]
    dy = py[None, :, None] - v[:, None, :]
    power = -0.5 * (ia[:, None] * dx * dx + ic[:, None] * dy * dy) - ib[:, None] * dx * dy
    alpha = (op[:, None] * torch.exp(power.clamp(max=0))).clamp(max=0.99)
    logT = torch.cumsum(torch.log1p(-alpha), dim=-1)
    T = torch.exp(logT - torch.log1p(-alpha))  # exclusive transmittance
    w = alpha * T
    return w @ fe, (w * z[:, None]).sum(-1), w.sum(-1)


def covariance(normal, scale):
    """normal (...,3) unit, scale (...,) -> (...,3,3)"""
    n = normal[..., :, None]
    nnT = n @ n.transpose(-1, -2)
    I = torch.eye(3, device=normal.device, dtype=normal.dtype)
    s2 = (scale ** 2)[..., None, None]
    return s2 * (I - nnT) + (FLAT_EPS ** 2) * s2 * nnT


def rasterize(means, cov, opacity, feats, w2c, K, res=RES, chunk_views=None):
    """means (B,N,3) cov (B,N,3,3) opacity (B,N) feats (B,N,C); w2c (B,V,4,4), K (3,3).
    Returns feat image (B,V,HW,C), depth (B,V,HW), acc (B,V,HW)."""
    B, N, _ = means.shape
    V = w2c.shape[1]
    R = w2c[..., :3, :3]; t = w2c[..., :3, 3]                       # (B,V,3,3),(B,V,3)
    pc = torch.einsum("bvij,bnj->bvni", R, means) + t[:, :, None]     # (B,V,N,3)
    z = pc[..., 2].clamp(min=1e-3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pc[..., 0] / z + cx
    v = fy * pc[..., 1] / z + cy
    J = torch.zeros(B, V, N, 2, 3, device=means.device, dtype=means.dtype)
    J[..., 0, 0] = fx / z; J[..., 0, 2] = -fx * pc[..., 0] / z ** 2
    J[..., 1, 1] = fy / z; J[..., 1, 2] = -fy * pc[..., 1] / z ** 2
    Rc = R[:, :, None]                                                # (B,V,1,3,3)
    Sc = Rc @ cov[:, None] @ Rc.transpose(-1, -2)                     # (B,V,N,3,3)
    S2 = J @ Sc @ J.transpose(-1, -2)
    a = S2[..., 0, 0] + DILATE; b = S2[..., 0, 1]; c = S2[..., 1, 1] + DILATE
    det = (a * c - b * b).clamp(min=1e-8)
    ia, ib, ic = c / det, -b / det, a / det                           # conic
    valid = (pc[..., 2] > 0.2).float()

    # sort by depth (front to back) per view
    order = z.argsort(dim=-1)
    g = lambda x: torch.gather(x, -1, order)
    u, v, z, ia, ib, ic, valid = map(g, (u, v, z, ia, ib, ic, valid))
    op = torch.gather(opacity[:, None].expand(B, V, N), -1, order)
    fe = torch.gather(feats[:, None].expand(B, V, N, feats.shape[-1]), 2,
                      order[..., None].expand(B, V, N, feats.shape[-1]))

    BV = B * V
    flat = lambda x: x.reshape(BV, *x.shape[2:])
    args = [flat(x) for x in (u, v, z, ia, ib, ic, op * valid, fe)]
    outs = []
    for s in range(0, BV, VIEW_CHUNK):
        a = [x[s:s + VIEW_CHUNK] for x in args]
        if torch.is_grad_enabled() and means.requires_grad:
            outs.append(checkpoint(_composite, *a, res, use_reentrant=False))
        else:
            outs.append(_composite(*a, res))
    img = torch.cat([o[0] for o in outs]).reshape(B, V, res * res, -1)
    depth = torch.cat([o[1] for o in outs]).reshape(B, V, res * res)
    acc = torch.cat([o[2] for o in outs]).reshape(B, V, res * res)
    return img, depth, acc


def decode_render(g, w2c, K, light_names=("L0_train",), res=RES):
    """g: decoded gaussian dict (B,N,...). Returns dict of (B,V,H,W,C) maps."""
    feats = torch.cat([g["albedo"], g["rough"], g["metal"], g["normal"]], -1)
    cov = covariance(g["normal"], g["scale"])
    img, depth, acc = rasterize(g["xyz"], cov, g["opacity"], feats, w2c, K, res)
    accd = acc[..., None].clamp(min=1e-4)
    alb = img[..., 0:3] / accd; ro = img[..., 3:4] / accd; me = img[..., 4:5] / accd
    nrm = F.normalize(img[..., 5:8], dim=-1)
    B, V = w2c.shape[:2]
    o, d, _ = pixel_rays(w2c.reshape(-1, 4, 4), K, res)
    d = d.reshape(B, V, -1, 3)
    out = dict(acc=acc[..., None], depth=depth[..., None] / accd, albedo=alb * acc[..., None],
               rough=ro * acc[..., None], metal=me * acc[..., None], normal=nrm * acc[..., None])
    out["depth"] = out["depth"] * acc[..., None]   # "premultiplied" depth, like GT depth*mask
    for ln in light_names:
        rad = shade(alb.clamp(0, 1), ro.clamp(0, 1), me.clamp(0, 1), nrm, -d, LIGHTS[ln])
        out["rgb_" + ln] = to_srgb(rad) * acc[..., None]
    return {k: v.reshape(B, V, res, res, -1) for k, v in out.items()}
