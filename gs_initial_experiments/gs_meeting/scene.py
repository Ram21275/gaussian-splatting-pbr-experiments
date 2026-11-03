"""Fixed decoder and small differentiable isotropic Gaussian reference renderer.

Perspective footprints use the center-depth approximation and one-sided normal culling. GGX directional-light shading
has no shadows, indirect lighting, antialiasing or visibility-gradient correction.
Never describe this renderer as the DreamLifting renderer or production 2DGS.
"""
import math
import torch
import torch.nn.functional as F

CHANNELS = ["dx", "dy", "dz", "scale", "nx", "ny", "nz",
            "albedo_r", "albedo_g", "albedo_b", "roughness", "metallic", "opacity"]


def anchors(n, device="cpu"):
    i = torch.arange(n, device=device, dtype=torch.float32) + .5
    y = 1 - 2*i/n
    r = (1-y*y).clamp_min(0).sqrt()
    a = i * (math.pi * (3-math.sqrt(5)))
    return torch.stack([r*a.cos(), y, r*a.sin()], -1)


def decode(raw):
    """raw [...,N,13] -> bounded attributes; same analytic decoder for all runs."""
    base = anchors(raw.shape[-2], raw.device)
    normal = F.normalize(base + .6*raw[..., 4:7].tanh(), dim=-1)
    return {"position": .7*base + .25*raw[..., :3].tanh(),
            "scale": .025 + .11*raw[..., 3:4].sigmoid(),
            "normal": normal,
            "albedo": .05 + .9*raw[..., 7:10].sigmoid(),
            "roughness": .08 + .84*raw[..., 10:11].sigmoid(),
            "metallic": raw[..., 11:12].sigmoid(),
            "opacity": .05 + .9*raw[..., 12:13].sigmoid()}


def camera(azimuth, elevation=.25, radius=3.):
    eye = torch.tensor([radius*math.cos(elevation)*math.sin(azimuth),
                        radius*math.sin(elevation),
                        radius*math.cos(elevation)*math.cos(azimuth)])
    forward = F.normalize(-eye, dim=0)
    right = F.normalize(torch.linalg.cross(forward, torch.tensor([0., 1., 0.])), dim=0)
    up = torch.linalg.cross(right, forward)
    return torch.cat([torch.stack([right, up, forward]), eye[None]], 0)


def _shade(s, eye, light):
    n = s["normal"]
    l = F.normalize(light, dim=-1).expand_as(n)
    v = F.normalize(eye-s["position"], dim=-1)
    h = F.normalize(l+v, dim=-1)
    nl = (n*l).sum(-1, keepdim=True).clamp(0, 1)
    nv = (n*v).sum(-1, keepdim=True).clamp_min(1e-4)
    nh = (n*h).sum(-1, keepdim=True).clamp(0, 1)
    vh = (v*h).sum(-1, keepdim=True).clamp(0, 1)
    a2 = s["roughness"].pow(4)
    distribution = a2/(math.pi*(nh.square()*(a2-1)+1).square()+1e-6)
    k = (s["roughness"]+1).square()/8
    masking = nv/(nv*(1-k)+k) * nl/(nl*(1-k)+k+1e-6)
    f0 = .04*(1-s["metallic"]) + s["metallic"]*s["albedo"]
    fresnel = f0+(1-f0)*(1-vh).pow(5)
    spec = distribution*masking*fresnel/(4*nl*nv+1e-5)
    diffuse = (1-fresnel)*(1-s["metallic"])*s["albedo"]/math.pi
    # Linear RGB. Constant ambient is an explicit approximation, not path tracing.
    return (.08*s["albedo"] + 2.4*nl*(diffuse+spec)).clamp(0, 1)


def render(raw, cam, light, size, return_weights=False):
    """One object/camera; front-to-back sorted dense splats. Memory O(N*H*W)."""
    s = decode(raw)
    pos = s["position"]
    xyz = (pos-cam[3]) @ cam[:3].T
    z = xyz[:, 2].clamp_min(.05)
    focal = .5*size/math.tan(math.radians(45)/2)
    xy = torch.stack([focal*xyz[:, 0]/z+(size-1)/2,
                      -focal*xyz[:, 1]/z+(size-1)/2], -1)
    yy, xx = torch.meshgrid(torch.arange(size, device=raw.device),
                            torch.arange(size, device=raw.device), indexing="ij")
    d2 = (xx[None]-xy[:, 0, None, None]).square()+(yy[None]-xy[:, 1, None, None]).square()
    radius = (focal*s["scale"][:, 0]/z).clamp_min(.5)
    alpha = s["opacity"][:, 0, None, None]*torch.exp(-.5*d2/radius[:, None, None].square())
    # Surface proxy: suppress back-facing anchors so the pilot has genuinely
    # unobserved back surfaces. This differs from ordinary two-sided 3DGS.
    facing = (s["normal"]*F.normalize(cam[3]-pos, dim=-1)).sum(-1)>0
    alpha = (alpha * ((xyz[:, 2]>.05)&facing)[:, None, None]).clamp(0, .995)
    order = z.argsort()
    a = alpha[order]
    trans = torch.cumprod(torch.cat([torch.ones_like(a[:1]), 1-a+1e-8], 0), 0)[:-1]
    weights = a*trans
    acc = weights.sum(0)
    def composite(v):
        return torch.einsum("nhw,nc->hwc", weights, v[order]) / acc[..., None].clamp_min(1e-6)
    rgb = torch.einsum("nhw,nc->hwc", weights, _shade(s, cam[3], light)[order])
    rgb = rgb + (1-acc[..., None])*.03
    out = {"rgb": rgb, "alpha": acc, "depth": composite(z[:, None]),
           "albedo": composite(s["albedo"]), "roughness": composite(s["roughness"]),
           "metallic": composite(s["metallic"]), "normal": composite(s["normal"])}
    if return_weights:
        out["weights"] = weights[order.argsort()]
    return out


def observed_energy(raw, images, cameras, lights, size):
    # RGB only: input depth is unavailable. All variants use the same observations.
    return torch.stack([(render(raw, c, l, size)["rgb"]-im).square().mean()
                        for im, c, l in zip(images, cameras, lights)]).mean()
