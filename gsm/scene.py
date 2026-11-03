"""Procedural PBR objects, cameras, lights, analytic SDF ray-marcher and GGX shading.

Conventions (shared by GT renderer and Gaussian renderer):
  world: z-up, objects centred at origin, bounding radius ~0.9.
  camera: OpenCV (x right, y down, z forward); w2c = [R|t]; pixel (u,v) centres at +0.5.
  depth: camera-space z. Background: black (radiance 0), mask = 0.
  shading: local Cook-Torrance GGX + Lambert, directional lights + ambient, NO cast shadows,
           linear radiance -> clamp[0,1] -> sRGB gamma 2.2 for display/metrics.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

RES = 128
FOV_DEG = 40.0
CAM_DIST = 3.0

# ---------------------------------------------------------------- cameras

def look_at(eye, target=(0, 0, 0), up=(0, 0, 1)):
    eye = np.asarray(eye, np.float64); target = np.asarray(target, np.float64); up = np.asarray(up, np.float64)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)  # camera y (down)
    R = np.stack([r, d, f], 0)  # rows: cam axes in world
    t = -R @ eye
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
    return w2c


def cam_from_angles(az_deg, el_deg, dist=CAM_DIST):
    az, el = math.radians(az_deg), math.radians(el_deg)
    eye = dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return look_at(eye)


def intrinsics(res=RES, fov=FOV_DEG):
    f = 0.5 * res / math.tan(math.radians(fov) / 2)
    return np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]], np.float64)


# fixed camera sets (identical for every object)
INPUT_CAMS = [(15.0, 20.0), (135.0, 20.0), (255.0, 20.0)]          # 3 posed input views
INPUT5_CAMS = INPUT_CAMS + [(75.0, 45.0), (195.0, -5.0)]           # 5-view robustness set (future)
TARGET_CAMS = [(60.0, 10.0), (100.0, 40.0), (180.0, 5.0), (220.0, 35.0),
               (300.0, 15.0), (340.0, 45.0), (30.0, -10.0), (160.0, 60.0)]  # 8 held-out cameras
ALL_CAMS = INPUT_CAMS + TARGET_CAMS  # index 0..2 inputs, 3..10 targets


def cam_tensors(cams, device="cpu", res=RES):
    w2c = torch.tensor(np.stack([cam_from_angles(a, e) for a, e in cams]), dtype=torch.float32, device=device)
    K = torch.tensor(intrinsics(res), dtype=torch.float32, device=device)
    return w2c, K


def pixel_rays(w2c, K, res=RES):
    """w2c (V,4,4) -> origins (V,3), dirs (V,H*W,3) world, unit; dirs_cam (H*W,3) with z=1."""
    dev = w2c.device
    ys, xs = torch.meshgrid(torch.arange(res, device=dev), torch.arange(res, device=dev), indexing="ij")
    uv = torch.stack([xs + 0.5, ys + 0.5], -1).reshape(-1, 2).float()
    dc = torch.cat([(uv[:, :1] - K[0, 2]) / K[0, 0], (uv[:, 1:] - K[1, 2]) / K[1, 1], torch.ones_like(uv[:, :1])], -1)
    R = w2c[:, :3, :3]; t = w2c[:, :3, 3]
    o = -(R.transpose(1, 2) @ t[..., None])[..., 0]
    d = torch.einsum("vji,pj->vpi", R, dc)  # R^T dc
    return o, F.normalize(d, dim=-1), dc


# ---------------------------------------------------------------- lights

def _dir(az, el):
    az, el = math.radians(az), math.radians(el)
    return [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]

# each light set: list of (direction-to-light, rgb irradiance) + ambient rgb
LIGHTS = {
    "L0_train": dict(dirs=[_dir(30, 50), _dir(200, 20)], cols=[[3.0, 3.0, 3.0], [0.8, 0.8, 0.8]], amb=[0.15, 0.15, 0.15]),
    "L1_heldout_warm": dict(dirs=[_dir(150, 10)], cols=[[3.6, 2.7, 1.8]], amb=[0.10, 0.09, 0.08]),
    "L2_heldout_topcool": dict(dirs=[_dir(0, 85), _dir(270, -15)], cols=[[1.8, 2.4, 3.2], [1.2, 1.2, 1.2]], amb=[0.08, 0.08, 0.10]),
}
LIGHT_NAMES = list(LIGHTS.keys())


def shade(albedo, rough, metal, normal, viewdir, light):
    """GGX/Lambert local shading. albedo (...,3) linear, rough/metal (...,1), normal unit (...,3),
    viewdir = unit vector from surface to camera (...,3). Returns linear radiance (...,3)."""
    dev = albedo.device
    n = F.normalize(normal, dim=-1)
    v = viewdir
    nv = (n * v).sum(-1, keepdim=True).clamp(min=1e-4)
    r = rough.clamp(0.05, 1.0)
    a2 = (r * r) ** 2
    k = (r + 1) ** 2 / 8
    F0 = 0.04 * (1 - metal) + albedo * metal
    out = torch.zeros_like(albedo)
    for ldir, col in zip(light["dirs"], light["cols"]):
        l = torch.tensor(ldir, device=dev, dtype=albedo.dtype)
        c = torch.tensor(col, device=dev, dtype=albedo.dtype)
        h = F.normalize(v + l, dim=-1)
        nl = (n * l).sum(-1, keepdim=True).clamp(min=0)
        nh = (n * h).sum(-1, keepdim=True).clamp(min=0)
        vh = (v * h).sum(-1, keepdim=True).clamp(min=0)
        D = a2 / (math.pi * (nh * nh * (a2 - 1) + 1) ** 2)
        G = (nv / (nv * (1 - k) + k)) * (nl / (nl * (1 - k) + k))
        Fr = F0 + (1 - F0) * (1 - vh) ** 5
        spec = D * G * Fr / (4 * nv * nl.clamp(min=1e-4))
        diff = (1 - Fr) * (1 - metal) * albedo / math.pi
        out = out + (diff + spec) * c * nl
    amb = torch.tensor(light["amb"], device=dev, dtype=albedo.dtype)
    out = out + amb * (albedo * (1 - metal) + F0 * metal)
    return out


def to_srgb(x):
    return x.clamp(0, 1).clamp(min=1e-6) ** (1 / 2.2)


# ---------------------------------------------------------------- procedural objects

PRIMS = ["box", "ellipsoid", "cylinder", "torus", "sphere"]


def _rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rand_material(rng):
    metallic = float(rng.random() < 0.3)
    hue = rng.random(); sat = rng.uniform(0.0, 0.7)
    if metallic:
        val = rng.uniform(0.55, 0.95); sat *= 0.5
    else:
        val = rng.uniform(0.08, 0.85)
    import colorsys
    rgb = np.array(colorsys.hsv_to_rgb(hue, sat, val)) ** 2.2  # perceptual -> linear albedo
    rgb = np.clip(rgb, 0.03, 0.9)
    return dict(albedo=rgb.tolist(), roughness=float(rng.uniform(0.15, 0.9)), metallic=metallic)


def make_object(shape_id, material_seed=None, regime="part"):
    """Factorial procedural object. Geometry depends ONLY on shape_id; materials ONLY on
    (material_seed, regime). regime="part": one material per primitive part (part-aligned).
    regime="partition": 4 materials assigned by 2 random planes through the object, cutting
    across parts (material layout independent of part structure)."""
    rng = np.random.default_rng(10_000 + shape_id)
    parts = []
    btype = rng.choice(["box", "ellipsoid", "sphere", "cylinder"])
    parts.append(dict(type=str(btype), c=[0.0, 0.0, 0.0], R=_rand_rot(rng).tolist(),
                      s=rng.uniform(0.3, 0.6, size=3).tolist()))
    for _ in range(int(rng.integers(2, 5))):
        t = str(rng.choice(PRIMS))
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        c = (d * rng.uniform(0.3, 0.6)).tolist()
        s = rng.uniform(0.1, 0.35, size=3).tolist()
        parts.append(dict(type=t, c=c, R=_rand_rot(rng).tolist(), s=s))
    if material_seed is None:
        material_seed = shape_id
    mrng = np.random.default_rng(900_000 + 7919 * material_seed + (0 if regime == "part" else 1))
    if regime == "part":
        mats = [_rand_material(mrng) for _ in parts]
        planes = []
    else:
        mats = [_rand_material(mrng) for _ in range(4)]
        planes = []
        for _ in range(2):
            n = mrng.normal(size=3); n /= np.linalg.norm(n)
            planes.append(n.tolist() + [float(mrng.uniform(-0.15, 0.15))])
    obj = dict(id=int(shape_id), shape_id=int(shape_id), material_seed=int(material_seed), regime=regime,
               parts=parts, mats=mats, planes=planes, scale=1.0)
    sdf = ObjectSDF(obj, "cpu")
    g = torch.linspace(-1.6, 1.6, 64)
    P = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    inside = P[sdf.sdf(P) < 0]
    obj["scale"] = float(0.9 / inside.norm(dim=-1).max())
    return obj


class ObjectSDF:
    def __init__(self, obj, device):
        self.dev = device
        self.scale = obj["scale"]
        self.parts = []
        for p in obj["parts"]:
            self.parts.append(dict(type=p["type"], c=torch.tensor(p["c"], dtype=torch.float32, device=device),
                                   R=torch.tensor(p["R"], dtype=torch.float32, device=device),
                                   s=torch.tensor(p["s"], dtype=torch.float32, device=device)))
        mats = obj["mats"]
        self.albedo = torch.tensor([m["albedo"] for m in mats], dtype=torch.float32, device=device)
        self.rough = torch.tensor([[m["roughness"]] for m in mats], dtype=torch.float32, device=device)
        self.metal = torch.tensor([[m["metallic"]] for m in mats], dtype=torch.float32, device=device)
        self.regime = obj.get("regime", "part")
        self.planes = torch.tensor(obj.get("planes") or [[0, 0, 1, 0]], dtype=torch.float32, device=device)

    @staticmethod
    def _prim(t, q, s):
        if t == "box":
            r = 0.03
            d = q.abs() - (s - r)
            return d.clamp(min=0).norm(dim=-1) + d.max(-1).values.clamp(max=0) - r
        if t == "sphere":
            return q.norm(dim=-1) - s[0]
        if t == "ellipsoid":  # conservative approx (scaled by min axis ratio)
            k0 = (q / s).norm(dim=-1)
            return (k0 - 1) * s.min()
        if t == "cylinder":  # axis z, radius s0, half-height s2
            d = torch.stack([q[..., :2].norm(dim=-1) - s[0], q[..., 2].abs() - s[2]], -1)
            return d.clamp(min=0).norm(dim=-1) + d.max(-1).values.clamp(max=0)
        if t == "torus":  # major s0+0.1, minor 0.35*s0+0.04 in xy-plane
            R_, r_ = s[0] + 0.1, 0.35 * s[0] + 0.04
            qx = q[..., :2].norm(dim=-1) - R_
            return torch.stack([qx, q[..., 2]], -1).norm(dim=-1) - r_
        raise ValueError(t)

    def part_sdf(self, P):
        Q = P / self.scale
        ds = []
        for p in self.parts:
            q = (Q - p["c"]) @ p["R"]  # into local frame (R columns = local axes)
            ds.append(self._prim(p["type"], q, p["s"]))
        return torch.stack(ds, -1) * self.scale

    def sdf(self, P):
        return self.part_sdf(P).min(-1).values

    def normal(self, P, eps=1e-3):
        e = torch.eye(3, device=P.device) * eps
        g = torch.stack([self.sdf(P + e[i]) - self.sdf(P - e[i]) for i in range(3)], -1)
        return F.normalize(g, dim=-1)

    def material_index(self, P):
        if self.regime == "part":
            return self.part_sdf(P).argmin(-1)
        side = ((P / self.scale) @ self.planes[:, :3].T - self.planes[:, 3] > 0).long()
        return side[:, 0] + 2 * side[:, 1]

    def material(self, P):
        idx = self.material_index(P)
        return self.albedo[idx], self.rough[idx], self.metal[idx]

    def raymarch(self, o, d, tmin=1.5, tmax=4.6, steps=192):
        """o (...,3), d (...,3) unit. Returns hit mask, t."""
        t = torch.full(d.shape[:-1], tmin, device=d.device)
        o = o.expand_as(d)
        for _ in range(steps):
            dist = self.sdf(o + t[..., None] * d)
            t = (t + 0.9 * dist).clamp(max=tmax + 1.0)  # misses stop at tmax+1 (no inf/NaN)
        P = o + t[..., None] * d
        hit = (self.sdf(P).abs() < 2e-3) & (t < tmax)
        return hit, t

    def sample_surface(self, n, seed=0):
        """Approximately uniform surface samples via near-surface rejection + projection."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        pts = []
        tot = 0
        while tot < n:
            P = (torch.rand(200_000, 3, generator=g) * 2 - 1).to(self.dev)
            d = self.sdf(P)
            P = P[d.abs() < 0.01]
            for _ in range(4):
                P = P - self.sdf(P)[:, None] * self.normal(P)
            pts.append(P); tot += len(P)
        return torch.cat(pts)[:n]


@torch.no_grad()
def render_gt(obj, cams, device, res=RES, lights=None):
    """Ray-marched GT maps for cameras list. Returns dict of (V,H,W,C) float tensors."""
    sdf = ObjectSDF(obj, device)
    w2c, K = cam_tensors(cams, device, res)
    o, d, dc = pixel_rays(w2c, K, res)
    V, P = d.shape[:2]
    hit, t = sdf.raymarch(o[:, None, :], d)
    X = o[:, None, :] + t[..., None] * d
    n = sdf.normal(X.reshape(-1, 3)).reshape(V, P, 3)
    alb, ro, me = sdf.material(X.reshape(-1, 3))
    part = sdf.part_sdf(X.reshape(-1, 3)).argmin(-1).reshape(V, P, 1).float()
    matid = sdf.material_index(X.reshape(-1, 3)).reshape(V, P, 1).float()
    alb, ro, me = alb.reshape(V, P, 3), ro.reshape(V, P, 1), me.reshape(V, P, 1)
    m = hit[..., None].float()
    depth = torch.einsum("vpi,vi->vp", X - o[:, None, :], w2c[:, 2, :3])[..., None]  # camera z
    out = dict(mask=m, depth=depth * m, normal=n * m, albedo=alb * m, rough=ro * m, metal=me * m,
               part=(part + 1) * m - 1, matid=(matid + 1) * m - 1)  # -1 = background
    for ln in (lights or LIGHT_NAMES):
        rad = shade(alb, ro, me, n, -d, LIGHTS[ln])
        out["rgb_" + ln] = to_srgb(rad) * m
    out = {k: torch.where(hit[..., None] | (k in ("part", "matid")), v, torch.zeros_like(v)) for k, v in out.items()}
    return {k: v.reshape(V, res, res, -1) for k, v in out.items()}
