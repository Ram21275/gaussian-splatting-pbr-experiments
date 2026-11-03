"""Identity-disjoint procedural data and explicit external NPZ contract."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .scene import anchors, camera, render, CHANNELS


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def fingerprint(root):
    """Hash all object bytes, not just filenames or the split manifest."""
    h = hashlib.sha256()
    for p in sorted(Path(root).glob("*.npz")) + [Path(root)/"splits.json", Path(root)/"stats.npz"]:
        h.update(p.name.encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def make_data(cfg):
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    root = Path(cfg["data_root"])
    if (root/"splits.json").exists(): raise FileExistsError("Data exists; use another data_root.")
    root.mkdir(parents=True, exist_ok=True)
    n, size = cfg["anchors"], cfg["size"]
    a = anchors(n)
    # Inputs lie on one arc to leave a meaningful disoccluded back side.
    cams = torch.stack([camera(x) for x in [-.65, 0., .65]])
    ev = torch.stack([camera(2*math.pi*(i+.5)/cfg["target_views"], .35)
                      for i in range(cfg["target_views"])])
    lights = F.normalize(torch.tensor([[.3, 1., 1.], [-1., .5, -.2]]), dim=-1)
    splits = {}; train_states = []; index = 0
    generator = torch.Generator().manual_seed(cfg["data_seed"])
    for split, count in cfg["split_counts"].items():
        splits[split] = []
        for _ in range(count):
            oid = f"object_{index:04d}"; index += 1; splits[split].append(oid)
            # Low-frequency asymmetric family. Each object has unique geometry/materials.
            basis = torch.cat([torch.ones(n, 1), a, a.square(),
                               torch.sin(a*3), torch.cos(a*4)], -1)
            coeff = torch.randn(basis.shape[1], len(CHANNELS), generator=generator)*.28
            raw = basis @ coeff
            raw[:, 3] = .3 + .25*raw[:, 3]
            raw[:, 4:7] *= .25
            raw[:, 11] -= 1.5
            raw[:, 12] += 1.5
            with torch.no_grad():
                inputs = torch.stack([render(raw, c, lights[0], size)["rgb"] for c in cams])
                targets = [[render(raw, c, l, size) for c in ev] for l in lights]
                fields = {f"target_{k}": torch.stack([torch.stack([x[k] for x in row]) for row in targets]).numpy()
                          for k in ["rgb", "alpha", "depth", "albedo", "roughness", "metallic"]}
                # Input-visible primitive definition independent of all predictions.
                iw = torch.stack([render(raw, c, lights[0], size, True)["weights"] for c in cams])
                seen = (iw.amax((0, 2, 3))>.04).float()
                vis = []
                for c in ev:
                    r = render(raw, c, lights[0], size, True)
                    vis.append((r["weights"]*seen[:, None, None]).sum(0)/r["alpha"].clamp_min(1e-6))
            np.savez_compressed(root/f"{oid}.npz", state=raw.numpy(),
                                input_rgb=inputs.numpy(), input_cameras=cams.numpy(),
                                input_lights=lights[0].expand(3, -1).numpy(),
                                target_cameras=ev.numpy(), target_lights=lights.numpy(),
                                visible_fraction=torch.stack(vis).numpy(), **fields)
            if split == "train": train_states.append(raw.numpy())
    train = np.concatenate(train_states)
    np.savez(root/"stats.npz", mean=train.mean(0), std=np.maximum(train.std(0), .1))
    write_json(root/"splits.json", splits)
    write_json(root/"dataset.json", {"kind": "procedural_isotropic_gaussian_pilot",
               "claims": "synthetic feasibility only", "channels": CHANNELS,
               "configuration": cfg, "fingerprint": fingerprint(root)})


class Dataset:
    def __init__(self, root, split, device="cpu"):
        self.root = Path(root); self.device = device
        self.splits = json.loads((self.root/"splits.json").read_text())
        flat = sum(self.splits.values(), [])
        if len(flat)!=len(set(flat)): raise ValueError("Object identity split leakage")
        self.ids = self.splits[split]
        if not self.ids: raise ValueError(f"Empty split {split}")
        with np.load(self.root/"stats.npz", allow_pickle=False) as z:
            self.mean = torch.tensor(z["mean"], device=device)
            self.std = torch.tensor(z["std"], device=device)
        if self.mean.shape != (13,) or self.std.shape != (13,) or not (self.std>0).all():
            raise ValueError("Expected train-only 13-channel normalization")
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        with np.load(self.root/f"{self.ids[i]}.npz", allow_pickle=False) as z:
            d = {k: torch.tensor(z[k], device=self.device, dtype=torch.float32) for k in z.files}
        d["id"] = self.ids[i]
        d["z"] = (d["state"]-self.mean)/self.std
        return d
    def raw(self, z): return z*self.std+self.mean


def conditions(d):
    """Frozen deterministic RGB pool + camera/light metadata. No target access."""
    rgb = d["input_rgb"].permute(0, 3, 1, 2)
    pixels = F.adaptive_avg_pool2d(rgb, (8, 8)).flatten(1)
    return torch.cat([pixels, d["input_cameras"].flatten(1)/3., d["input_lights"]], -1)


def audit_data(cfg):
    """Fail closed on leakage/shapes; flag empty completion regions before training."""
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    result = {"data_hash": fingerprint(cfg["data_root"]), "splits": {}, "warnings": []}
    for split in ["train", "val", "test"]:
        ds = Dataset(cfg["data_root"], split)
        fractions = []; minimum = 1.
        for i in range(len(ds)):
            d = ds[i]; n, size = cfg["anchors"], cfg["size"]
            if d["state"].shape != (n, 13): raise ValueError(f"{d['id']}: state shape")
            if d["input_rgb"].shape[1:] != (size, size, 3): raise ValueError(f"{d['id']}: image shape")
            if not (0<=d["input_rgb"].min()<=d["input_rgb"].max()<=1): raise ValueError("RGB range")
            for key, value in d.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all(): raise ValueError(f"Nonfinite {key}")
            for key in ["input_cameras", "target_cameras"]:
                axes = d[key][:, :3]
                if not torch.allclose(axes@axes.transpose(-1, -2), torch.eye(3).expand(len(axes), -1, -1), atol=1e-4):
                    raise ValueError(f"{d['id']}: nonorthonormal cameras")
            if d["visible_fraction"].min()<0 or d["visible_fraction"].max()>1.00001: raise ValueError("Visibility range")
            fg = d["target_alpha"][0]>.1
            if not fg.any(): raise ValueError(f"{d['id']}: no foreground")
            disocc = fg & (d["visible_fraction"]<.5)
            frac = float(disocc.sum()/fg.sum()); fractions.append(frac); minimum = min(minimum, frac)
        result["splits"][split] = {"objects": len(ds), "mean_disoccluded_foreground_fraction": float(np.mean(fractions)),
                                   "min_disoccluded_foreground_fraction": minimum}
        if minimum==0: result["warnings"].append(f"{split}: some objects have no disoccluded foreground; inspect masks before completion claims")
    if cfg["debug_only"]: result["warnings"].append("Debug configuration; code-path checks only")
    write_json(Path(cfg["data_root"])/"audit.json", result)
    return result
