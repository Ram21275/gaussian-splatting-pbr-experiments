"""Data preparation: procedural objects -> GT maps, aligned target latents (teacher fit), frozen
DINOv2 input-view features, surface samples. One .pt file per split under data/pbr_pilot/.

  python -m gsm.prep --split train   (ids 0..63)
  python -m gsm.prep --split val     (ids 1000..1015)
  python -m gsm.prep --split test    (ids 2000..2031)
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
from .scene import make_object, render_gt, ALL_CAMS, INPUT_CAMS, cam_tensors, LIGHT_NAMES, ObjectSDF
from .latent import init_raw_latent, decode_raw, LOGIT_CLAMP
from .splat import decode_render

SPLITS = {"train": list(range(0, 64)), "val": list(range(1000, 1016)), "test": list(range(2000, 2032)),
          "debug": [0, 1]}
MAPS = ["mask", "depth", "normal", "albedo", "rough", "metal"] + ["rgb_" + l for l in LIGHT_NAMES]


def render_loss(pred, gt, w_rgb=1.0):
    m = gt["mask"]
    l = w_rgb * (pred["rgb_L0_train"] - gt["rgb_L0_train"]).abs().mean()
    l = l + (pred["albedo"] - gt["albedo"]).abs().mean() + (pred["rough"] - gt["rough"]).abs().mean() \
        + (pred["metal"] - gt["metal"]).abs().mean()
    l = l + 0.5 * (pred["depth"] - gt["depth"]).abs().mean() + (pred["acc"] - m).abs().mean()
    l = l + 0.2 * ((1 - (pred["normal"] * gt["normal"]).sum(-1, keepdim=True)) * m).mean()
    return l


def teacher_fit(raw0, gts, w2c, K, steps=400, views=4, lr=3e-3, tie=10.0, gen=None):
    """Batched per-object refinement of all channels against full-view GT (training-time teacher).
    raw0 (B,N,C); gts dict of (B,V,H,W,c)."""
    raw = raw0.clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    B, V = w2c.shape[:2]
    for it in range(steps):
        vi = torch.randperm(V, generator=gen)[:views].to(w2c.device)
        pred = decode_render(decode_raw(raw), w2c[:, vi], K)
        gt = {k: gts[k][:, vi] for k in MAPS}
        loss = render_loss(pred, gt) + tie * ((raw[..., :3] - raw0[..., :3]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            raw[..., 7:].clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)
    return raw.detach(), loss.item()


@torch.no_grad()
def dino_features(imgs, model):
    """imgs (M,3,H,W) in [0,1] -> patch tokens (M,256,384) + cls (M,1,384), 224px input."""
    x = F.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    out = model.forward_features((x - mean) / std)
    return torch.cat([out["x_norm_clstoken"][:, None], out["x_norm_patchtokens"]], 1)


def load_dino(device):
    m = torch.hub.load(os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"), "dinov2_vits14",
                       source="local", verbose=False)
    return m.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="data/pbr_pilot")
    ap.add_argument("--fit_steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    ids = SPLITS[args.split]
    torch.manual_seed(0)
    w2c, K = cam_tensors(ALL_CAMS, dev)
    dino = load_dino(dev)
    rec = dict(ids=ids, objs=[], gt={k: [] for k in MAPS}, raw_init=[], raw=[], hit=[], feats=[], surf=[], fit_loss=[])
    t0 = time.time(); t_fit = 0.0
    for s in range(0, len(ids), args.bs):
        chunk = ids[s:s + args.bs]
        objs = [make_object(i) for i in chunk]
        gts = [render_gt(o, ALL_CAMS, dev) for o in objs]
        G = {k: torch.stack([g[k] for g in gts]) for k in MAPS}
        r0 = [init_raw_latent(o, dev) for o in objs]
        raw0 = torch.stack([r[0] for r in r0])
        tf = time.time()
        raw, fl = teacher_fit(raw0, G, w2c[None].expand(len(chunk), -1, -1, -1), K, steps=args.fit_steps)
        torch.cuda.synchronize(); t_fit += time.time() - tf
        inp = G["rgb_L0_train"][:, :3].reshape(-1, 128, 128, 3).permute(0, 3, 1, 2)
        feats = dino_features(inp, dino).reshape(len(chunk), 3, 257, 384)
        for j, o in enumerate(objs):
            rec["objs"].append(o)
            rec["surf"].append(ObjectSDF(o, dev).sample_surface(4096, seed=o["id"]).cpu())
        for k in MAPS:
            rec["gt"][k].append(G[k].half().cpu())
        rec["raw_init"].append(raw0.cpu()); rec["raw"].append(raw.cpu())
        rec["hit"].append(torch.stack([r[1] for r in r0]).cpu()); rec["feats"].append(feats.half().cpu())
        rec["fit_loss"].append(fl)
        print(f"[{args.split}] {s + len(chunk)}/{len(ids)} fit_loss={fl:.4f} elapsed={time.time() - t0:.0f}s", flush=True)
    for k in MAPS:
        rec["gt"][k] = torch.cat(rec["gt"][k])
    for k in ["raw_init", "raw", "hit", "feats"]:
        rec[k] = torch.cat(rec[k])
    rec["surf"] = torch.stack(rec["surf"])
    rec["prep_seconds"] = time.time() - t0
    rec["teacher_fit_seconds"] = t_fit
    rec["gpu"] = torch.cuda.get_device_name(0)
    torch.save(rec, f"{args.out}/{args.split}.pt")
    print(json.dumps(dict(split=args.split, n=len(ids), prep_s=rec["prep_seconds"], fit_s=t_fit)))


if __name__ == "__main__":
    main()
