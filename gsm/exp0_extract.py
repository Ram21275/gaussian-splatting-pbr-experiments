"""Experiment 0 step 1: factorial procedural renders + frozen Marigold feature extraction.

For every object (shape_id x material_seed x regime) we render 3 posed input views (512 px, light
L0_train, black background) and run three frozen extractors on the SAME RGB images:
  normals  : prs-eth/marigold-normals-v1-1        (geometry branch)
  depth    : prs-eth/marigold-depth-v1-1          (geometry-geometry positive control)
  iid      : prs-eth/marigold-iid-appearance-v1-1 (PBR branch: albedo / roughness / metallicity)
UNet up-block activations (same block names in all models) are captured at the single
denoising step and average-pooled to a 16x16 patch grid. Only input RGB enters the extractors;
GT maps are stored solely as probe / reconstruction targets.

  python -m gsm.exp0_extract --split train|val|test [--limit N]
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .scene import make_object, render_gt, INPUT_CAMS

OUT = "data/exp0"
# v2 (2026-11-03): every object gets its OWN material seed (3 draws per shape), disjoint seed
# ranges per split. v1 crossed only 3 seeds over all shapes -> 3 training palettes (design flaw,
# see EXP0_PLAN.md amendment). Geometry still depends only on shape_id.
SPLITS = {  # (shape ids, material-seed base); seeds = base + 3*shape_index + j, j in {0,1,2}
    "train": (list(range(0, 80)), 10_000),
    "val": (list(range(1000, 1016)), 20_000),
    "test": (list(range(2000, 2032)), 30_000),
}
DRAWS = 3
REGIMES = ["part", "partition"]
RES = 512
GRID = 16
LAYERS = ["up0", "up1", "up2"]   # SD2 UNet up_blocks[0..2] outputs (1280@16, 1280@32, 640@64 for 512 input)
MODELS = {"normals": "prs-eth/marigold-normals-v1-1", "depth": "prs-eth/marigold-depth-v1-1",
          "iid": "prs-eth/marigold-iid-appearance-v1-1"}
STEPS = 1     # Marigold v1-1 (trailing schedule) supports 1-step inference; same for all models
SEED = 0


def objects(split):
    shapes, base = SPLITS[split]
    return [(s, base + DRAWS * i + j + (0 if r == "part" else 5000), r)
            for r in REGIMES for i, s in enumerate(shapes) for j in range(DRAWS)]


def load_pipe(name, device):
    from diffusers import MarigoldNormalsPipeline, MarigoldDepthPipeline, MarigoldIntrinsicsPipeline
    cls = {"normals": MarigoldNormalsPipeline, "depth": MarigoldDepthPipeline, "iid": MarigoldIntrinsicsPipeline}[name]
    pipe = cls.from_pretrained(MODELS[name], variant="fp16", torch_dtype=torch.float16)
    # Marigold conditions only on a constant empty-prompt embedding: compute it once, then drop the
    # text encoder to save VRAM (identical numerics to the stock pipeline).
    with torch.no_grad():
        ids = pipe.tokenizer("", padding="do_not_pad", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids
        pipe.empty_text_embedding = pipe.text_encoder.float()(ids)[0].half()
    pipe.register_modules(text_encoder=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


class Hooks:
    def __init__(self, unet):
        self.acts = {}
        self.h = [unet.up_blocks[i].register_forward_hook(self._mk(f"up{i}")) for i in range(3)]

    def _mk(self, k):
        def f(m, i, o):
            o = o[0] if isinstance(o, tuple) else o
            self.acts[k] = F.adaptive_avg_pool2d(o.float(), GRID).half().cpu()
        return f


def pool_target(x, mode="mean"):
    """(V,H,W,C) -> (V,GRID,GRID,C)"""
    t = x.permute(0, 3, 1, 2).float()
    if mode == "mean":
        return F.adaptive_avg_pool2d(t, GRID).permute(0, 2, 3, 1)
    k = t.shape[-1] // GRID  # mode for label maps via one-hot pooling
    lab = t[:, 0].long() + 1
    oh = F.one_hot(lab.clamp(0, 15), 16).permute(0, 3, 1, 2).float()
    return (F.avg_pool2d(oh, k).argmax(1) - 1).float()[..., None]


def run_model(pipe, imgs, bs, gen_seed):
    """imgs list of PIL. Returns per-layer pooled features (M,C,G,G) and prediction arrays."""
    hk = Hooks(pipe.unet)
    feats = {k: [] for k in LAYERS}; preds = []
    for s in range(0, len(imgs), bs):
        g = torch.Generator(device=pipe.device).manual_seed(gen_seed + s)
        out = pipe(imgs[s:s + bs], num_inference_steps=STEPS, ensemble_size=1, processing_resolution=RES,
                   match_input_resolution=True, generator=g, output_type="np", batch_size=bs)
        for k in LAYERS:
            feats[k].append(hk.acts[k])
        preds.append(np.asarray(out.prediction, dtype=np.float32))
    for h in hk.h:
        h.remove()
    return {k: torch.cat(v) for k, v in feats.items()}, np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=6)
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    objs = objects(args.split)[: args.limit]
    t0 = time.time()
    # ---- render
    from PIL import Image
    imgs, tgt = [], {k: [] for k in ["mask", "normal_world", "depth", "albedo", "rough", "metal", "part", "matid"]}
    full = {k: [] for k in ["mask", "normal_world", "depth", "albedo", "rough", "metal", "rgb"]}
    for (s, m, r) in objs:
        g = render_gt(make_object(s, m, r), INPUT_CAMS, dev, res=RES, lights=["L0_train"])
        for k_, v_ in g.items():
            if not torch.isfinite(v_).all():
                raise FloatingPointError(f"non-finite GT {k_} for object {(s, m, r)}")
        for v in range(3):
            imgs.append(Image.fromarray((g["rgb_L0_train"][v].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)))
        tgt["mask"].append(pool_target(g["mask"])); tgt["normal_world"].append(pool_target(g["normal"]))
        for k in ["depth", "albedo", "rough", "metal"]:
            tgt[k].append(pool_target(g[k]))
        tgt["part"].append(pool_target(g["part"], "mode")); tgt["matid"].append(pool_target(g["matid"], "mode"))
        ds = lambda x: F.interpolate(x.permute(0, 3, 1, 2).float(), size=128, mode="area").permute(0, 2, 3, 1).half().cpu()
        full["mask"].append(ds(g["mask"])); full["normal_world"].append(ds(g["normal"])); full["rgb"].append(ds(g["rgb_L0_train"]))
        for k in ["depth", "albedo", "rough", "metal"]:
            full[k].append(ds(g[k]))
    t_render = time.time() - t0
    rec = dict(objects=objs, split=args.split, res=RES, grid=GRID, steps=STEPS, seed=SEED,
               targets={k: torch.stack(v).cpu() for k, v in tgt.items()},
               full128={k: torch.stack(v) for k, v in full.items()}, feats={}, preds={}, timing={"render_s": t_render})
    # ---- extract (same RGB images into every frozen model)
    for name in MODELS:
        t1 = time.time()
        pipe = load_pipe(name, dev)
        f, p = run_model(pipe, imgs, args.bs, SEED)
        rec["feats"][name] = {k: v.reshape(len(objs), 3, *v.shape[1:]) for k, v in f.items()}
        p = torch.from_numpy(p)
        if p.ndim == 3:
            p = p[..., None]
        T = p.shape[0] // (len(objs) * 3)   # number of predicted targets per image (iid: >1)
        p = F.interpolate(p.permute(0, 3, 1, 2), size=128, mode="area").permute(0, 2, 3, 1)
        rec["preds"][name] = p.reshape(len(objs), 3, T, 128, 128, -1).half()
        rec["timing"][name + "_s"] = time.time() - t1
        if name == "iid":
            rec["iid_target_properties"] = getattr(pipe, "target_properties", None)
        del pipe; torch.cuda.empty_cache()
        print(f"[{args.split}] {name}: {time.time() - t1:.0f}s pred shape {tuple(p.shape)}", flush=True)
    path = f"{OUT}/{args.split}{'' if args.limit is None else f'_lim{args.limit}'}.pt"
    torch.save(rec, path)
    print(json.dumps(dict(path=path, n=len(objs), timing=rec["timing"])))


if __name__ == "__main__":
    main()
