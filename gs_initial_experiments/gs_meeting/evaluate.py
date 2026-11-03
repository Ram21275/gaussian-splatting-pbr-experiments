import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from .data import Dataset, conditions, fingerprint, write_json
from .engine import device_for, load_model, validate_coarse, coarse_prediction, seed_all, sync, file_hash
from .methods import sample, VARIANTS
from .scene import decode, render, camera


def masked_mean(value, mask):
    if value.ndim == 3: mask = mask[..., None].expand_as(value)
    return float(value[mask].mean()) if mask.any() else None


def ssim_map(x, y):
    # Local SSIM in linear RGB, data range 1. Valid for small >=7 px pilot images.
    x = x.permute(2, 0, 1)[None]; y = y.permute(2, 0, 1)[None]
    pool = lambda z: F.avg_pool2d(z, 7, 1, 3)
    mx, my = pool(x), pool(y)
    vx = (pool(x*x)-mx*mx).clamp_min(0); vy = (pool(y*y)-my*my).clamp_min(0)
    xy = pool(x*y)-mx*my
    v = ((2*mx*my+.01**2)*(2*xy+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return v[0].mean(0)


def interval_metrics(draws, target, mask, name):
    result = {}
    if draws.shape[0]<2 or not mask.any(): return result
    for level in [.5, .8, .95]:
        low = torch.quantile(draws, (1-level)/2, dim=0)
        high = torch.quantile(draws, (1+level)/2, dim=0)
        covered = ((target>=low)&(target<=high)).float()
        result[f"{name}_coverage_{int(level*100)}"] = masked_mean(covered, mask)
        result[f"{name}_width_{int(level*100)}"] = masked_mean(high-low, mask)
    # Ensemble CRPS; finite-sample estimator, not an analytic posterior score.
    crps = (draws-target).abs().mean(0)-.5*(draws[:, None]-draws[None, :]).abs().mean((0, 1))
    result[f"{name}_crps"] = masked_mean(crps, mask)
    return result


def _mean(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def panel(path, images, captions, size):
    tile = max(size, 128)
    canvas = Image.new("RGB", (tile*len(images), tile+24), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, captions)):
        a = im.detach().cpu().numpy()
        if a.ndim==2: a = np.repeat(a[..., None], 3, -1)
        if a.shape[-1]==1: a = np.repeat(a, 3, -1)
        image = Image.fromarray((np.clip(a, 0, 1)*255).astype("uint8")).resize((tile, tile))
        canvas.paste(image, (i*tile, 24))
        draw.text((i*tile+3, 3), label, fill="black")
    canvas.save(path)


def evaluate(config, checkpoint, split="test", nfe=8, samples=8, device="auto",
             coarse_path=None, variant_override=None, lpips=False, overwrite=False):
    if samples<1: raise ValueError("samples must be positive")
    device = device_for(device); torch.set_num_threads(config.get("cpu_threads", 2))
    model, ck = load_model(checkpoint, device)
    variant = variant_override or ck["variant"]
    # Only B3 is a legitimate inference-only switch from an A1 checkpoint.
    if variant != ck["variant"] and not (variant=="B3" and ck["variant"]=="A1"):
        raise ValueError("Only --variant B3 from A1 is allowed")
    if variant not in VARIANTS: raise ValueError(variant)
    for k in config:
        if k not in {"steps", "checkpoint_every", "output_root", "eval_seed", "latency_repeats"} and config[k]!=ck["config"].get(k):
            raise ValueError(f"Evaluation config mismatch: {k}")
    ds = Dataset(config["data_root"], split, device)
    dh = fingerprint(config["data_root"])
    if ck["data_hash"]!=dh: raise ValueError("Evaluation data changed since training")
    coarse_model = validate_coarse(coarse_path, config, dh, device) if variant.startswith("B") else None
    if ck["variant"].startswith("B") and ck["coarse_hash"]!=file_hash(coarse_path):
        raise ValueError("Evaluation coarse checkpoint differs from training")
    cfg = ck["config"].copy()
    cfg["eval_seed"] = config["eval_seed"]; cfg["latency_repeats"] = config["latency_repeats"]
    out = Path(config["output_root"])/"evaluations"/f"{variant}_seed{ck['seed']}_{split}_nfe{nfe}_k{samples}"
    if out.exists() and not overwrite: raise FileExistsError(f"{out} exists; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    perceptual = None
    if lpips:
        try:
            import lpips as lp
            perceptual = lp.LPIPS(net="alex").to(device).eval()
        except (ImportError, OSError, RuntimeError) as e:
            raise RuntimeError("LPIPS requested: install compatible lpips/torchvision and cache its weights first") from e
    rows = []; qualities = []; sample_nfe = 1 if variant=="A0" else nfe
    start_eval = time.perf_counter()
    with torch.no_grad():
        for index in range(len(ds)):
            d = ds[index]; size = cfg["size"]
            context = conditions(d)[None]
            coarse = coarse_prediction(coarse_model, context, cfg["anchors"]) if coarse_model else None
            seed_all(cfg["eval_seed"]+index*1000)
            raw_draws = []
            for k in range(samples):
                z, counted = sample(model, variant, context, (1, cfg["anchors"], 13), cfg, nfe, coarse)
                assert counted==sample_nfe
                raw_draws.append(ds.raw(z[0]))
            raw_draws = torch.stack(raw_draws)
            if not torch.isfinite(raw_draws).all(): raise FloatingPointError("Nonfinite sampled state")
            metrics = {}
            def record(key, value): metrics.setdefault(key, []).append(value)
            gt_state = decode(d["state"])
            for raw in raw_draws:
                attr = decode(raw)
                record("anchor_position_rmse", float((attr["position"]-gt_state["position"]).square().mean().sqrt()))
                record("anchor_normal_cosine_error", float((1-(attr["normal"]*gt_state["normal"]).sum(-1)).mean()))
                for k in ["albedo", "roughness", "metallic"]:
                    record(f"anchor_{k}_mae", float((attr[k]-gt_state[k]).abs().mean()))
            for li, light in enumerate(d["target_lights"]):
                for vi, cam in enumerate(d["target_cameras"]):
                    draws = [render(raw, cam, light, size) for raw in raw_draws]
                    gt = {k: d[f"target_{k}"][li, vi] for k in ["rgb", "alpha", "depth", "albedo", "roughness", "metallic"]}
                    fg = gt["alpha"]>.1
                    visible = fg & (d["visible_fraction"][vi]>=.5)
                    disoccluded = fg & (d["visible_fraction"][vi]<.5)
                    record("disoccluded_pixel_fraction", float(disoccluded.float().mean()))
                    for pred in draws:
                        mse = float((pred["rgb"]-gt["rgb"]).square().mean())
                        record("psnr_db", -10*math.log10(max(mse, 1e-12)))
                        record("ssim", float(ssim_map(pred["rgb"], gt["rgb"]).mean()))
                        fg_mse = masked_mean((pred["rgb"]-gt["rgb"]).square(), fg)
                        record("foreground_psnr_db", -10*math.log10(max(fg_mse, 1e-12)) if fg_mse is not None else None)
                        # Light 0 matches input lighting; light 1 was absent from the inputs.
                        record("heldout_light_rgb_mae" if li>0 else "input_light_rgb_mae", float((pred["rgb"]-gt["rgb"]).abs().mean()))
                        for name, mask in [("visible", visible), ("disoccluded", disoccluded)]:
                            err = masked_mean((pred["rgb"]-gt["rgb"]).square(), mask)
                            record(f"{name}_psnr_db", -10*math.log10(max(err, 1e-12)) if err is not None else None)
                            record(f"{name}_rgb_mae", masked_mean((pred["rgb"]-gt["rgb"]).abs(), mask))
                        for key in ["depth", "albedo", "roughness", "metallic"]:
                            record(f"{key}_mae", masked_mean((pred[key]-gt[key]).abs(), fg))
                        record("lpips", float(perceptual(pred["rgb"].permute(2, 0, 1)[None]*2-1,
                                                           gt["rgb"].permute(2, 0, 1)[None]*2-1).item()) if perceptual else None)
                    for key in ["depth", "albedo", "roughness", "metallic"]:
                        for k, v in interval_metrics(torch.stack([r[key] for r in draws]), gt[key], fg, key).items(): record(k, v)
                    if index<6 and li==0 and vi==0:
                        p = draws[0]
                        panel(out/f"{d['id']}_first_sample.png", [gt["rgb"], p["rgb"], p["depth"]/4,
                              p["albedo"], p["roughness"], p["metallic"]],
                              ["GT", "sample 0", "depth/4", "albedo", "rough", "metal"], size)
            # Timings include conditioning, frozen A0 when needed, sampling, decode,
            # and one render. Disk IO and evaluation/GT work are explicitly excluded.
            def infer_once():
                c = conditions(d)[None]
                x0 = coarse_prediction(coarse_model, c, cfg["anchors"]) if coarse_model else None
                z, _ = sample(model, variant, c, (1, cfg["anchors"], 13), cfg, nfe, x0)
                return render(ds.raw(z[0]), d["target_cameras"][0], d["target_lights"][1], size)
            for _ in range(2): infer_once()
            timings = []
            for _ in range(cfg["latency_repeats"]):
                sync(device); t0 = time.perf_counter(); infer_once(); sync(device)
                timings.append((time.perf_counter()-t0)*1000)
            row = {"object_id": d["id"], "variant": variant, "training_seed": ck["seed"],
                   "split": split, "requested_nfe": nfe, "generator_nfe": sample_nfe,
                   "pipeline_nfe": sample_nfe+int(coarse_model is not None), "samples": samples,
                   "latency_median_ms": float(np.median(timings)), "latency_p95_ms": float(np.percentile(timings, 95)),
                   **{k: _mean(v) for k,v in metrics.items()}}
            rows.append(row)
            # Save predictions, not ground truth masquerading as a generated asset.
            np.savez_compressed(out/f"{d['id']}_predicted_states.npz", raw=raw_draws.cpu().numpy())
            if index==0:
                frames = []
                for a in np.linspace(0, 2*math.pi, 16, endpoint=False):
                    c = camera(float(a)).to(device)
                    im = render(raw_draws[0], c, d["target_lights"][1], size)["rgb"]
                    frames.append(Image.fromarray((im.cpu().numpy().clip(0, 1)*255).astype("uint8")))
                frames[0].save(out/"first_sample_relighting_turntable.gif", save_all=True,
                               append_images=frames[1:], duration=100, loop=0)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (out/"results_per_object.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    means = {k: _mean([r[k] for r in rows]) for k in fields if k not in {"object_id", "variant", "split"}}
    write_json(out/"evaluation.json", {"status": "debug_only" if cfg["debug_only"] else "pilot_measurements",
               "variant": variant, "checkpoint": str(checkpoint), "checkpoint_hash": file_hash(checkpoint),
               "training_step": ck["step"], "training_seed": ck["seed"],
               "data_hash": dh, "split": split, "object_ids": ds.ids,
               "config": cfg, "requested_nfe": nfe, "sample_count": samples,
               "coarse_checkpoint": str(coarse_path) if coarse_path else None,
               "coarse_hash": file_hash(coarse_path) if coarse_path else None,
               "lpips_available": perceptual is not None,
               "seconds": time.perf_counter()-start_eval, "means": means,
               "metric_scope": "linear RGB; aligned-anchor errors, not mesh surface metrics; empirical pixel ensemble intervals",
               "timing_scope": "pooled input features + optional coarse A0 + generator + decoder + one reference render; excludes disk IO"})
    print(f"Evaluated {variant}: {out}")
    return out
