import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .data import Dataset, conditions, fingerprint, write_json
from .model import Core
from .methods import VARIANTS, ANCHORED, objective, rollout_for_anchor
from .scene import observed_energy


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sync(device):
    if str(device).startswith("cuda"): torch.cuda.synchronize()


def device_for(requested):
    if requested == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return requested


def hardware():
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "git_commit": commit, "precision": "float32", "renderer": "dense_reference_GGX_isotropic_one_sided_no_shadows"}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def load_model(path, device):
    # Checkpoints include optimizer/RNG data. Only load this package's TRUSTED local checkpoints.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Core(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    return model, checkpoint


def coarse_prediction(model, context, count):
    with torch.no_grad():
        return model(torch.zeros(1, count, 13, device=context.device),
                     torch.zeros(1, device=context.device), context)[0]


def validate_coarse(path, cfg, data_hash, device):
    model, ckpt = load_model(path, device)
    if ckpt["variant"] != "A0": raise ValueError("Coarse checkpoint must be A0")
    if ckpt["data_hash"] != data_hash: raise ValueError("Coarse dataset fingerprint mismatch")
    if ckpt["config"]["anchors"] != cfg["anchors"]: raise ValueError("Anchor mismatch")
    for p in model.parameters(): p.requires_grad_(False)
    return model


def train(cfg, variant, seed, device="auto", coarse_path=None, resume=False, max_hours=None):
    if variant not in VARIANTS: raise ValueError(variant)
    device = device_for(device)
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    seed_all(seed)
    ds = Dataset(cfg["data_root"], "train", device)
    data_hash = fingerprint(cfg["data_root"])
    needs_coarse = variant.startswith("B")
    if needs_coarse and not coarse_path: raise ValueError("B variants require --coarse <A0 checkpoint>")
    coarse_model = validate_coarse(coarse_path, cfg, data_hash, device) if needs_coarse else None
    model = Core(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    run = Path(cfg["output_root"])/f"{variant}_seed{seed}"
    checkpoint_path = run/"checkpoint.pt"
    if run.exists() and not resume: raise FileExistsError(f"{run} exists; --resume or a fresh output_root")
    run.mkdir(parents=True, exist_ok=True)
    start = 0; elapsed_prior = 0.; render_backward_views = 0
    if resume:
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        old = {k:v for k,v in ck["config"].items() if k not in {"steps", "checkpoint_every"}}
        new = {k:v for k,v in cfg.items() if k not in {"steps", "checkpoint_every"}}
        if old != new or ck["variant"]!=variant or ck["seed"]!=seed or ck["data_hash"]!=data_hash:
            raise ValueError("Resume config/variant/seed/data mismatch")
        if ck["coarse_hash"] != (file_hash(coarse_path) if coarse_path else None):
            raise ValueError("Resume requires the same frozen coarse checkpoint bytes")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        random.setstate(ck["rng_python"]); np.random.set_state(ck["rng_numpy"])
        torch.set_rng_state(ck["rng_torch"].cpu())
        if device.startswith("cuda") and ck["rng_cuda"] is not None: torch.cuda.set_rng_state_all(ck["rng_cuda"])
        start = ck["step"]; elapsed_prior = ck["elapsed_seconds"]
        render_backward_views = ck["render_backward_views"]
    if start >= cfg["steps"]: return checkpoint_path
    if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    log_path = run/"training.csv"
    # Discard log rows after the restored checkpoint; those updates will be replayed.
    if resume and log_path.exists():
        with log_path.open() as f: rows = list(csv.DictReader(f))
        rows = [r for r in rows if int(r["step"])<=start]
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "loss", "main", "score", "rgb", "anchor", "grad_norm", "seconds"])
            w.writeheader(); w.writerows(rows)
    log = log_path.open("a" if resume and log_path.exists() else "w", newline="")
    writer = csv.writer(log)
    if log.tell()==0: writer.writerow(["step", "loss", "main", "score", "rgb", "anchor", "grad_norm", "seconds"])
    model.train(); sync(device); begin = time.perf_counter(); last_step = start
    coarse_hash = file_hash(coarse_path) if coarse_path else None
    def save(step):
        sync(device)
        elapsed = elapsed_prior+time.perf_counter()-begin
        ck = {"format": 1, "config": cfg, "variant": variant, "seed": seed,
              "step": step, "data_hash": data_hash, "coarse_hash": coarse_hash,
              "model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
              "rng_torch": torch.get_rng_state(),
              "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
              "elapsed_seconds": elapsed, "render_backward_views": render_backward_views}
        tmp = checkpoint_path.with_suffix(".tmp"); torch.save(ck, tmp); os.replace(tmp, checkpoint_path)
        manifest = {"status": "completed" if step==cfg["steps"] else "budget_paused",
                    "scope": "DreamLifting-inspired synthetic feasibility prototype",
                    "config": cfg, "variant": variant, "seed": seed, "step": step,
                    "data_hash": data_hash, "train_object_ids": ds.ids,
                    "splits": ds.splits, "coarse_checkpoint": str(coarse_path), "coarse_hash": coarse_hash,
                    "hardware": hardware(), "total_parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "parameters_with_gradient_last_step": sum(p.numel() for p in model.parameters() if p.grad is not None),
                    "training_wall_seconds": elapsed,
                    "training_gpu_hours": elapsed/3600 if device.startswith("cuda") else None,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
                    "render_backward_views": render_backward_views,
                    "checkpoint": str(checkpoint_path)}
        write_json(run/"manifest.json", manifest)
    try:
        for step in range(start+1, cfg["steps"]+1):
            optimizer.zero_grad(set_to_none=True)
            totals = dict.fromkeys(["main", "score", "rgb", "anchor"], 0.)
            for _ in range(cfg["accumulation"]):
                d = ds[np.random.randint(len(ds))]
                if tuple(d["state"].shape) != (cfg["anchors"], 13): raise ValueError("State shape does not match config")
                context = conditions(d)[None]
                coarse = coarse_prediction(coarse_model, context, cfg["anchors"]) if coarse_model else None
                pieces, clean = objective(model, variant, d["z"][None], context, coarse, cfg)
                rgb = clean.new_zeros(()); penalty = clean.new_zeros(())
                if cfg["rgb_weight"]>0:
                    rgb = observed_energy(ds.raw(clean[0]), d["input_rgb"], d["input_cameras"], d["input_lights"], cfg["size"])
                    render_backward_views += len(d["input_rgb"])
                # Compute the SAME rollout and render work for all B comparisons.
                # Zero-weight controls still build/backpropagate this graph.
                if needs_coarse and step % cfg["anchor_every"] == 0:
                    end_time = float(torch.rand((), device=device)*.9+.1)
                    predicted = rollout_for_anchor(model, variant, coarse, context, cfg, end_time)
                    ep = observed_energy(ds.raw(predicted[0]), d["input_rgb"], d["input_cameras"], d["input_lights"], cfg["size"])
                    with torch.no_grad():
                        e0 = observed_energy(ds.raw(coarse[0]), d["input_rgb"], d["input_cameras"], d["input_lights"], cfg["size"])
                    penalty = torch.relu(ep-e0-cfg["anchor_tolerance"])
                    render_backward_views += len(d["input_rgb"])
                aw = cfg["anchor_weight"] if variant in ANCHORED else 0.
                loss = pieces["main"]+cfg["score_weight"]*pieces["score"]+cfg["rgb_weight"]*rgb+aw*penalty
                if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss at step {step}")
                (loss/cfg["accumulation"]).backward()
                for k, v in {**pieces, "rgb": rgb, "anchor": penalty}.items(): totals[k] += float(v.detach())/cfg["accumulation"]
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step(); last_step = step
            weighted = totals["main"]+cfg["score_weight"]*totals["score"]+cfg["rgb_weight"]*totals["rgb"]+(cfg["anchor_weight"] if variant in ANCHORED else 0)*totals["anchor"]
            writer.writerow([step, weighted, totals["main"], totals["score"], totals["rgb"], totals["anchor"], float(grad), elapsed_prior+time.perf_counter()-begin]); log.flush()
            if step % cfg["checkpoint_every"]==0 or step==cfg["steps"]: save(step)
            if max_hours is not None and time.perf_counter()-begin>=max_hours*3600:
                save(step); break
    except KeyboardInterrupt:
        if last_step>start: save(last_step)
        raise
    finally: log.close()
    print(f"{variant}: checkpoint {checkpoint_path}")
    return checkpoint_path
