import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def number(v):
    try: return float(v) if v!="" else None
    except (TypeError, ValueError): return None


def summarize(root, split="test"):
    root = Path(root)
    evaluations = []
    for p in sorted((root/"evaluations").glob("*/evaluation.json")):
        meta = json.loads(p.read_text())
        if meta["split"]==split: evaluations.append((p.parent, meta))
    if not evaluations: raise ValueError(f"No evaluations for {split}")
    hashes = {m["data_hash"] for _,m in evaluations}
    if len(hashes)!=1: raise ValueError("Do not aggregate distinct datasets")
    records = []
    for folder, meta in evaluations:
        with (folder/"results_per_object.csv").open() as f:
            for row in csv.DictReader(f):
                row["evaluation_path"] = str(folder); records.append(row)
    columns = sorted(set().union(*(r.keys() for r in records)))
    with (root/f"results_per_object_{split}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(records)
    debug = any(m["config"]["debug_only"] for _,m in evaluations)
    lines = [f"# Meeting results: {split}", "",
             "**DEBUG / PIPELINE CHECK ONLY**" if debug else "**Preliminary procedural-data pilot; not DreamLifting reproduction.**", "",
             "These are actual local run measurements. Synthetic Gaussian-family targets use the same approximate renderer as inference. No claim about real assets or a pretrained LGAA is supported.", "",
             "Means below average per-object, sample-mean metrics. No best-of-K selection is used. Missing metrics remain blank.", "",
             "| Variant | Seed | NFE | K | Objects | PSNR dB | SSIM | LPIPS | Disoccluded RGB MAE | Relit RGB MAE | Pipeline ms median |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if x is None else f"{x:.4f}"
    for _,m in evaluations:
        v = m["means"]
        lines.append(f"| {m['variant']} | {m['training_seed']} | {m['requested_nfe']} | {m['sample_count']} | {len(m['object_ids'])} | "+
                     " | ".join(fmt(v.get(k)) for k in ["psnr_db", "ssim", "lpips", "disoccluded_rgb_mae", "heldout_light_rgb_mae", "latency_median_ms"])+" |")
    lines += ["", "## Paired B2 differences", "",
              "95% percentile bootstrap across object identities, within each training seed and matched NFE/K. This is not a substitute for training-seed replication. Negative error differences favor B2.", "",
              "| Control | Seed | NFE | Metric | Mean B2 − control | 95% interval | Paired objects |",
              "|---|---:|---:|---|---:|---|---:|"]
    grouped = {}
    for r in records:
        key = (r["variant"], r["training_seed"], r["requested_nfe"], r["samples"])
        if r["object_id"] in grouped.setdefault(key, {}): raise ValueError("Duplicate evaluations; remove superseded K/seed runs from this root")
        grouped[key][r["object_id"]] = r
    pairs = []
    rng = np.random.default_rng(731)
    for (variant, seed, nfe, k), prop in grouped.items():
        if variant!="B2": continue
        for control in ["B0", "B1", "B3", "B0_anchor", "B3_anchor", "A2"]:
            base = grouped.get((control, seed, nfe, k))
            if base is None:
                lines.append(f"| {control} | {seed} | {nfe} | pending | — | — | 0 |")
                continue
            ids = sorted(set(prop)&set(base))
            for metric in ["disoccluded_rgb_mae", "visible_psnr_db", "heldout_light_rgb_mae", "depth_mae", "latency_median_ms"]:
                diffs = np.array([number(prop[i].get(metric))-number(base[i].get(metric)) for i in ids
                                  if number(prop[i].get(metric)) is not None and number(base[i].get(metric)) is not None])
                if len(diffs)<2: continue
                boot = diffs[rng.integers(0, len(diffs), size=(2000, len(diffs)))].mean(1)
                lo, hi = np.quantile(boot, [.025, .975])
                lines.append(f"| {control} | {seed} | {nfe} | {metric} | {diffs.mean():.5f} | [{lo:.5f}, {hi:.5f}] | {len(diffs)} |")
                pairs.append({"control": control, "seed": seed, "nfe": nfe, "samples": k,
                              "metric": metric, "difference": float(diffs.mean()), "ci_low": float(lo), "ci_high": float(hi), "objects": len(diffs)})
    if pairs:
        with (root/f"paired_differences_{split}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    trained = []
    lines += ["", "## Training and pending work", "",
              "| Run | Updates | Parameters allocated | Parameters with gradient | Wall hours | GPU hours | Peak VRAM GB | Backward-render views | Status |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for p in sorted(root.glob("*/manifest.json")):
        m = json.loads(p.read_text()); trained.append(m)
        lines.append(f"| {m['variant']} seed {m['seed']} | {m['step']} | {m['total_parameters']} | {m['parameters_with_gradient_last_step']} | {m['training_wall_seconds']/3600:.5f} | {fmt(m['training_gpu_hours'])} | {fmt(m['peak_vram_bytes']/1e9 if m['peak_vram_bytes'] is not None else None)} | {m['render_backward_views']} | {m['status']} |")
    variants = {m["variant"] for _,m in evaluations}
    diagnostics = []
    for m in trained:
        if m["variant"] not in {"B2", "B0_anchor", "B3_anchor"}: continue
        with (root/f"{m['variant']}_seed{m['seed']}"/"training.csv").open() as f:
            history = list(csv.DictReader(f))
        fraction = np.mean([float(r["anchor"])>0 for r in history])
        diagnostics.append(f"- {m['variant']} seed {m['seed']}: anchor penalty positive on {fraction:.1%} of logged updates. If zero throughout, this run did not exercise an active anchor; inspect tolerance/path length on validation before interpreting it.")
    lines += ["", "Anchor activity diagnostics:", ""]+diagnostics
    missing = sorted(set(["A0", "A1", "A2", "B0", "B1", "B2", "B3", "B0_anchor", "B3_anchor"])-variants)
    lines += ["", f"Pending evaluated variants: {', '.join(missing) if missing else 'none at this split/NFE inventory; check matched rows above' }.",
              f"Training seeds represented: {sorted({m['training_seed'] for _,m in evaluations})}.",
              "", "## Interpretation limits", "",
              "- Smoke configurations validate code execution only. A short loss curve or synthetic overfit does not establish the hypothesis.",
              "- B2 anchors learned rollouts. A penalty on the fixed paired bridge alone would have zero parameter gradient; that earlier formulation was corrected here.",
              "- B0_anchor and B3_anchor test whether the rendering penalty helps competing paths equally. Keep them pending explicitly if unfinished.",
              "- B3 reuses A1's checkpoint and uses a coarse-start DDIM sampler. It is a SDEdit-style warm-start control, not a faithful SDEdit implementation.",
              "- Pixelwise ensemble intervals describe empirical coverage on this dataset, not a proven multimodal posterior.",
              "- The original disoccluded-LPIPS screening margin is NOT tested by disoccluded RGB MAE. Add a validated region perceptual protocol before claiming it is met.",
              "- Test-set tables never choose hyperparameters. Use the validation split for sigma/anchor-weight selection first.",
              "- A publication claim needs nonprocedural data, the project's real decoder/renderer, geometry surface metrics and multiple training seeds.", ""]
    (root/f"MEETING_RESULTS_{split}.md").write_text("\n".join(lines))
    # Quality/cost curve; aggregate training seeds only for visualization, not inference.
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in sorted(variants):
        values = [(m["requested_nfe"], m["means"]["latency_median_ms"], m["means"]["psnr_db"]) for _,m in evaluations if m["variant"]==variant]
        by = {}
        for n, t, q in values: by.setdefault(n, []).append((t, q))
        pts = [(n, *np.mean(v, axis=0)) for n,v in sorted(by.items())]
        ax.plot([v[1] for v in pts], [v[2] for v in pts], "o-", label=variant)
        for n,t,q in pts: ax.annotate(str(n), (t, q), fontsize=7)
    ax.set(xlabel="Pipeline median latency (ms; one render)", ylabel="Sample-mean PSNR (dB)",
           title="DEBUG only" if debug else "Procedural held-out quality / cost")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(root/f"quality_latency_{split}.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in trained:
        p = root/f"{m['variant']}_seed{m['seed']}"/"training.csv"
        with p.open() as f: rows = list(csv.DictReader(f))
        ax.plot([int(r["step"]) for r in rows], [float(r["loss"]) for r in rows], label=f"{m['variant']} s{m['seed']}")
    ax.set(xlabel="Optimizer updates", ylabel="Own training loss (not comparable across objectives)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(root/"training_curves.png", dpi=160); plt.close(fig)
    print(root/f"MEETING_RESULTS_{split}.md")
