"""Presentation figures for Experiment 0.
  python -m gsm.exp0_figures --split test      (writes outputs/exp0/figures_<split>/*.png)
Colors: fixed categorical slots (reference palette, light mode); one entity -> one color everywhere.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .exp0_analyze import load, pca_fit, flat, train_bottleneck, probe_rows, pairing

OUT = "outputs/exp0"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"           # slots 1-3 (validated all-pairs)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KIND_C = {"separate": C1, "shared": C2, "shared_private": C3}
PAIR_C = {"true": C1, "same_shape": C2, "any": C3}
PAIR_L = {"true": "true pairs", "same_shape": "same-shape shuffle", "any": "any-object shuffle"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, d, name):
    fig.savefig(f"{d}/{name}.png"); plt.close(fig); print("wrote", f"{d}/{name}.png")


def img(ax, a, title):
    ax.imshow(np.clip(a, 0, 1), interpolation="nearest"); ax.set_title(title, fontsize=8, color=INK); ax.axis("off")


def pick_objects(D):
    """first test shape: its 3 part-regime draws + 1 partition draw (fixed, not selected on results)"""
    o = D["raw"]["objects"]; s0 = o[0][0]
    part = [i for i, x in enumerate(o) if x[0] == s0 and x[2] == "part"][:3]
    prt = [i for i, x in enumerate(o) if x[0] == s0 and x[2] == "partition"][:1]
    return part + prt


def fig_examples(D, d):
    f, p = D["raw"]["full128"], D["raw"]["preds"]
    ids = pick_objects(D); v = 0
    cols = ["input RGB", "GT normal (cam)", "Marigold normal (cam)", "GT albedo", "IID albedo", "GT roughness", "IID roughness", "GT material id"]
    fig, axs = plt.subplots(len(ids), len(cols), figsize=(1.45 * len(cols), 1.55 * len(ids)))
    for r, i in enumerate(ids):
        m = f["mask"][i, v].float()
        from .scene import cam_tensors, INPUT_CAMS
        R = cam_tensors(INPUT_CAMS)[0][v, :3, :3]
        gn = F.normalize(torch.einsum("ij,hwj->hwi", R, f["normal_world"][i, v].float()), dim=-1)
        gn = gn * torch.tensor([1., -1., -1.]) * 0.5 + 0.5          # OpenCV cam -> OpenGL cam (Marigold's convention)
        pn = (p["normals"][i, v, 0].float() * 0.5 + 0.5) * m
        ga = (f["albedo"][i, v].float() / m.clamp(min=1e-3)).clamp(0, 1) ** (1 / 2.2) * m
        pa = p["iid"][i, v, 0].float() * m
        gr = (f["rough"][i, v].float() / m.clamp(min=1e-3)).expand(-1, -1, 3) * m
        pr = p["iid"][i, v, 1][..., :1].float().expand(-1, -1, 3) * m
        lab = D["raw"]["targets"]["matid"][i, v, ..., 0]
        cm = plt.get_cmap("tab10")(np.clip(lab.numpy(), 0, 9).astype(int))[..., :3] * (lab.numpy()[..., None] >= 0)
        for c, a in enumerate([f["rgb"][i, v].float(), gn * m, pn, ga, pa, gr, pr, cm]):
            img(axs[r, c], a.numpy() if torch.is_tensor(a) else a, cols[c] if r == 0 else "")
        o = D["raw"]["objects"][i]
        axs[r, 0].text(-8, 64, f"shape {o[0]}\n{o[2]}\nmat {o[1]}", fontsize=7, color=INK2, ha="right", va="center")
    fig.suptitle("Same shape, independent material draws (3 part-aligned + 1 partition). GT vs frozen Marigold outputs",
                 fontsize=10, color=INK)
    save(fig, d, "fig1_examples_gt_vs_extractors")


def fig_feature_pca(D, TR, d, layer="up2"):
    ids = pick_objects(D); v = 0
    fig, axs = plt.subplots(2, len(ids), figsize=(1.9 * len(ids), 4.0))
    for r, ext in enumerate(["normals", "iid"]):
        X, _ = flat(TR["feats"][ext][layer], TR["fg"])
        mu, V = pca_fit(X[torch.randperm(len(X))[:40000]], 3)
        for c, i in enumerate(ids):
            P = (D["feats"][ext][layer][i, v] - mu) @ V                                  # (G,G,3)
            lo, hi = P.reshape(-1, 3).quantile(0.02, 0), P.reshape(-1, 3).quantile(0.98, 0)
            P = ((P - lo) / (hi - lo)).clamp(0, 1) * D["fg"][i, v][..., None]
            o = D["raw"]["objects"][i]
            img(axs[r, c], P.numpy(), f"{o[2]}, mat {o[1]}" if r == 0 else "")
            if c == 0:
                axs[r, c].text(-1.5, 8, "geometry\n(normals)" if r == 0 else "PBR\n(IID)", fontsize=9, color=INK, ha="right", va="center")
    fig.suptitle(f"Top-3 PCA of frozen features ({layer}, PCA fit on train), 16×16 patches, same shape and view.\n"
                 "Row 1 changes with material = appearance leaking into geometry features", fontsize=9, color=INK)
    save(fig, d, "fig2_feature_pca_patches")


def fig_variance(R, d):
    v = R["material_variance_fraction"]
    names = ["normals/up0", "normals/up1", "normals/up2", "depth/up2", "iid/up0", "iid/up1", "iid/up2", "gt_geo", "gt_mat"]
    vals = [v[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3))
    cols = [C1 if n.startswith(("normals", "depth")) else C2 if n.startswith("iid") else INK2 for n in names]
    ax.bar(range(len(names)), vals, color=cols, width=0.6)
    for i, x in enumerate(vals):
        ax.text(i, x + 0.01, f"{x:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("material-driven variance fraction"); ax.set_ylim(0, 1)
    ax.set_title("Fixed shape/view/patch, vary only the material draw. Blue: geometry extractors, orange: PBR, gray: GT",
                 fontsize=9, color=INK)
    save(fig, d, "fig3_material_variance_fraction")


def fig_cka(R, d):
    for layer in ["up1", "up2"]:
        t = R["cka"][layer]
        pairs = ["normals~depth", "normals~iid", "depth~iid", "normals~gt_geo", "iid~gt_geo", "iid~gt_mat"]
        fig, ax = plt.subplots(figsize=(8, 3.2)); w = 0.26
        for j, mode in enumerate(["true", "same_shape", "any"]):
            y = [t[f"{p}|{mode}"]["cka"] for p in pairs]
            lo = [y[i] - t[f"{p}|{mode}"]["ci95"][0] for i, p in enumerate(pairs)]
            hi = [t[f"{p}|{mode}"]["ci95"][1] - y[i] for i, p in enumerate(pairs)]
            ax.bar(np.arange(len(pairs)) + (j - 1) * w, y, w - 0.03, color=PAIR_C[mode], label=PAIR_L[mode],
                   yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], error_kw=dict(ecolor=INK2, lw=0.8))
        ax.set_xticks(range(len(pairs))); ax.set_xticklabels(pairs, rotation=15)
        ax.set_ylabel("linear CKA (no adapter)"); ax.set_ylim(0, 1); ax.legend(ncol=3, fontsize=8, loc="upper right")
        ax.set_title(f"Natural alignment of raw frozen features, layer {layer} (95% object-bootstrap CI)", fontsize=9, color=INK)
        save(fig, d, f"fig4_cka_{layer}")


def fig_retrieval(R, d):
    layers = ["up0", "up1", "up2"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, key, title in [(axs[0], "r1_all", "among all eval objects"), (axs[1], "r1_same_shape", "among same-shape draws")]:
        w = 0.26
        for j, (name, lab, col) in enumerate([("raw_normals_to_iid", "geometry→PBR, raw (natural)", C1),
                                              ("adapter_normals_to_iid", "geometry→PBR, ridge adapter (learned)", C2),
                                              ("raw_normals_to_depth", "geometry→geometry, raw (reference)", C3)]):
            y = [R["retrieval"][l][name][key] for l in layers]
            ax.bar(np.arange(3) + (j - 1) * w, y, w - 0.03, color=col, label=lab)
        ch = R["retrieval"]["up1"]["raw_normals_to_iid"]["chance_all" if key == "r1_all" else "chance_same_shape"]
        ax.axhline(ch, color=INK2, lw=1, ls="--"); ax.text(2.45, ch + 0.02, "chance", fontsize=7, color=INK2, ha="right")
        ax.set_xticks(range(3)); ax.set_xticklabels(layers); ax.set_title(f"R@1 {title}", fontsize=9, color=INK)
    axs[0].set_ylabel("retrieval R@1"); axs[0].set_ylim(0, 1.05); axs[1].legend(fontsize=7, loc="lower right")
    save(fig, d, "fig5_retrieval")


def fig_probes(R, d):
    fig, ax = plt.subplots(figsize=(6.5, 3.2)); w = 0.26
    probes = [("G->mat_of_M_object", "geometry feats → material of paired object"),
              ("M->geo_of_G_object", "PBR feats → geometry of paired object")]
    for j, mode in enumerate(["true", "same_shape", "any"]):
        y = [R["probes"][mode][p]["all"] for p, _ in probes]
        ax.bar(np.arange(2) + (j - 1) * w, y, w - 0.03, color=PAIR_C[mode], label=PAIR_L[mode])
        for i, v in enumerate(y):
            ax.text(i + (j - 1) * w, max(v, 0) + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(2)); ax.set_xticklabels([l for _, l in probes], fontsize=8)
    ax.set_ylabel("held-out R² (ridge)"); ax.legend(fontsize=8)
    ax.set_title("Cross-modal probes: correspondence vs shuffled-pair controls", fontsize=9, color=INK)
    save(fig, d, "fig6_cross_probes")


def fig_bottleneck(R, d):
    bt = pd.read_csv(R["bottleneck_csv"]); bt["zeroed"] = bt["zeroed"].fillna("")
    mp = R["mean_predictor"]
    a = bt[(bt.regime == "all") & (bt.zeroed == "")]
    mets = [("normal_deg", "normal error (deg)"), ("albedo_mae", "albedo MAE"), ("rough_mae", "roughness MAE"), ("metal_mae", "metallic MAE")]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6))
    for row, src in enumerate(["marigold", "gt_oracle"]):
        s = a[(a.input == src) & (a.pairing == "true")]
        for c, (m, lab) in enumerate(mets):
            ax = axs[row, c]
            for kind in ["separate", "shared", "shared_private"]:
                g = s[s.kind == kind].groupby("k")[m].agg(["mean", "std"])
                ax.errorbar(g.index, g["mean"], yerr=g["std"], color=KIND_C[kind], lw=2, marker="o", ms=5,
                            label=kind.replace("_", "+"), capsize=2)
            if src == "marigold":
                ax.axhline(mp[m], color=INK2, ls="--", lw=1); ax.text(32, mp[m], " mean predictor", fontsize=7, color=INK2, va="bottom", ha="right")
            ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels([4, 8, 16, 32])
            ax.set_title(f"{'Marigold features' if src == 'marigold' else 'GT maps (oracle)'}: {lab}", fontsize=8, color=INK)
            if row == 1: ax.set_xlabel("total latent dims per patch (k)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Matched-capacity bottlenecks: separate (k/2+k/2) vs shared (k) vs shared+private (k/2+k/4+k/4); mean ± sd over 3 seeds",
                 fontsize=10, color=INK)
    fig.tight_layout(); save(fig, d, "fig7_bottleneck_tradeoff")
    # gap vs pairing
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (cols, lab) in zip(axs, [(["normal_deg", "depth_mae"], "geometry"), (["albedo_mae", "rough_mae"], "material (albedo+rough)")]):
        for mode in ["true", "same_shape", "any"]:
            s = a[(a.input == "marigold") & (a.pairing == mode)].groupby(["k", "kind"])[cols].mean().reset_index()
            ref = s[(s.kind == "separate") & (s.k == 32)].iloc[0]
            comp = lambda df: sum(df[c] / ref[c] for c in cols) / len(cols)
            sh = s[s.kind == "shared"].set_index("k"); se = s[s.kind == "separate"].set_index("k")
            gap = (comp(sh) / comp(se) - 1) * 100
            ax.plot(gap.index, gap.values, marker="o", lw=2, color=PAIR_C[mode], label=PAIR_L[mode])
        ax.axhline(0, color=INK2, lw=0.8); ax.axhline(5, color=INK2, lw=0.8, ls=":"); ax.text(32, 5.3, "+5% rule", fontsize=7, color=INK2, ha="right")
        ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels([4, 8, 16, 32])
        ax.set_title(f"shared vs separate, {lab}", fontsize=9, color=INK); ax.set_xlabel("k")
    axs[0].set_ylabel("shared error relative to separate (%)\n(< 0 = shared better)"); axs[1].legend(fontsize=8)
    save(fig, d, "fig8_shared_gap_vs_pairing")


def fig_recon(D, TR, d, layer="up2", k=8, pca=256, dev="cuda", split=""):
    """Retrain separate/shared k=8 (seed 0, same recipe as the analysis) and show decoded 16x16 maps."""
    gtr, _ = flat(TR["feats"]["normals"][layer], TR["fg"]); mtr, _ = flat(TR["feats"]["iid"][layer], TR["fg"])
    sub = torch.randperm(len(gtr))[:60000]
    pg = pca_fit(gtr[sub], pca); pm = pca_fit(mtr[sub], pca)
    def P(Dx):
        G = (Dx["feats"]["normals"][layer] - pg[0]) @ pg[1]; M = (Dx["feats"]["iid"][layer] - pm[0]) @ pm[1]
        return G, M
    TG, TM = P(TR); EG, EM = P(D)
    a = probe_rows(TR, pairing(TR, "true"), TG, TM)
    tr = (a[0].float(), a[1].float(), a[2].float(), a[3].float())
    models = {kind: train_bottleneck(kind, k, tr, 0, steps=3000, dev=dev).eval() for kind in ["separate", "shared"]}
    ids = pick_objects(D); v = 0
    cols = ["GT normal", "separate", "shared", "GT albedo", "separate", "shared", "GT rough", "separate", "shared"]
    fig, axs = plt.subplots(len(ids), len(cols), figsize=(1.35 * len(cols), 1.45 * len(ids)))
    for r, i in enumerate(ids):
        fg = D["fg"][i, v]
        out = {}
        with torch.no_grad():
            for kind, mdl in models.items():
                pgx, pmx = mdl(EG[i, v].reshape(-1, EG.shape[-1]).float().to(dev), EM[i, v].reshape(-1, EM.shape[-1]).float().to(dev))
                out[kind] = (pgx.cpu().reshape(16, 16, -1), pmx.cpu().reshape(16, 16, -1))
        m = fg[..., None].float()
        nmap = lambda x: (F.normalize(x[..., :3], dim=-1) * 0.5 + 0.5) * m
        amap = lambda x: x[..., :3].clamp(0, 1) ** (1 / 2.2) * m
        rmap = lambda x: x[..., 3:4].clamp(0, 1).expand(-1, -1, 3) * m
        gt_g, gt_m = D["geo"][i, v], D["mat"][i, v]
        ims = [nmap(gt_g), nmap(out["separate"][0]), nmap(out["shared"][0]),
               amap(gt_m), amap(out["separate"][1]), amap(out["shared"][1]),
               rmap(gt_m), rmap(out["separate"][1]), rmap(out["shared"][1])]
        for c, a_ in enumerate(ims):
            img(axs[r, c], a_.numpy(), cols[c] if r == 0 else "")
    fig.suptitle(f"Decoded patch maps from Marigold features, k={k} (camera-frame normals), first {split} shape, view 0. Same seed-0 recipe as the analysis",
                 fontsize=9, color=INK)
    save(fig, d, "fig9_bottleneck_reconstructions")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); a = ap.parse_args()
    d = f"{OUT}/figures_{a.split}"; os.makedirs(d, exist_ok=True)
    R = json.load(open(f"{OUT}/exp0_{a.split}.json"))
    fig_variance(R, d); fig_cka(R, d); fig_retrieval(R, d); fig_probes(R, d); fig_bottleneck(R, d)
    TR = load("train"); D = load(a.split)
    fig_examples(D, d); fig_feature_pca(D, TR, d); fig_recon(D, TR, d, split=a.split)


if __name__ == "__main__":
    main()
