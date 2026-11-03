"""Experiment 0 report: tables, plots and the locked decision rule (EXP0_PLAN.md section 6/6a).
  python -m gsm.exp0_report --split val
"""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/exp0"
GEO = ["normal_deg", "depth_mae"]; MAT = ["albedo_mae", "rough_mae"]   # v2: metallic reported separately (not identifiable here)
ALL = GEO + MAT + ["metal_mae"]


def composites(df, ref):
    g = sum(df[c] / ref[c] for c in GEO) / len(GEO)
    m = sum(df[c] / ref[c] for c in MAT) / len(MAT)
    return g, m


def decide(bt, meanp):
    d = bt[(bt.input == "marigold") & (bt.pairing == "true") & (bt.regime == "all")]
    agg = d.groupby(["k", "kind", "zeroed"])[GEO + MAT].mean().reset_index()
    ref = agg[(agg.kind == "separate") & (agg.k == agg.k.max())].iloc[0]
    agg["geo"], agg["mat"] = composites(agg, ref)
    mp = pd.DataFrame([meanp]); mpg, mpm = composites(mp, ref)
    # v2 precondition: a modality is decidable only if separate@k=32 beats the mean predictor by >=10%
    decidable = dict(geo=bool(1.0 <= 0.9 * float(mpg.iloc[0])), mat=bool(1.0 <= 0.9 * float(mpm.iloc[0])))
    sep = agg[agg.kind == "separate"].set_index("k")
    ok = sep[(sep.geo + sep.mat) / 2 <= 1.10]
    kstar = int(ok.index.min()) if len(ok) else int(agg.k.max())
    a = agg[agg.k == kstar].set_index(["kind", "zeroed"])
    s, sh, sp = a.loc[("separate", "")], a.loc[("shared", "")], a.loc[("shared_private", "")]
    spz = a.loc[("shared_private", "private")]
    sh_rel = dict(geo=sh.geo / s.geo - 1, mat=sh.mat / s.mat - 1)
    sp_rel = dict(geo=sp.geo / s.geo - 1, mat=sp.mat / s.mat - 1)
    priv_cost = dict(geo=spz.geo / sp.geo - 1, mat=spz.mat / sp.mat - 1)
    if not (decidable["geo"] and decidable["mat"]):
        choice = "not_decidable (" + ", ".join(k for k, v in decidable.items() if not v) + " at mean-predictor level)"
    elif sh_rel["geo"] <= 0.05 and sh_rel["mat"] <= 0.05:
        choice = "shared"
    elif sp_rel["geo"] <= 0.05 and sp_rel["mat"] <= 0.05 and any(
            priv_cost[mo] > 0.10 for mo in ["geo", "mat"] if sh_rel[mo] > 0.05):
        choice = "shared_private"
    else:
        choice = "separate"
    return dict(k_star=kstar, mean_predictor_composite=dict(geo=float(mpg.iloc[0]), mat=float(mpm.iloc[0])),
                decidable=decidable, shared_vs_separate=sh_rel, sharedprivate_vs_separate=sp_rel,
                private_zeroing_cost=priv_cost, choice=choice), agg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    R = json.load(open(f"{OUT}/exp0_{a.split}{a.tag}.json"))
    bt = pd.read_csv(R["bottleneck_csv"]); bt["zeroed"] = bt["zeroed"].fillna("")
    dec, agg = decide(bt, R["mean_predictor"])
    L = [f"# Experiment 0 results ({a.split} split)", "",
         "**Status:** measured on procedural factorial objects. The frozen extractors are Marigold v1-1 (normals, depth, IID-appearance) with 1-step inference. These are not real-asset results.", ""]
    L += ["## 0. Extractor sanity (vs GT, 128 px)", "", "| metric | value |", "|---|---:|"]
    L += [f"| {k} | {v:.4f} |" for k, v in R["extractor_sanity"].items()]
    if "material_variance_fraction" in R:
        L += ["", "## 1. Material-driven variance fraction (fixed shape, view and patch; vary material seed)", "",
              "0 = invariant to material; 1 = all variance is material-driven.", "", "| features | fraction |", "|---|---:|"]
        L += [f"| {k} | {v:.3f} |" for k, v in R["material_variance_fraction"].items()]
    if "cka" in R:
        L += ["", "## 2. Natural alignment: linear CKA, no adapter (object-bootstrap 95% CI)", ""]
        for layer, tab in R["cka"].items():
            L += [f"### layer {layer}", "", "| pair | true | same-shape shuffle | any shuffle |", "|---|---|---|---|"]
            pairs = sorted({k.split("|")[0] for k in tab})
            f = lambda d: f"{d['cka']:.3f} [{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]"
            L += [f"| {p} | {f(tab[p + '|true'])} | {f(tab[p + '|same_shape'])} | {f(tab[p + '|any'])} |" for p in pairs]
            L += [""]
    if "retrieval" in R:
        L += ["## 3. Object retrieval (R@1). `raw` is natural, with no fitted map; `adapter` is ridge fitted on train (learned alignment)", "",
              "| layer | query→candidates | R@1 all (chance) | R@1 same-shape (chance) |", "|---|---|---|---|"]
        for layer, t in R["retrieval"].items():
            for name, r in t.items():
                L += [f"| {layer} | {name} | {r['r1_all']:.3f} ({r['chance_all']:.3f}) | {r['r1_same_shape']:.3f} ({r['chance_same_shape']:.3f}) |"]
    if "probes" in R:
        L += ["", "## 4. Cross-prediction probes (ridge on PCA-64 patch features; held-out R²)", "",
              "| probe | pairing | all | part regime | partition regime |", "|---|---|---:|---:|---:|"]
        for mode, t in R["probes"].items():
            for name, r in t.items():
                L += [f"| {name} | {mode} | {r['all']:.3f} | {r['part']:.3f} | {r['partition']:.3f} |"]
        L += ["", "### Adjacent-patch affinity (logistic on |f_a − f_b|; AUC, 0.5 = chance)", "",
              "| probe | part regime | partition regime | positive rate |", "|---|---:|---:|---:|"]
        for k, r in R["affinity_auc"].items():
            L += [f"| {k} | {r['part']:.3f} | {r['partition']:.3f} | {r['pos_rate']:.3f} |"]
    # bottleneck tables
    L += ["", "## 5. Bottlenecks at matched latent k (mean ± sd over seeds, regime = all)", ""]
    d = bt[bt.regime == "all"]
    g = d.groupby(["input", "pairing", "k", "kind", "zeroed"])[ALL + ["params"]].agg(["mean", "std"])
    L += ["| input | pairing | k | kind | zeroed | normal° | depth | albedo | rough | metal | params |", "|---|---|---:|---|---|---|---|---|---|---|---:|"]
    for key, r in g.iterrows():
        L += ["| " + " | ".join(str(x) for x in key) + " | " +
              " | ".join(f"{r[(c, 'mean')]:.4f}±{r[(c, 'std')]:.4f}" for c in ALL) + f" | {int(r[('params', 'mean')])} |"]
    mp = R["mean_predictor"]
    L += ["| mean predictor | – | – | – | – | " + " | ".join(f"{mp[c]:.4f}" for c in ALL) + " | 0 |", ""]
    # shared - separate gap: true vs shuffled
    L += ["", "### Shared-minus-separate composite gap: true vs shuffled pairing (Marigold, regime = all)", "",
          "| pairing | k | Δgeo (shared/sep − 1) | Δmat |", "|---|---:|---:|---:|"]
    for mode in ["true", "same_shape", "any"]:
        dd = bt[(bt.input == "marigold") & (bt.pairing == mode) & (bt.regime == "all") & (bt.zeroed.fillna("") == "")]
        if not len(dd): continue
        ag = dd.groupby(["k", "kind"])[GEO + MAT].mean().reset_index()
        ref = ag[(ag.kind == "separate") & (ag.k == ag.k.max())].iloc[0]
        ag["geo"], ag["mat"] = composites(ag, ref)
        for k in sorted(ag.k.unique()):
            s = ag[(ag.k == k) & (ag.kind == "separate")].iloc[0]; h = ag[(ag.k == k) & (ag.kind == "shared")].iloc[0]
            L += [f"| {mode} | {k} | {h.geo / s.geo - 1:+.3f} | {h.mat / s.mat - 1:+.3f} |"]
    L += ["", "## 6. Locked decision rule → Experiment A representation", "", "```", json.dumps(dec, indent=1), "```"]
    open(f"{OUT}/EXP0_RESULTS_{a.split}{a.tag}.md", "w").write("\n".join(L) + "\n")
    # plot: composite error vs k
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for kind, mk in [("separate", "o-"), ("shared", "s-"), ("shared_private", "^-")]:
        s = agg[(agg.kind == kind) & (agg.zeroed.fillna("") == "")].sort_values("k")
        ax[0].plot(s.k, s.geo, mk, label=kind); ax[1].plot(s.k, s.mat, mk, label=kind)
    for i, t in enumerate(["geometry composite (÷ separate@k=32)", "material composite (÷ separate@k=32)"]):
        ax[i].set_xscale("log", base=2); ax[i].set_xlabel("total latent k"); ax[i].set_title(t); ax[i].axhline(1, c="gray", lw=.5)
    ax[0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/exp0_bottleneck_{a.split}{a.tag}.png", dpi=120)
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
