"""Experiment 0, Amendment 2 (3): Marigold-IID accuracy / convention checks.
Fixed subset: first 4 val shapes x 3 draws x 2 regimes = 24 objects (72 views). One-factor-at-a-time
from the setting used in the main analysis (black bg, 512 px, 1 step, ensemble 1).
  python -m gsm.exp0_iid_checks
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .scene import make_object, render_gt, INPUT_CAMS
from .exp0_extract import objects, load_pipe

OUT = "outputs/exp0/iid_checks"
SETTINGS = {  # name: (background, processing_res, steps, ensemble)
    "main(black,512,1step)": ("black", 512, 1, 1),
    "grey_background": ("grey", 512, 1, 1),
    "4_steps": ("black", 512, 4, 1),
    "4_steps_ens5": ("black", 512, 4, 5),
    "proc_res_768": ("black", 768, 1, 1),
}


def pear(x, y):
    return float(torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1])


def main():
    dev = "cuda"; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    shapes = sorted({o[0] for o in objects("val")})[:4]
    objs = [o for o in objects("val") if o[0] in shapes]
    assert len(objs) == 24
    G = {k: [] for k in ["rgb", "mask", "albedo", "rough", "metal"]}
    for (s, m, r) in objs:
        g = render_gt(make_object(s, m, r), INPUT_CAMS, dev, res=512, lights=["L0_train"])
        for k, src in [("rgb", "rgb_L0_train"), ("mask", "mask"), ("albedo", "albedo"), ("rough", "rough"), ("metal", "metal")]:
            G[k].append(g[src].cpu())
    G = {k: torch.cat(v) for k, v in G.items()}                      # (72,512,512,c)
    m = G["mask"][..., 0] > 0.5
    ga = G["albedo"]; gr = G["rough"][..., 0]; gm = G["metal"][..., 0]
    pipe = load_pipe("iid", dev)
    res = {"objects": objs, "settings": {}}
    for name, (bg, pres, steps, ens) in SETTINGS.items():
        rgb = G["rgb"] if bg == "black" else G["rgb"] + 0.5 * (1 - G["mask"])
        imgs = [Image.fromarray((x.clamp(0, 1).numpy() * 255).astype(np.uint8)) for x in rgb]
        preds = []
        for s in range(0, len(imgs), 4):
            out = pipe(imgs[s:s + 4], num_inference_steps=steps, ensemble_size=ens, processing_resolution=pres,
                       match_input_resolution=True, generator=torch.Generator(device=dev).manual_seed(s),
                       output_type="np", batch_size=4)
            preds.append(torch.from_numpy(np.asarray(out.prediction, dtype=np.float32)))
        p = torch.cat(preds).reshape(len(imgs), 2, 512, 512, 3)           # target 0 albedo, 1 material stack
        alb, mat = p[:, 0], p[:, 1]
        r = {}
        for cs, a in [("as_sRGB->linear", alb.clamp(0, 1) ** 2.2), ("as_linear", alb)]:
            r[f"albedo_mae[{cs}]"] = float((a - ga).abs().mean(-1)[m].mean())
            r[f"albedo_pearson[{cs}]"] = pear(a[m], ga[m])
            gains = [((a[i][m[i]] * ga[i][m[i]]).sum() / (a[i][m[i]] ** 2).sum()) for i in range(len(a))]
            r[f"albedo_mae_gain_aligned[{cs}]"] = float(torch.cat([(gains[i] * a[i][m[i]] - ga[i][m[i]]).abs().mean(-1)
                                                                  for i in range(len(a))]).mean())
        r["material_channel_vs_GT_pearson"] = {f"ch{c}": {"rough": pear(mat[..., c][m], gr[m]), "metal": pear(mat[..., c][m], gm[m])}
                                               for c in range(3)}
        r["roughness_mae(ch0)"] = float((mat[..., 0] - gr).abs()[m].mean())
        r["metallic_mae(ch1)"] = float((mat[..., 1] - gm).abs()[m].mean())
        r["metal_ch1_mean_on_metal_vs_dielectric"] = [float(mat[..., 1][m & (gm > .5)].mean()), float(mat[..., 1][m & (gm < .5)].mean())]
        r["seconds"] = time.time() - t0
        res["settings"][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}), flush=True)
    res["mean_baselines"] = dict(albedo_mae=float((ga[m] - ga[m].mean(0)).abs().mean()), rough_mae=float((gr[m] - gr[m].mean()).abs().mean()),
                                 metal_mae=float((gm[m] - gm[m].mean()).abs().mean()), metal_fraction=float((gm[m] > .5).float().mean()))
    json.dump(res, open(f"{OUT}/iid_checks.json", "w"), indent=1)
    L = ["# Marigold-IID accuracy/convention checks (24 val objects, 72 views; Amendment 2, part 3)", "",
         f"Mean-predictor baselines: {json.dumps({k: round(v, 4) for k, v in res['mean_baselines'].items()})}", "",
         "| setting | albedo MAE sRGB→lin / as-lin | albedo r | gain-aligned albedo MAE | rough MAE (ch0) | metal MAE (ch1) | ch0~rough r | ch1~metal r | ch1 on metal / dielectric |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["settings"].items():
        cm = r["material_channel_vs_GT_pearson"]
        L += [f"| {n} | {r['albedo_mae[as_sRGB->linear]']:.3f} / {r['albedo_mae[as_linear]']:.3f} | {r['albedo_pearson[as_sRGB->linear]']:.3f} | "
              f"{r['albedo_mae_gain_aligned[as_sRGB->linear]']:.3f} | {r['roughness_mae(ch0)']:.3f} | {r['metallic_mae(ch1)']:.3f} | "
              f"{cm['ch0']['rough']:.3f} | {cm['ch1']['metal']:.3f} | {r['metal_ch1_mean_on_metal_vs_dielectric'][0]:.2f} / {r['metal_ch1_mean_on_metal_vs_dielectric'][1]:.2f} |"]
    L += ["", "Channel-order matrix (Pearson of predicted material channel vs GT), main setting:", "", "```",
          json.dumps(res["settings"]["main(black,512,1step)"]["material_channel_vs_GT_pearson"], indent=1), "```"]
    open(f"{OUT}/IID_CHECKS.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{OUT}/IID_CHECKS.md", time.time() - t0)


if __name__ == "__main__":
    main()
