"""Experiment 0, Amendment 2 (1): shared vs separate-BOTH-inputs bottlenecks, matched latent k AND
matched total parameter count (+-2%), shape-grouped bootstrap CIs, individual metrics.
  python -m gsm.exp0_bottleneck2 --eval_split val|test
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .exp0_analyze import load, pca_fit, flat, subsample, pairing, probe_rows, mlp

OUT = "outputs/exp0/bottleneck2"
VARIANTS = ["separate_own", "separate_both", "shared", "shared_private"]
METRICS = ["normal_deg", "depth_mae", "albedo_mae", "rough_mae", "metal_mae"]
HD = 256                       # decoder hidden width (identical decoders across variants)


def mlp_params(i, h, o):
    return i * h + h + h * h + h + h * o + o


def enc_spec(kind, d, k):
    """list of (input_dim, output_dim) encoders; d = per-modality input dim"""
    return {"separate_own": [(d, k // 2), (d, k // 2)],
            "separate_both": [(2 * d, k // 2), (2 * d, k // 2)],
            "shared": [(2 * d, k)],
            "shared_private": [(2 * d, k // 2), (d, k // 4), (d, k // 4)]}[kind]


def code_dims(kind, k):
    return {"separate_own": (k // 2, k // 2), "separate_both": (k // 2, k // 2), "shared": (k, k),
            "shared_private": (k // 2 + k // 4, k // 2 + k // 4)}[kind]


def total_params(kind, d, k, h):
    cg, cm = code_dims(kind, k)
    return sum(mlp_params(i, h, o) for i, o in enc_spec(kind, d, k)) + mlp_params(cg, HD, 4) + mlp_params(cm, HD, 5)


def solve_width(kind, d, k, target):
    best = min(range(16, 2048), key=lambda h: abs(total_params(kind, d, k, h) - target))
    return best, total_params(kind, d, k, best)


class Net(nn.Module):
    def __init__(self, kind, d, k, h):
        super().__init__()
        self.kind = kind
        self.encs = nn.ModuleList([mlp(i, h, o) for i, o in enc_spec(kind, d, k)])
        cg, cm = code_dims(kind, k)
        self.dg, self.dm = mlp(cg, HD, 4), mlp(cm, HD, 5)

    def forward(self, g, m, zero=None):
        gm = torch.cat([g, m], -1)
        if self.kind == "separate_own":
            zg, zm = self.encs[0](g), self.encs[1](m)
        elif self.kind == "separate_both":
            zg, zm = self.encs[0](gm), self.encs[1](gm)
        elif self.kind == "shared":
            zg = zm = self.encs[0](gm)
        else:
            zs, pg, pm = self.encs[0](gm), self.encs[1](g), self.encs[2](m)
            if zero == "shared": zs = torch.zeros_like(zs)
            if zero == "private": pg, pm = torch.zeros_like(pg), torch.zeros_like(pm)
            zg, zm = torch.cat([zs, pg], -1), torch.cat([zs, pm], -1)
        return self.dg(zg), self.dm(zm)


def train(kind, d, k, h, tr, seed, steps, bs=4096, dev="cuda"):
    torch.manual_seed(seed)
    g, m, tg, tm = [x.to(dev) for x in tr]
    net = Net(kind, d, k, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(g), (bs,), device=dev)
        pg, pm = net(g[i], m[i])
        loss = (1 - F.cosine_similarity(pg[:, :3], tg[i, :3], dim=-1)).mean() + (pg[:, 3] - tg[i, 3]).abs().mean() \
            + (pm - tm[i]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net.eval()


@torch.no_grad()
def row_errors(net, ev, dev="cuda", zero=None):
    g, m, tg, tm = [x.to(dev) for x in ev]
    pg, pm = net(g, m, zero)
    ang = torch.rad2deg(torch.acos((F.normalize(pg[:, :3], dim=-1) * tg[:, :3]).sum(-1).clamp(-1, 1)))
    return torch.stack([ang, (pg[:, 3] - tg[:, 3]).abs(), (pm[:, :3] - tm[:, :3]).abs().mean(-1),
                        (pm[:, 3] - tm[:, 3]).abs(), (pm[:, 4] - tm[:, 4]).abs()], -1).cpu()   # (rows,5)


def shape_sums(err, shape_idx, n_shapes):
    s = torch.zeros(n_shapes, err.shape[1], dtype=torch.float64); c = torch.zeros(n_shapes, dtype=torch.float64)
    s.index_add_(0, shape_idx, err.double()); c.index_add_(0, shape_idx, torch.ones(len(shape_idx), dtype=torch.float64))
    return s, c


def boot(sums, cnt, n_boot, gen, ref=None):
    """shape-grouped bootstrap of row-weighted means; if ref given -> difference (sums-ref) and relative."""
    S = len(cnt); est, rel = [], []
    for _ in range(n_boot):
        w = torch.bincount(torch.randint(S, (S,), generator=gen), minlength=S).double()
        mu = (w[:, None] * sums).sum(0) / (w * cnt).sum()
        if ref is not None:
            mr = (w[:, None] * ref).sum(0) / (w * cnt).sum()
            est.append((mu - mr).numpy()); rel.append(((mu - mr) / mr).numpy())
        else:
            est.append(mu.numpy())
    q = lambda a: np.percentile(np.stack(a), [2.5, 97.5], axis=0)
    return (q(est), q(rel)) if ref is not None else q(est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_split", default="val"); ap.add_argument("--layer", default="up2")
    ap.add_argument("--pca", type=int, default=256); ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n_boot", type=int, default=2000); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    TR, EV = load("train"), load(a.eval_split)
    # shape-grouped split audit: shapes disjoint; every variant of a shape in exactly one split
    st, se = set(TR["meta"]["shape"].tolist()), set(EV["meta"]["shape"].tolist())
    assert not (st & se), "shape leakage between train and eval"
    for D in (TR, EV):
        for s in D["meta"]["shape"].unique():
            sel = D["meta"]["shape"] == s
            assert sel.sum() == 6, "each shape must carry 3 draws x 2 regimes"
    l = a.layer
    gtr, _ = flat(TR["feats"]["normals"][l], TR["fg"]); mtr, _ = flat(TR["feats"]["iid"][l], TR["fg"])
    sub = subsample(len(gtr), 60000)
    pg, pm = pca_fit(gtr[sub], a.pca), pca_fit(mtr[sub], a.pca)
    P = lambda X, p: ((X - p[0]) @ p[1])
    def rows(D):
        G = P(D["feats"]["normals"][l], pg); M = P(D["feats"]["iid"][l], pm)
        r = probe_rows(D, pairing(D, "true"), G, M)
        oi = torch.nonzero(D["fg"])[:, 0]
        return (r[0].float(), r[1].float(), r[2].float(), r[3].float()), r[6], D["meta"]["shape"][oi]
    tr, _, _ = rows(TR); ev, reg, shp = rows(EV)
    shape_idx = torch.unique(shp, return_inverse=True)[1]; n_shapes = int(shape_idx.max()) + 1
    d = a.pca
    res = dict(eval_split=a.eval_split, layer=l, pca=d, ks=a.ks, seeds=a.seeds, steps=a.steps, n_boot=a.n_boot,
               n_eval_shapes=n_shapes, n_eval_rows=len(shp), widths={}, results={}, contrasts={})
    gen = torch.Generator().manual_seed(11)
    for k in a.ks:
        target = total_params("separate_both", d, k, 256)
        res["widths"][k] = {v: dict(zip(["enc_hidden", "params"], solve_width(v, d, k, target))) for v in VARIANTS}
        for v, w in res["widths"][k].items():
            assert abs(w["params"] / target - 1) <= 0.02, (k, v, w, target)
        S = {}
        for v in VARIANTS:
            h = res["widths"][k][v]["enc_hidden"]
            acc = {"": None, "private": None, "shared": None}
            for seed in a.seeds:
                net = train(v, d, k, h, tr, seed, a.steps, dev=dev)
                for z in (["", "private", "shared"] if v == "shared_private" else [""]):
                    s_, c_ = shape_sums(row_errors(net, ev, dev, z or None), shape_idx, n_shapes)
                    acc[z] = s_ / len(a.seeds) if acc[z] is None else acc[z] + s_ / len(a.seeds)
            for z, s_ in acc.items():
                if s_ is None: continue
                name = v + (f"|zero_{z}" if z else "")
                S[name] = s_
                mean = (s_.sum(0) / c_.sum()).numpy(); ci = boot(s_, c_, a.n_boot, gen)
                res["results"][f"k{k}|{name}"] = {m: dict(mean=float(mean[i]), ci95=[float(ci[0][i]), float(ci[1][i])])
                                                   for i, m in enumerate(METRICS)}
                res["results"][f"k{k}|{name}"]["params"] = res["widths"][k][v]["params"]
            print(f"k={k} {v} done {time.time() - t0:.0f}s", flush=True)
        for A, B in [("shared", "separate_both"), ("shared", "separate_own"), ("separate_both", "separate_own"),
                     ("shared_private", "separate_both")]:
            (dlo_hi, rlo_hi) = boot(S[A], c_, a.n_boot, gen, ref=S[B])
            dm = ((S[A].sum(0) - S[B].sum(0)) / c_.sum()).numpy(); rm = dm / (S[B].sum(0) / c_.sum()).numpy()
            res["contrasts"][f"k{k}|{A}-{B}"] = {m: dict(diff=float(dm[i]), diff_ci95=[float(dlo_hi[0][i]), float(dlo_hi[1][i])],
                                                         rel=float(rm[i]), rel_ci95=[float(rlo_hi[0][i]), float(rlo_hi[1][i])])
                                                  for i, m in enumerate(METRICS)}
    res["seconds"] = time.time() - t0
    json.dump(res, open(f"{OUT}/bottleneck2_{a.eval_split}.json", "w"), indent=1)
    # markdown
    L = [f"# Capacity-controlled bottlenecks ({a.eval_split}; Amendment 2, part 1)", "",
         f"Inputs: Marigold normals/IID {l} PCA-{d} (fit on train). True pairs. Total params matched to separate_both (h=256) within ±2%. "
         f"Mean over {len(a.seeds)} seeds. Shape-grouped bootstrap ({a.n_boot}×, {n_shapes} shapes) 95% CI.", ""]
    for k in a.ks:
        L += [f"## k = {k}", "", "| variant | params | " + " | ".join(METRICS) + " |", "|---|---:|" + "---|" * len(METRICS)]
        for key, r in res["results"].items():
            if not key.startswith(f"k{k}|"): continue
            L += [f"| {key.split('|', 1)[1]} | {r['params']} | " +
                  " | ".join(f"{r[m]['mean']:.4f} [{r[m]['ci95'][0]:.4f}, {r[m]['ci95'][1]:.4f}]" for m in METRICS) + " |"]
        L += ["", "| contrast (A − B) | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for key, r in res["contrasts"].items():
            if not key.startswith(f"k{k}|"): continue
            L += [f"| {key.split('|', 1)[1]} | " + " | ".join(
                f"{100 * r[m]['rel']:+.1f}% [{100 * r[m]['rel_ci95'][0]:+.1f}, {100 * r[m]['rel_ci95'][1]:+.1f}]" for m in METRICS) + " |"]
        L += [""]
    open(f"{OUT}/BOTTLENECK2_{a.eval_split}.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{OUT}/BOTTLENECK2_{a.eval_split}.md", res["seconds"])


if __name__ == "__main__":
    main()
