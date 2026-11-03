"""Experiment 0 analyses on extracted features (see experiments/meeting/EXP0_PLAN.md).

  python -m gsm.exp0_analyze --eval_split val   # decision split
  python -m gsm.exp0_analyze --eval_split test  # once, after the rule is locked

Patch-level units: (object, view, 16x16 patch) with GT foreground fraction > 0.5.
G = geometry features (marigold-normals), M = PBR features (marigold-iid-appearance),
Dp = marigold-depth (geometry-geometry positive control). Targets are GT maps only.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = "data/exp0"
OUT = "outputs/exp0"
LAYERS = ["up0", "up1", "up2"]
EXTRACTORS = ["normals", "depth", "iid"]


# ------------------------------------------------------------------ loading
def load(split):
    r = torch.load(f"{DATA}/{split}.pt", weights_only=False)
    objs = r["objects"]
    n = len(objs)
    T = r["targets"]                                    # each (n,3,G,G,c)
    fg = (T["mask"][..., 0] > 0.5)                      # (n,3,G,G)
    from .scene import cam_tensors, INPUT_CAMS
    w2c, _ = cam_tensors(INPUT_CAMS)
    ncam = torch.einsum("vij,nvhwj->nvhwi", w2c[:, :3, :3], T["normal_world"])
    ncam = F.normalize(ncam, dim=-1)
    depth = T["depth"] / T["mask"].clamp(min=1e-3)
    geo_t = torch.cat([ncam, depth], -1)                # 4: camera normal + depth
    alb = T["albedo"] / T["mask"].clamp(min=1e-3)
    ro = T["rough"] / T["mask"].clamp(min=1e-3); me = T["metal"] / T["mask"].clamp(min=1e-3)
    mat_t = torch.cat([alb, ro, me], -1)                # 5
    meta = dict(shape=torch.tensor([o[0] for o in objs]), mseed=torch.tensor([o[1] for o in objs]),
                regime=torch.tensor([0 if o[2] == "part" else 1 for o in objs]))
    feats = {e: {l: r["feats"][e][l].float().permute(0, 1, 3, 4, 2) for l in LAYERS} for e in EXTRACTORS}  # (n,3,G,G,C)
    feats["gt_geo"] = {"maps": geo_t}; feats["gt_mat"] = {"maps": mat_t}
    return dict(n=n, fg=fg, geo=geo_t, mat=mat_t, part=T["part"][..., 0], matid=T["matid"][..., 0],
                feats=feats, meta=meta, raw=r)


def flat(x, fg):
    """(n,3,G,G,C) -> (P,C) over foreground patches; also returns object index per patch."""
    idx = torch.nonzero(fg)                      # (P,4)
    return x[fg], idx[:, 0]


# ------------------------------------------------------------------ alignment metrics
def linear_cka(X, Y):
    X = X.float() - X.float().mean(0); Y = Y.float() - Y.float().mean(0)
    X = X / X.norm(); Y = Y / Y.norm()               # scale-invariant; keeps fp32 well-conditioned
    hsic = (X.T @ Y).double().norm() ** 2
    return float(hsic / ((X.T @ X).double().norm() * (Y.T @ Y).double().norm()))


def subsample(P, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(P, generator=g)[:k] if P > k else torch.arange(P)




def pca_fit(X, k=64):
    """PCA on CPU (this torch build's CUDA cuSOLVER linalg has a symbol mismatch); result on X.device."""
    dev = X.device
    X = X.float().cpu()
    mu = X.mean(0)
    q = min(k, X.shape[0] - 1, X.shape[1])
    U, S, V = torch.pca_lowrank(X - mu, q=q, center=False)
    return mu.to(dev), V[:, :q].to(dev)


def material_variance_fraction(F_, D, fg):
    """F_ (n,3,G,G,C). Groups objects with identical (shape, regime) -> same geometry and view,
    differing only by material seed. Returns var across material seeds / total var (on patches
    foreground in the group)."""
    key = D["meta"]["shape"] * 2 + D["meta"]["regime"]
    within, total_list = [], []
    for k in key.unique():
        ids = torch.nonzero(key == k)[:, 0]
        if len(ids) < 2:
            continue
        m = fg[ids].all(0)                               # identical geometry -> identical mask
        X = F_[ids][:, m]                                # (S,P,C)
        within.append(((X - X.mean(0, keepdim=True)) ** 2).sum(-1).mean(0))  # per patch
        total_list.append(X.reshape(-1, X.shape[-1]))
    Xall = torch.cat(total_list)
    tot = ((Xall - Xall.mean(0)) ** 2).sum(-1).mean()
    return float(torch.cat(within).mean() / tot)


# ------------------------------------------------------------------ bottleneck models
def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class Bottleneck(nn.Module):
    """kind in {separate, shared, shared_private}; total latent k; equal encoder/decoder widths."""

    def __init__(self, kind, dg, dm, k, h=256, og=4, om=5):
        super().__init__()
        self.kind = kind
        if kind == "separate":
            self.eg, self.em = mlp(dg, h, k // 2), mlp(dm, h, k // 2)
            self.dg, self.dm = mlp(k // 2, h, og), mlp(k // 2, h, om)
        elif kind == "shared":
            self.e = mlp(dg + dm, h, k)
            self.dg, self.dm = mlp(k, h, og), mlp(k, h, om)
        else:
            ks, kp = k // 2, k // 4
            self.e = mlp(dg + dm, h, ks); self.eg, self.em = mlp(dg, h, kp), mlp(dm, h, kp)
            self.dg, self.dm = mlp(ks + kp, h, og), mlp(ks + kp, h, om)

    def forward(self, g, m, zero=None):
        if self.kind == "separate":
            zg, zm = self.eg(g), self.em(m)
            return self.dg(zg), self.dm(zm)
        if self.kind == "shared":
            z = self.e(torch.cat([g, m], -1))
            return self.dg(z), self.dm(z)
        zs = self.e(torch.cat([g, m], -1)); pg = self.eg(g); pm = self.em(m)
        if zero == "shared":
            zs = torch.zeros_like(zs)
        if zero == "private":
            pg, pm = torch.zeros_like(pg), torch.zeros_like(pm)
        return self.dg(torch.cat([zs, pg], -1)), self.dm(torch.cat([zs, pm], -1))


def geo_err(p, t):
    n = F.normalize(p[:, :3], dim=-1)
    ang = torch.rad2deg(torch.acos((n * t[:, :3]).sum(-1).clamp(-1, 1))).mean()
    dep = (p[:, 3] - t[:, 3]).abs().mean()
    return float(ang), float(dep)


def mat_err(p, t):
    return float((p[:, :3] - t[:, :3]).abs().mean()), float((p[:, 3] - t[:, 3]).abs().mean()), \
        float((p[:, 4] - t[:, 4]).abs().mean())


def train_bottleneck(kind, k, tr, seed, steps=3000, bs=4096, dev="cuda"):
    torch.manual_seed(seed)
    g, m, tg, tm = [x.to(dev) for x in tr]
    model = Bottleneck(kind, g.shape[1], m.shape[1], k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        i = torch.randint(0, len(g), (bs,), device=dev)
        pg, pm = model(g[i], m[i])
        # geometry: cosine for normals + L1 depth ; materials: L1. Equal weight per modality.
        lg = (1 - F.cosine_similarity(pg[:, :3], tg[i, :3], dim=-1)).mean() + (pg[:, 3] - tg[i, 3]).abs().mean()
        lm = (pm - tm[i]).abs().mean()
        loss = lg + lm
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def eval_bottleneck(model, ev, zero=None, dev="cuda"):
    g, m, tg, tm = [x.to(dev) for x in ev]
    pg, pm = model(g, m, zero)
    a, d = geo_err(pg, tg); al, ro, me = mat_err(pm, tm)
    # scalar summaries normalised later against separate
    return dict(normal_deg=a, depth_mae=d, albedo_mae=al, rough_mae=ro, metal_mae=me)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ================================================================== analyses
def to_lin(x):
    return x.clamp(0, 1) ** 2.2


def sanity(D):
    """Extractor accuracy vs GT at 128 px (evaluation-only alignment for affine depth)."""
    r = D["raw"]; full = r["full128"]; pr = r["preds"]
    from .scene import cam_tensors, INPUT_CAMS
    w2c, _ = cam_tensors(INPUT_CAMS)
    m = full["mask"][..., 0].float() > 0.5                               # (n,3,H,W)
    reg = D["meta"]["regime"][:, None, None, None].expand_as(m)
    out = {}
    nc = torch.einsum("vij,nvhwj->nvhwi", w2c[:, :3, :3], full["normal_world"].float())
    nc = F.normalize(nc, dim=-1)
    pn = pr["normals"][:, :, 0].float() * torch.tensor([1., -1., -1.])  # OpenGL cam -> OpenCV cam (measured)
    ang = torch.rad2deg(torch.acos((F.normalize(pn, dim=-1) * nc).sum(-1).clamp(-1, 1)))
    out["normals_angular_err_deg_mean"] = float(ang[m].mean()); out["normals_angular_err_deg_median"] = float(ang[m].median())
    # depth: per-view least-squares scale/shift (eval only) -> AbsRel
    gd = full["depth"][..., 0].float() / full["mask"][..., 0].float().clamp(min=1e-3)
    pd = pr["depth"][:, :, 0, ..., 0].float()
    rel = []
    for i in range(pd.shape[0]):
        for v in range(3):
            mm = m[i, v]
            A = torch.stack([pd[i, v][mm], torch.ones(int(mm.sum()))], -1)
            sol = torch.linalg.lstsq(A, gd[i, v][mm][:, None]).solution
            rel.append(((A @ sol)[:, 0] - gd[i, v][mm]).abs() / gd[i, v][mm])
    out["depth_absrel_affine_aligned"] = float(torch.cat(rel).mean())
    ga = full["albedo"].float() / full["mask"].float().clamp(min=1e-3)
    pa = to_lin(pr["iid"][:, :, 0].float())                              # IID albedo is sRGB-space
    gr = full["rough"][..., 0].float() / full["mask"][..., 0].float().clamp(min=1e-3)
    gm = full["metal"][..., 0].float() / full["mask"][..., 0].float().clamp(min=1e-3)
    prr = pr["iid"][:, :, 1, ..., 0].float(); pmm = pr["iid"][:, :, 1, ..., 1].float()
    for name, sel in [("all", m), ("part", m & (reg == 0)), ("partition", m & (reg == 1))]:
        out[f"albedo_mae_{name}"] = float((pa - ga).abs().mean(-1)[sel].mean())
        out[f"roughness_mae_{name}"] = float((prr - gr).abs()[sel].mean())
        out[f"metallic_mae_{name}"] = float((pmm - gm).abs()[sel].mean())
    # structure-vs-scale diagnostics (evaluation-only): per-view least-squares gain for albedo,
    # and pooled Pearson correlation per quantity
    sc = []
    for i in range(pa.shape[0]):
        for v in range(3):
            mm = m[i, v]
            x, y = pa[i, v][mm], ga[i, v][mm]
            gain = (x * y).sum() / (x * x).sum().clamp(min=1e-8)
            sc.append((gain * x - y).abs().mean(-1))
    out["albedo_mae_gain_aligned_evalonly"] = float(torch.cat(sc).mean())
    pear = lambda x, y: float(torch.corrcoef(torch.stack([x, y]))[0, 1])
    out["albedo_pearson"] = pear(pa[m].reshape(-1), ga[m].reshape(-1))
    out["roughness_pearson"] = pear(prr[m], gr[m]); out["metallic_pearson"] = pear(pmm[m], gm[m])
    # trivial baselines (predict GT mean of this split) for scale
    out["albedo_mae_meanbaseline"] = float((ga[m] - ga[m].mean(0)).abs().mean())
    out["roughness_mae_meanbaseline"] = float((gr[m] - gr[m].mean()).abs().mean())
    out["metallic_mae_meanbaseline"] = float((gm[m] - gm[m].mean()).abs().mean())
    return out


def pairing(D, mode, seed=0):
    """Object permutation. 'true': identity. 'any': random other object (same regime).
    'same_shape': same shape & regime, different material seed (geometry identical)."""
    n = D["n"]; g = torch.Generator().manual_seed(seed)
    meta = D["meta"]
    if mode == "true":
        return torch.arange(n)
    perm = torch.arange(n)
    if mode == "same_shape":
        key = meta["shape"] * 2 + meta["regime"]
        for k in key.unique():
            ids = torch.nonzero(key == k)[:, 0]
            perm[ids] = ids.roll(1)                              # derangement within group
        return perm
    for r in [0, 1]:
        ids = torch.nonzero(meta["regime"] == r)[:, 0]
        while True:
            p = ids[torch.randperm(len(ids), generator=g)]
            if (meta["shape"][p] != meta["shape"][ids]).all():
                break
        perm[ids] = p
    return perm


def paired_rows(D, perm):
    """Patch rows where object i (G side) and perm[i] (M side) are both foreground at the same
    (view,row,col). Returns index tuples for G side and M side + regime per row."""
    fg2 = D["fg"] & D["fg"][perm]
    idx = torch.nonzero(fg2)
    gi = (idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3])
    mi = (perm[idx[:, 0]], idx[:, 1], idx[:, 2], idx[:, 3])
    return gi, mi, D["meta"]["regime"][idx[:, 0]], idx[:, 0]


def feats_of(D, ext, layer):
    return D["feats"][ext]["maps"] if ext.startswith("gt_") else D["feats"][ext][layer]


def cka_table(D, layer, dev, n_boot=50, max_rows=8000):
    res = {}
    pairs = [("normals", "iid"), ("normals", "depth"), ("depth", "iid"),
             ("normals", "gt_geo"), ("normals", "gt_mat"), ("iid", "gt_geo"), ("iid", "gt_mat"),
             ("depth", "gt_geo"), ("depth", "gt_mat"), ("gt_geo", "gt_mat")]
    for mode in ["true", "same_shape", "any"]:
        perm = pairing(D, mode)
        gi, mi, reg, oi = paired_rows(D, perm)
        sel = subsample(len(oi), max_rows)
        for a, b in pairs:
            X = feats_of(D, a, layer)[gi][sel].to(dev); Y = feats_of(D, b, layer)[mi][sel].to(dev)
            val = linear_cka(X, Y)
            # object-bootstrap CI
            objs = oi[sel].to(dev); uo = objs.unique(); bs = []
            gen = torch.Generator(device="cpu").manual_seed(1)
            for _ in range(n_boot):
                pick = uo[torch.randint(len(uo), (len(uo),), generator=gen).to(dev)]
                cnt = torch.bincount(pick, minlength=int(objs.max()) + 1)
                w = cnt[objs].float()
                rows = torch.repeat_interleave(torch.arange(len(objs), device=dev), w.long())
                bs.append(linear_cka(X[rows], Y[rows]))
            res[f"{a}~{b}|{mode}"] = dict(cka=val, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return res


def obj_embed(D, ext, layer):
    X = feats_of(D, ext, layer)                                  # (n,3,G,G,C)
    fg = D["fg"][..., None].float()
    e = (X * fg).sum((2, 3)) / fg.sum((2, 3)).clamp(min=1)       # (n,3,C)
    return e.reshape(e.shape[0], -1)


def retrieval(Eq, Ec, D):
    """Eq query (n,d), Ec candidates (n,d) in the SAME space. R@1 over all, and within same-shape groups."""
    q = F.normalize(Eq - Eq.mean(0), dim=-1); c = F.normalize(Ec - Ec.mean(0), dim=-1)
    S = q @ c.T
    n = len(S)
    r1_all = float((S.argmax(1) == torch.arange(n)).float().mean())
    key = D["meta"]["shape"] * 2 + D["meta"]["regime"]
    same = key[:, None] == key[None]
    Sm = S.masked_fill(~same, -1e9)
    r1_same = float((Sm.argmax(1) == torch.arange(n)).float().mean())
    group = same.sum(1).float()
    return dict(r1_all=r1_all, chance_all=1 / n, r1_same_shape=r1_same, chance_same_shape=float((1 / group).mean()))


def ridge_fit(X, Y, lam=1e-1):
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    W = torch.linalg.solve(Xc.T @ Xc + lam * len(X) * torch.eye(X.shape[1], dtype=X.dtype), Xc.T @ (Y - my))
    return lambda Z: (Z - mx) @ W + my


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum(0).sum() / ((t - t.mean(0)) ** 2).sum(0).sum())


def probe_rows(D, perm, gfeat, mfeat):
    gi, mi, reg, oi = paired_rows(D, perm)
    return gfeat[gi], mfeat[mi], D["geo"][gi], D["mat"][mi], D["geo"][mi], D["mat"][gi], reg


def auc(score, label):
    o = score.argsort(); l = label[o].float()
    npos = l.sum(); nneg = len(l) - npos
    ranks = torch.arange(1, len(l) + 1, dtype=torch.float64)
    return float((ranks[l.bool()].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos > 0 and nneg > 0 else float("nan")


def adjacent_pairs(D, feat, label_map):
    """Right/down neighbouring foreground patches within a view -> (|fa-fb|, same-label)."""
    X, Y, R = [], [], []
    fg = D["fg"]
    for dy, dx in [(0, 1), (1, 0)]:
        a = fg[:, :, : fg.shape[2] - dy, : fg.shape[3] - dx] & fg[:, :, dy:, dx:]
        idx = torch.nonzero(a)
        ia = (idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]); ib = (idx[:, 0], idx[:, 1], idx[:, 2] + dy, idx[:, 3] + dx)
        X.append((feat[ia] - feat[ib]).abs()); Y.append(label_map[ia] == label_map[ib]); R.append(D["meta"]["regime"][idx[:, 0]])
    return torch.cat(X), torch.cat(Y), torch.cat(R)


def logreg(Xtr, ytr, steps=500, dev="cpu"):
    Xtr = Xtr.to(dev); ytr = ytr.float().to(dev)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lin = nn.Linear(Xtr.shape[1], 1).to(dev)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=steps, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(lin((Xtr - mu) / sd)[:, 0], ytr) + 1e-3 * lin.weight.pow(2).sum()
        l.backward(); return l
    opt.step(closure)
    return lambda Z: lin((Z.to(dev) - mu) / sd)[:, 0].detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--eval_split", default="val")
    ap.add_argument("--layer", default="up2")   # v2 amendment: chosen on train-only held-out-shape diagnostic
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    TR = load(args.train_split); EV = load(args.eval_split)
    res = dict(train_split=args.train_split, eval_split=args.eval_split, layer=args.layer,
               n_train_objects=TR["n"], n_eval_objects=EV["n"])
    # 0. sanity
    res["extractor_sanity"] = sanity(EV)
    print("sanity", json.dumps(res["extractor_sanity"], indent=1), flush=True)
    # 1. factor-partitioned variance (eval split; geometry identical within (shape,regime) groups)
    if "variance" not in args.skip:
        res["material_variance_fraction"] = {f"{e}/{l}": material_variance_fraction(EV["feats"][e][l], EV, EV["fg"])
                                             for e in EXTRACTORS for l in LAYERS}
        res["material_variance_fraction"]["gt_geo"] = material_variance_fraction(EV["geo"], EV, EV["fg"])
        res["material_variance_fraction"]["gt_mat"] = material_variance_fraction(EV["mat"], EV, EV["fg"])
        print("matvar", json.dumps(res["material_variance_fraction"], indent=1), flush=True)
    # 2. natural alignment (no adapter): CKA true vs shuffled
    if "cka" not in args.skip:
        res["cka"] = {l: cka_table(EV, l, dev) for l in LAYERS}
        print("cka", json.dumps({k: round(v["cka"], 3) for k, v in res["cka"][args.layer].items()}, indent=1), flush=True)
    # 2b/3. retrieval: raw (natural) vs ridge adapter (learned), normals->iid, per layer
    if "retrieval" not in args.skip:
        res["retrieval"] = {}
        for l in LAYERS:
            Eg, Em = obj_embed(EV, "normals", l), obj_embed(EV, "iid", l)
            Ed = obj_embed(EV, "depth", l)
            Tg, Tm = obj_embed(TR, "normals", l), obj_embed(TR, "iid", l)
            mu, V = pca_fit(Tg, 128); mu2, V2 = pca_fit(Tm, 128)
            f = ridge_fit((Tg - mu) @ V, (Tm - mu2) @ V2, lam=1e-2)
            res["retrieval"][l] = dict(raw_normals_to_iid=retrieval(Eg, Em, EV),
                                       raw_normals_to_depth=retrieval(Eg, Ed, EV),
                                       raw_iid_to_iid_selfcheck=retrieval(Em, Em, EV),
                                       adapter_normals_to_iid=retrieval(f((Eg - mu) @ V), (Em - mu2) @ V2, EV))
        print("retrieval", json.dumps(res["retrieval"][args.layer], indent=1), flush=True)
    # PCA-64 patch features fit on train foreground patches
    l = args.layer
    gtr, _ = flat(TR["feats"]["normals"][l], TR["fg"]); mtr, _ = flat(TR["feats"]["iid"][l], TR["fg"])
    sub = subsample(len(gtr), 60000)
    pg = pca_fit(gtr[sub].to(dev), args.pca); pm = pca_fit(mtr[sub].to(dev), args.pca)
    proj = lambda X, p: ((X.to(dev) - p[0]) @ p[1]).cpu()
    def proj_maps(D):
        return proj(D["feats"]["normals"][l].reshape(-1, D["feats"]["normals"][l].shape[-1]), pg).reshape(*D["fg"].shape, -1), \
               proj(D["feats"]["iid"][l].reshape(-1, D["feats"]["iid"][l].shape[-1]), pm).reshape(*D["fg"].shape, -1)
    TRg, TRm = proj_maps(TR); EVg, EVm = proj_maps(EV)
    # 4. probes (cross-prediction), ridge on PCA-64; true vs same-shape shuffle vs any shuffle
    if "probes" not in args.skip:
        res["probes"] = {}
        for mode in ["true", "same_shape", "any"]:
            a = probe_rows(TR, pairing(TR, mode), TRg, TRm); b = probe_rows(EV, pairing(EV, mode), EVg, EVm)
            # a/b: g, m, geo(G-side), mat(M-side), geo(M-side), mat(G-side), regime
            out = {}
            # a/b tuple: 0 G-feat(G obj) 1 M-feat(M obj) 2 geo(G obj) 3 mat(M obj) 4 geo(M obj) 5 mat(G obj) 6 regime
            probes = {"G->mat_of_M_object": (0, 3), "M->geo_of_G_object": (1, 2)}   # cross-object: depend on pairing
            if mode == "true":                                                        # own-object: pairing-independent
                probes.update({"G->geo_own": (0, 2), "G->mat_own": (0, 5), "M->geo_own": (1, 4), "M->mat_own": (1, 3)})
            for name, (xi, ti) in probes.items():
                f = ridge_fit(a[xi].double(), a[ti].double(), lam=1e-3)
                p = f(b[xi].double())
                out[name] = {"all": r2(p, b[ti].double()),
                             "part": r2(p[b[6] == 0], b[ti][b[6] == 0].double()),
                             "partition": r2(p[b[6] == 1], b[ti][b[6] == 1].double())}
            res["probes"][mode] = out
        print("probes", json.dumps(res["probes"], indent=1), flush=True)
        # affinity probes: same material / same part between adjacent patches
        res["affinity_auc"] = {}
        for fname, (Ftr, Fev) in {"G": (TRg, EVg), "M": (TRm, EVm)}.items():
            for lab in ["matid", "part"]:
                Xa, ya, _ = adjacent_pairs(TR, Ftr, TR[lab]); Xb, yb, rb = adjacent_pairs(EV, Fev, EV[lab])
                f = logreg(Xa, ya, dev=dev); s = f(Xb)
                res["affinity_auc"][f"{fname}->same_{lab}"] = {"part": auc(s[rb == 0], yb[rb == 0]),
                                                                "partition": auc(s[rb == 1], yb[rb == 1]),
                                                                "pos_rate": float(yb.float().mean())}
        print("affinity", json.dumps(res["affinity_auc"], indent=1), flush=True)
    # mean predictor (train foreground mean) on eval foreground patches, same error definitions
    gm_, mm_ = TR["geo"][TR["fg"]].mean(0), TR["mat"][TR["fg"]].mean(0)
    eg_, em_ = EV["geo"][EV["fg"]], EV["mat"][EV["fg"]]
    a_, d_ = geo_err(gm_.expand_as(eg_).clone(), eg_); al_, ro_, me_ = mat_err(mm_.expand_as(em_).clone(), em_)
    res["mean_predictor"] = dict(normal_deg=a_, depth_mae=d_, albedo_mae=al_, rough_mae=ro_, metal_mae=me_)
    # 5. bottlenecks: separate / shared / shared+private, matched k; true vs shuffled pairing
    if "bottleneck" not in args.skip:
        rows = []
        inputs = {"marigold": (TRg, TRm, EVg, EVm), "gt_oracle": (TR["geo"], TR["mat"], EV["geo"], EV["mat"])}
        for src, (tg_, tm_, eg_, em_) in inputs.items():
            for mode in (["true", "same_shape", "any"] if src == "marigold" else ["true"]):
                a = probe_rows(TR, pairing(TR, mode), tg_, tm_); b = probe_rows(EV, pairing(EV, mode), eg_, em_)
                tr = (a[0].float(), a[1].float(), a[2].float(), a[3].float())
                ev = (b[0].float(), b[1].float(), b[2].float(), b[3].float()); reg = b[6]
                for k in args.ks:
                    for kind in ["separate", "shared", "shared_private"]:
                        for seed in args.seeds:
                            m = train_bottleneck(kind, k, tr, seed, steps=args.steps, dev=dev)
                            for rname, sel in [("all", slice(None)), ("part", reg == 0), ("partition", reg == 1)]:
                                evs = tuple(x[sel] for x in ev)
                                for zero in ([None, "private", "shared"] if kind == "shared_private" else [None]):
                                    rows.append(dict(input=src, pairing=mode, k=k, kind=kind, seed=seed, regime=rname,
                                                     zeroed=zero or "", params=n_params(m), **eval_bottleneck(m, evs, zero, dev)))
                print(f"bottleneck {src}/{mode} done {time.time() - t0:.0f}s", flush=True)
        import csv
        path = f"{OUT}/bottleneck_{args.eval_split}{args.tag}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        res["bottleneck_csv"] = path
    res["seconds"] = time.time() - t0
    with open(f"{OUT}/exp0_{args.eval_split}{args.tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("done", res["seconds"])


if __name__ == "__main__":
    main()
