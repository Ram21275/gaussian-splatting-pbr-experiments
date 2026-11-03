"""Experiment 0, Amendment 2 (2): common-cause alignment controls.

Extract features (same up_blocks[0..2] hooks, same 16x16 pooling) on the SAME 512px input images
from control networks, then compare alignment against Marigold normals~IID with shape-grouped
bootstrap CIs.
  sources: randN{0,1}  random-init normals-architecture UNet (8-ch input), Marigold normals pipeline inputs
           randI{0,1}  random-init IID-architecture UNet (12-ch input), Marigold IID pipeline inputs
           base999 / base261  untouched SD2 UNet (sd2-community/stable-diffusion-2 mirror) on the VAE latent
                       of the same image, noised with fixed noise to t, empty-prompt embedding
  python -m gsm.exp0_controls extract --split val|test
  python -m gsm.exp0_controls analyze --split val|test
"""
import argparse, hashlib, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .scene import make_object, render_gt, INPUT_CAMS
from .exp0_extract import objects, RES, GRID, LAYERS, load_pipe, Hooks, SEED, STEPS
from .exp0_analyze import load, linear_cka, pairing, paired_rows, subsample, retrieval, obj_embed

DATA = "data/exp0"; OUT = "outputs/exp0/controls"
SD2 = "sd2-community/stable-diffusion-2"
CONTROL_SOURCES = ["randN0", "randN1", "randI0", "randI1", "base999", "base261"]


def state_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def render_inputs(split, dev):
    from PIL import Image
    imgs = []
    for (s, m, r) in objects(split):
        g = render_gt(make_object(s, m, r), INPUT_CAMS, dev, res=RES, lights=["L0_train"])
        for v in range(3):
            imgs.append(Image.fromarray((g["rgb_L0_train"][v].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)))
    return imgs


def run_random(arch, seed, imgs, dev, bs=6):
    """Marigold pipeline (normals or IID) with its UNet replaced by a random-init copy (fp32)."""
    from diffusers import UNet2DConditionModel
    pipe = load_pipe("normals" if arch == "N" else "iid", dev)
    torch.manual_seed(1000 + seed)
    pipe.unet = UNet2DConditionModel.from_config(pipe.unet.config).to(dev, torch.float16)
    hk = Hooks(pipe.unet); feats = {k: [] for k in LAYERS}
    for s in range(0, len(imgs), bs):
        g = torch.Generator(device=dev).manual_seed(SEED + s)
        pipe(imgs[s:s + bs], num_inference_steps=STEPS, ensemble_size=1, processing_resolution=RES,
             match_input_resolution=True, generator=g, output_type="np", batch_size=bs)
        for k in LAYERS: feats[k].append(hk.acts[k])
    h = state_hash(pipe.unet)
    del pipe; torch.cuda.empty_cache()
    return {k: torch.cat(v) for k, v in feats.items()}, h


@torch.no_grad()
def run_base(t_val, imgs, dev, bs=6):
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
    unet = UNet2DConditionModel.from_pretrained(SD2, subfolder="unet", variant="fp16", torch_dtype=torch.float16).to(dev)
    vae = AutoencoderKL.from_pretrained(SD2, subfolder="vae", variant="fp16", torch_dtype=torch.float16).to(dev)
    sched = DDPMScheduler.from_pretrained(SD2, subfolder="scheduler")
    mp = load_pipe("normals", "cpu")                          # empty-prompt embedding (same frozen OpenCLIP text encoder)
    emb = mp.empty_text_embedding.to(dev, torch.float16)
    hashes = dict(sd2_unet=state_hash(unet), sd2_vae=state_hash(vae), marigold_normals_vae=state_hash(mp.vae))
    del mp
    hk = Hooks(unet); feats = {k: [] for k in LAYERS}
    gen = torch.Generator(device=dev).manual_seed(SEED)
    for s in range(0, len(imgs), bs):
        x = torch.stack([torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.) for im in imgs[s:s + bs]])
        x = (x.permute(0, 3, 1, 2) * 2 - 1).to(dev, torch.float16)
        lat = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        noise = torch.randn(lat.shape, generator=gen, device=dev, dtype=lat.dtype)
        t = torch.full((len(lat),), t_val, device=dev, dtype=torch.long)
        xt = sched.add_noise(lat, noise, t)
        unet(xt, t, encoder_hidden_states=emb.expand(len(lat), -1, -1))
        for k in LAYERS: feats[k].append(hk.acts[k])
    del unet, vae; torch.cuda.empty_cache()
    return {k: torch.cat(v) for k, v in feats.items()}, hashes


def extract(split):
    dev = "cuda"; os.makedirs(DATA, exist_ok=True); t0 = time.time()
    objs = objects(split); n = len(objs)
    imgs = render_inputs(split, dev)
    # consistency: re-rendered inputs must equal the stored v2 renders (downsampled check)
    ref = torch.load(f"{DATA}/{split}.pt", weights_only=False)
    assert ref["objects"] == objs, "object list mismatch vs stored features"
    chk = torch.stack([torch.from_numpy(np.asarray(imgs[i], dtype=np.float32) / 255.) for i in range(0, 3 * n, 3)])
    chk = F.interpolate(chk.permute(0, 3, 1, 2), size=128, mode="area").permute(0, 2, 3, 1)
    diff = (chk - ref["full128"]["rgb"][:, 0].float()).abs().max().item()
    assert diff < 0.02, f"re-render differs from stored inputs: {diff}"
    del ref
    rec = dict(objects=objs, split=split, feats={}, hashes={}, timing={}, rerender_maxdiff=diff)
    for src in CONTROL_SOURCES:
        t1 = time.time()
        if src.startswith("rand"):
            f, h = run_random(src[4], int(src[5:]), imgs, dev); rec["hashes"][src] = h
        else:
            f, h = run_base(int(src[4:]), imgs, dev); rec["hashes"].update(h)
        for k, v in f.items():
            if not torch.isfinite(v.float()).all():
                raise FloatingPointError(f"{src}/{k} has non-finite activations")
        rec["feats"][src] = {k: v.reshape(n, 3, *v.shape[1:]) for k, v in f.items()}
        rec["timing"][src] = time.time() - t1
        print(f"[{split}] {src} {time.time() - t1:.0f}s", flush=True)
    torch.save(rec, f"{DATA}/controls_{split}.pt")
    print(json.dumps(dict(split=split, hashes=rec["hashes"], timing=rec["timing"], total=time.time() - t0)))


# ------------------------------------------------------------------ analysis (shape-grouped bootstrap)
def shape_boot_cka(X, Y, shapes, n_boot, gen):
    shapes = shapes.cpu(); us = shapes.unique(); out = []
    for _ in range(n_boot):
        pick = us[torch.randint(len(us), (len(us),), generator=gen)]
        cnt = torch.zeros(int(shapes.max()) + 1, dtype=torch.long); cnt.index_add_(0, pick, torch.ones_like(pick))
        rows = torch.repeat_interleave(torch.arange(len(shapes)), cnt[shapes])
        rows = rows.to(X.device)
        out.append(linear_cka(X[rows], Y[rows]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def analyze(split, n_boot=300, max_rows=8000, dev="cuda"):
    os.makedirs(OUT, exist_ok=True)
    D = load(split)
    C = torch.load(f"{DATA}/controls_{split}.pt", weights_only=False)
    assert C["objects"] == D["raw"]["objects"]
    for src, f in C["feats"].items():
        D["feats"][src] = {l: f[l].float().permute(0, 1, 3, 4, 2) for l in LAYERS}
    pairs = [("normals", "iid"), ("normals", "depth"),
             ("normals", "base999"), ("iid", "base999"), ("depth", "base999"),
             ("normals", "base261"), ("iid", "base261"), ("base999", "base261"),
             ("randN0", "randI1"), ("randN0", "randN1"), ("randI0", "randI1"),
             ("normals", "randN0"), ("iid", "randI0"), ("base999", "randN0")]
    res = dict(split=split, hashes=C["hashes"], n_boot=n_boot, cka={}, retrieval={})
    gen = torch.Generator().manual_seed(7)
    for l in LAYERS:
        res["cka"][l] = {}
        for mode in ["true", "same_shape", "any"]:
            perm = pairing(D, mode)
            gi, mi, reg, oi = paired_rows(D, perm)
            sel = subsample(len(oi), max_rows)
            shapes = D["meta"]["shape"][oi[sel]]
            shapes = torch.unique(shapes, return_inverse=True)[1]
            for a, b in pairs:
                X = D["feats"][a][l][gi][sel].to(dev); Y = D["feats"][b][l][mi][sel].to(dev)
                v = linear_cka(X, Y)
                ci = shape_boot_cka(X, Y, shapes, n_boot, gen)
                res["cka"][l][f"{a}~{b}|{mode}"] = dict(cka=v, ci95=ci)
        # raw retrieval with shape-bootstrap CI on per-query hits
        res["retrieval"][l] = {}
        key = D["meta"]["shape"]
        for a, b in [("normals", "iid"), ("normals", "depth"), ("normals", "base999"), ("iid", "base999"),
                     ("normals", "base261"), ("iid", "base261"), ("base261", "base999"),
                     ("randN0", "randI1"), ("randN0", "randN1"), ("normals", "randN0")]:
            Eq, Ec = obj_embed(D, a, l), obj_embed(D, b, l)
            q = F.normalize(Eq - Eq.mean(0), dim=-1); c = F.normalize(Ec - Ec.mean(0), dim=-1)
            S = q @ c.T; n = len(S)
            hit_all = (S.argmax(1) == torch.arange(n)).float()
            grp = (key * 2 + D["meta"]["regime"]); same = grp[:, None] == grp[None]
            hit_same = (S.masked_fill(~same, -1e9).argmax(1) == torch.arange(n)).float()
            us = key.unique(); bs_a, bs_s = [], []
            for _ in range(n_boot):
                pick = us[torch.randint(len(us), (len(us),), generator=gen)]
                idx = torch.cat([torch.nonzero(key == s_)[:, 0] for s_ in pick])
                bs_a.append(hit_all[idx].mean().item()); bs_s.append(hit_same[idx].mean().item())
            res["retrieval"][l][f"{a}->{b}"] = dict(
                r1_all=float(hit_all.mean()), ci_all=[float(np.percentile(bs_a, 2.5)), float(np.percentile(bs_a, 97.5))],
                r1_same_shape=float(hit_same.mean()), ci_same=[float(np.percentile(bs_s, 2.5)), float(np.percentile(bs_s, 97.5))],
                chance_all=1 / n, chance_same=float((1 / same.sum(1).float()).mean()))
        print(f"layer {l} done", flush=True)
    json.dump(res, open(f"{OUT}/controls_{split}.json", "w"), indent=1)
    # markdown
    L = [f"# Alignment controls ({split}): common-cause nulls (Amendment 2, part 2)", "",
         "Linear CKA with shape-grouped bootstrap 95% CI. Every source sees the same 512 px images, uses the same hooked blocks and the same 16×16 pooling.",
         "**CKA and retrieval are not evidence of a same embedding space** (pre-registered reading).", "",
         f"Hashes: `{json.dumps(C['hashes'])}`", ""]
    for l in LAYERS:
        L += [f"## layer {l}", "", "| pair | true | same-shape shuffle | any shuffle |", "|---|---|---|---|"]
        f_ = lambda d: f"{d['cka']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
        for a, b in pairs:
            t = res["cka"][l]
            L += [f"| {a}~{b} | {f_(t[f'{a}~{b}|true'])} | {f_(t[f'{a}~{b}|same_shape'])} | {f_(t[f'{a}~{b}|any'])} |"]
        L += ["", "| raw retrieval | R@1 all [CI] | R@1 same-shape [CI] (chance) |", "|---|---|---|"]
        for k_, r in res["retrieval"][l].items():
            L += [f"| {k_} | {r['r1_all']:.3f} [{r['ci_all'][0]:.3f}, {r['ci_all'][1]:.3f}] | "
                  f"{r['r1_same_shape']:.3f} [{r['ci_same'][0]:.3f}, {r['ci_same'][1]:.3f}] ({r['chance_same']:.2f}) |"]
        L += [""]
    open(f"{OUT}/CONTROLS_{split}.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{OUT}/CONTROLS_{split}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--split", required=True)
    a = ap.parse_args()
    extract(a.split) if a.cmd == "extract" else analyze(a.split)
