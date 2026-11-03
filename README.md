# Geometry/PBR representation and single-flow Gaussian experiments

This is a research workspace for a **DreamLifting-inspired feasibility prototype**. It is not a DreamLifting, LGAA or SplatFlow reproduction. The DreamLifting repository (audited 2026-11-02) contains only a README and assets, with no code or weights.

The research plan is in [`Meeting_Experiment_Agent_Prompt.md`](Meeting_Experiment_Agent_Prompt.md). The work runs in this order:

| stage | question | status |
|---|---|---|
| **Experiment 0** | Do geometry and PBR features from posed multiview RGB share structure, and can a compact shared representation preserve both? | v2 analysis complete (val + test); controls round (Amendment 2) complete on val, test running |
| Experiment A | Can one conditional rectified-flow core drive a joint Gaussian/PBR decoder (A0 direct / A1 diffusion / A2 flow)? | GPU smoke test and timing only; the pilot is paused behind Exp 0 |
| Experiment B | Rendering-anchored stochastic-interpolant refinement (Hypothesis 3) | queued |

## Experiment 0 at a glance (procedural objects, frozen Marigold v1-1 extractors)
- **Plan, locked decision rule and dated amendments:** [`experiments/meeting/EXP0_PLAN.md`](experiments/meeting/EXP0_PLAN.md)
- **v2 summary, val and test side by side:** [`results/exp0/EXP0_SUMMARY.md`](results/exp0/EXP0_SUMMARY.md)
- **Controls round interpretation (written before test):** [`results/exp0/ROUND2_INTERPRETATION.md`](results/exp0/ROUND2_INTERPRETATION.md)
- **Figures:** `results/exp0/figures_{val,test}/`
- **Controls:**
  - capacity-matched bottlenecks: `results/exp0/bottleneck2/`
  - alignment nulls: `results/exp0/controls/`
  - extractor checks: `results/exp0/iid_checks/`

**Current reading**
- Frozen geometry and PBR features are mutually informative, object-specifically.
- At matched latent size **and** matched parameters, a shared code is *sufficient*: it is within about ±5% of separate codes that also see both inputs, on geometry and albedo. It is not shown to be better.
- **CKA cannot establish a shared space here.** Random-init UNets on the same images align more strongly than the trained extractors, and the trained models share a byte-identical VAE and an SD2 initialisation.
- Marigold-IID does not recover metallic on these renders. A black background is its main domain gap for albedo.

The v1 run (`*_v1_crossed.*`) is void: it had only 3 training material palettes. It is kept for transparency.

## Layout
```
gsm/                        Exp 0 code + the from-scratch GPU renderer prototype
  scene.py                  procedural SDF objects (shape and material drawn independently), cameras, GGX, ray-marched GT
  exp0_extract.py           renders + frozen Marigold feature extraction (UNet up-block hooks)
  exp0_analyze.py           sanity, variance partition, CKA, retrieval, probes, bottlenecks (v2)
  exp0_bottleneck2.py       shared vs separate-both-inputs, parameter-matched, shape-grouped bootstrap
  exp0_controls.py          random-init / base-SD2 alignment controls
  exp0_iid_checks.py        Marigold-IID convention and accuracy checks
  exp0_report.py, exp0_figures.py
  splat.py, latent.py, model.py, prep.py   pure-PyTorch Gaussian rasterizer + aligned-anchor latent (Exp A/B prototype, not yet used)
gs_initial_experiments/     supplied Exp A/B starter package (tests pass on this machine; see its README/METHODS)
experiments/meeting/        plans, launch scripts
results/                    copies of reports/JSON/CSV/figures (no checkpoints or feature tensors)
env/                        pip freeze + machine info
py                          venv wrapper used by every command
```

## Environment
- Python 3.13, torch 2.8.0+cu128, diffusers 0.37.0. The full freeze is in `env/requirements.lock.txt`.
- Hardware: RTX 6000 Ada (48 GB, shared with other jobs) and an RTX A400.
- Heavy artefacts (feature tensors, about 15 GB; checkpoints) stay on local disk and are not in this repo.
- Models used:
  - `prs-eth/marigold-{normals,depth,iid-appearance}-v1-1`
  - `sd2-community/stable-diffusion-2`, a community mirror of the removed stabilityai repo

## Rerun Experiment 0
```bash
./py -m gsm.exp0_extract --split {train,val,test}
./py -m gsm.exp0_analyze --train_split train --eval_split {val,test}
./py -m gsm.exp0_report  --split {val,test}; ./py -m gsm.exp0_figures --split {val,test}
./py -m gsm.exp0_bottleneck2 --eval_split {val,test}
./py -m gsm.exp0_controls extract --split {val,test}; ./py -m gsm.exp0_controls analyze --split {val,test}
./py -m gsm.exp0_iid_checks
```
