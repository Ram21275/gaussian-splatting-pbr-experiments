# Initial Gaussian-splatting experiments

A runnable **DreamLifting-inspired synthetic feasibility prototype**, with one joint generative core, posed multiview RGB inputs, aligned Gaussian geometry/material targets, training/evaluation code, and resumable checkpoints. Start here before connecting a large pretrained pipeline.

This is an implementation starter for the two meeting experiments. It is **not** DreamLifting, LGAA, SplatFlow or a pretrained foundation model. The included CPU runs are explicitly DEBUG checks, not hypothesis evidence.

## What is implemented

| Experiment | Variant | Training / sampling |
|---|---|---|
| A: single-core feasibility | A0 | Deterministic joint-state predictor; also the frozen input-only coarse model for B |
| A | A1 | Joint-state VP noise-prediction diffusion; deterministic DDIM sampling |
| A | A2 | Joint-state rectified flow; Gaussian noise to aligned target |
| B: hypothesis | B0 | Input-only coarse source to target, deterministic residual flow |
| B | B1 | Fixed stochastic interpolant; shared-trunk velocity and noise/score heads; corrected SDE |
| B | B2 | B1 plus an observed-view penalty on short **learned rollouts** |
| B: warm-start control | B3 | **Reuse A1 weights**, corrupt A0 estimate, then reverse DDIM; no redundant baseline training |
| B: regularizer controls | B0_anchor / B3_anchor | Add the rollout rendering penalty to coarse flow / coarse-start diffusion |

All variants allocate the same trunk and two heads. The unused head receives no gradients in some controls, so manifests report both allocated parameters and parameters with gradients. One A0 pass is additional frozen preprocessing for B; pipeline NFE includes it. A0's setup/training cost remains visible.

The included conditioner is frozen deterministic 8×8 RGB pooling with camera/light metadata, followed by a trainable projection. It is not a pretrained image encoder. The same fixed analytic decoder is used for every variant. Replace these with the project's frozen features and decoder before making architectural claims about a larger system.

## 1. Environment

Use a dedicated environment with a PyTorch build appropriate for the machine. Preserve the project's existing compatible CUDA renderer environment if it already works. Python 3.10+ is supported by the code; 3.11 is a convenient new-environment choice.

```bash
python -m venv .venv
source .venv/bin/activate
# Install a compatible CPU/CUDA PyTorch build first, using the official selector.
# CPU-only check, if no existing PyTorch is installed:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m gs_meeting.cli doctor --out machine.json
python -m unittest discover -s tests -v
```

Official PyTorch installation selector: https://pytorch.org/get-started/locally/
Do not use the CPU install command for a CUDA training environment. Dependencies are broad compatibility bounds, not an assertion that every CUDA/PyTorch combination has been tested. See `VALIDATION.md` for what actually ran here. No model downloads are needed for the default package.

## 2. Execute the smoke pipeline

Run commands from this directory. Included `data/smoke` and `runs/smoke` already contain the validated CPU DEBUG run; the following uses fresh paths so it cannot overwrite that evidence.

```bash
python -c 'import json; c=json.load(open("configs/smoke.json")); c.update(data_root="data/my_smoke",output_root="runs/my_smoke"); json.dump(c,open("configs/my_smoke.json","w"),indent=2)'
python -m gs_meeting.cli make-data --config configs/my_smoke.json
python -m gs_meeting.cli audit-data --config configs/my_smoke.json
python -m gs_meeting.cli suite --config configs/my_smoke.json --device auto --nfe 4 --samples 2 --split val --extra-controls --max-hours 0.5
```

This is 4/2/2 train/validation/test objects, 16 anchors, 16 px and 12 updates. Its only purpose is to check the entire pipeline. **Do not show its numbers as research results.** The included run evaluated the test split solely as a code-path check with an unchanged fixed configuration; do not use those test numbers for tuning. New pilot runs must select settings on validation.

## 3. Profile before the pilot

`configs/pilot.json` starts at 64 px, 128 anchors, 64/16/32 objects, 8 held-out cameras, 3 input views, 2 light directions, a small transformer and 2,000 optimizer updates. These are planning defaults, not a convergence claim. The reference renderer is dense and cannot be scaled like a CUDA rasterizer.

```bash
python -m gs_meeting.cli make-data --config configs/pilot.json
python -m gs_meeting.cli audit-data --config configs/pilot.json
python -m gs_meeting.cli train --config configs/profile.json --variant A0 --device auto --max-hours 0.25
```

`profile.json` uses the pilot data and writes 200 debug updates to `runs/profile`. Read `training.csv`/`manifest.json` to estimate per-update time. Profile A2 and B2 too, since learned rollouts and renderer backward work cost more than A0. For B2 profiling pass `--coarse runs/profile/A0_seed0/checkpoint.pt`. Freeze the chosen coarse checkpoint throughout a comparison; changing it invalidates a resumed B run. Do not count the profile as a converged coarse reconstructor.

Estimate the total work from measured seconds/update × updates × variants × seeds, plus evaluations. Set a practical first-pass cap such as 8 GPU-hours. Reduce common resolution/anchors/training updates before running the controlled suite if necessary; change these consistently across variants. The dense renderer needs machine profiling even on an RTX 6000 Ada.

## 4. Train the pilot and evaluate validation first

```bash
python -m gs_meeting.cli suite --config configs/pilot.json --device auto --nfe 4 8 16 --samples 8 --split val --extra-controls --max-hours 8
# Continue a stopped/paused suite without repeating completed checkpoints/evaluations:
python -m gs_meeting.cli suite --config configs/pilot.json --device auto --nfe 4 8 16 --samples 8 --split val --extra-controls --max-hours 8 --resume
```

Suite priority is A0 → A2 → B0 → B2 → A1 → B1 → regularizer controls, then evaluation. If the cap interrupts training, it stops before dependent comparisons. It resumes at a checkpoint with optimizer and Python/NumPy/PyTorch RNG state. The time cap is checked between updates/evaluation jobs, so a job can slightly overrun; use the machine's scheduler for a strict deadline. A fresh invocation gives a fresh time allowance. The cap includes this invocation's training and evaluation, excludes data preparation, and must not be presented as exact GPU accounting.

For an immediate early meeting result, run/evaluate A0 and A2 individually before queueing B; a full suite evaluates after training its selected models. Example:

```bash
python -m gs_meeting.cli train --config configs/pilot.json --variant A0 --device auto
python -m gs_meeting.cli train --config configs/pilot.json --variant A2 --device auto
python -m gs_meeting.cli evaluate --config configs/pilot.json --checkpoint runs/pilot/A2_seed0/checkpoint.pt --split val --nfe 8 --samples 8
```

Existing run directories are protected. Use `--resume` for the same experiment or a different `output_root` for a changed configuration. Resume only permits changing total steps/checkpoint frequency. A resumed `--steps` override must also be reflected in the saved configuration used for later work.

Exact CPU resume equivalence is tested at completed checkpoint boundaries. An interrupt in the middle of an update can consume random numbers from the abandoned update; resume remains usable but need not reproduce an uninterrupted run bit for bit. GPU kernel reproducibility also depends on the installed backend.

After validation chooses sigma/anchor weight/NFE, lock them before test evaluation. For a changed hyperparameter use a new config/output root, and use the same frozen coarse source across the compared B variants. Small suggested validation grids: sigma ∈ {0.02, 0.05, 0.1}; anchor weight ∈ {0.1, 1, 10}. Do not launch a full Cartesian grid on limited compute. Use the source checkpoint corresponding to the relevant training seed.

```bash
python -m gs_meeting.cli evaluate --config configs/pilot.json --checkpoint runs/pilot/B2_seed0/checkpoint.pt --coarse runs/pilot/A0_seed0/checkpoint.pt --split test --nfe 8 --samples 8
python -m gs_meeting.cli evaluate --config configs/pilot.json --checkpoint runs/pilot/A1_seed0/checkpoint.pt --variant B3 --coarse runs/pilot/A0_seed0/checkpoint.pt --split test --nfe 8 --samples 8
python -m gs_meeting.cli summarize --root runs/pilot --split test
```

Evaluate all matched controls using the same split/NFE/K, then replicate the shortlisted proposal/control pair with seeds 1 and 2. `suite --seed 1` trains its own A0 source for that replicate. Do not merge training and test objects or treat training seeds as extra independent objects.

## 5. Files produced

- Per-run `checkpoint.pt`: model, optimizer, normalizer provenance, dataset hash, frozen coarse hash, RNG state and update index. Load only checkpoints you trust: the optimizer/RNG format uses Python pickle.
- `manifest.json`: exact config, hardware, train IDs, dataset SHA-256, seed, wall/GPU time, parameter counts, peak training VRAM and backward-render view count. Preparation and frozen A0 costs must be reported separately. Each backward-render view is one camera forward whose graph participates in a loss backward, not a count of CUDA kernels.
- `training.csv`: all loss components and gradient norms. Different objectives' scalar losses are not directly comparable.
- Per-evaluation `results_per_object.csv`, `evaluation.json`, generated state NPZs, fixed first-sample qualitative panels for the same first 6 identities, and a relighting turntable GIF.
- Aggregated `MEETING_RESULTS_val.md` / `MEETING_RESULTS_test.md`, paired bootstrap CSVs, training curves and quality/latency plot.
- Parameter/latency/NFE accounting includes the explicit frozen coarse reconstructor where used. Latency includes deterministic pooling, model(s), decoding and one render; it excludes disk loading and full turntable generation. GPU calls synchronize before/after measurement.

Metrics implemented: linear-RGB PSNR/SSIM, full-image LPIPS when explicitly enabled, visible/disoccluded RGB error using prediction-independent teacher visibility, anchor-position/normal/material errors, rendered foreground depth/material MAE, held-out-light RGB error, and empirical 50/80/95% pixel depth/material coverage, interval width and ensemble CRPS. Anchor error is **not** mesh surface error. Depth is opacity-weighted expected camera depth, not first-surface ray intersection. LPIPS is blank by default; `--lpips` requires compatible `lpips`/`torchvision` and cached pretrained weights and can otherwise fail explicitly. The default report cannot claim the earlier disoccluded-LPIPS screening threshold is met. No best-of-K selection is used.

## Scientific details and limitations

Read `METHODS.md` for the exact path/sampler/anchor definitions and the correction to the previous prompt. This fixed-path SI is deliberately modest: a shared trunk has velocity and Gaussian-noise heads; the score is recovered with the proper gamma scaling. It does not establish a new interpolant family or a posterior theorem. The numerical sampler has finite-step error.

The reference representation is isotropic 3D Gaussians with position, radius, approximate normal, opacity and intrinsic albedo/roughness/metallic. It uses **one-sided normal culling as a surface proxy**, unlike ordinary two-sided 3DGS. This prevents back-surface anchors from leaking through a sparse translucent object into every input view. The decoder enforces bounds. The perspective footprint and GGX directional-light shading are simplified, without shadows, indirect lighting, anisotropic covariance or a learned LGAA. Procedural targets are sampled from this decoder's restricted family, so the teacher reconstruction ceiling is intentionally exact. This is a pipeline feasibility aid with a substantial same-renderer bias. There is no meaningful claim of recovering an arbitrary external asset from these runs. The data audit checks camera orthonormality, finite arrays, identity splits and nonempty disocclusion masks before a pilot suite.

Code map: `scene.py` decoder/renderer; `data.py` split and observation contract; `model.py` joint core; `methods.py` objectives and samplers; `engine.py` training/resume; `evaluate.py` metrics/qualitatives; `report.py` tables and bootstrap; `cli.py` commands. `DATA_CONTRACT.md` defines how your agent connects real data. `AGENT_HANDOFF.md` gives its next work order.
