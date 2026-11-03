# Implementation prompt: two Gaussian-splatting experiments for a research meeting

You are implementing and running a small, reproducible research study. Deliver real measurements and visual results for the two experiment families below. Proceed with repository inspection, environment setup, implementation, meaningful checks and available GPU runs. Do not stop after making a plan. If the GPU, weights or data are unavailable, finish the implementation, smoke checks and exact launch commands, and explicitly mark GPU results as pending. Never invent results or silently substitute a different task.

## Research scope and interpretation

Input is 3 posed RGB views initially, with 5 views as a robustness evaluation if supported. Output is a coherent Gaussian asset containing geometry, opacity and intrinsic PBR materials. Novel RGB views come from a physically based renderer. Use opaque, static objects and known synthetic lighting first. There must be one trainable generative trunk and one sampling loop per model, not separate geometry and material generators. Multiple deterministic output heads are allowed.

The first question is architectural feasibility: can a single conditional rectified-flow core supply a DreamLifting-style joint Gaussian/PBR decoder? The second question tests Hypothesis 3: does a rendering-anchored stochastic interpolant improve coarse-to-complete refinement compared with equally strong alternatives?

These experiments do not establish global novelty, full DreamLifting reproduction or publication readiness. A successful RGB-only test establishes only RGB/geometry feasibility. PBR claims require intrinsic-material targets and held-out-light evaluation.

## 0. Inspect the repository and establish an executable baseline

1. Read AGENTS.md and existing experiment logs. Inspect the current finalized model and reuse working encoders, rendering, data and evaluation paths. Create an isolated branch/worktree if appropriate. Preserve existing experiment outputs.
2. Create a dedicated environment using repository-compatible versions. Record Python, PyTorch, CUDA, renderer version, GPU, precision, dependency lockfile and git commit. Do not replace a working renderer merely to use a newer package.
3. Audit https://github.com/ZX-Yin/DreamLifting for actual code, weights and training data. The repository was publicly showing a work-in-progress release when this plan was prepared. Verify its current status. Record what can genuinely be reused.
4. Prefer a genuine LGAA adapter/decoder experiment only when compatible implementation and weights are available. Frozen diffusion features and rectified-flow features are not interchangeable just because their tensors have the same shape. Match feature interfaces and fine-tune explicitly.
5. If DreamLifting cannot be reproduced, implement the controlled fallback below and label every figure/table **DreamLifting-inspired feasibility prototype**. Reuse accessible components from DiffSplat or the existing project only when compatible. A frozen image encoder plus a small new decoder is not the original LGAA. Do not spend the meeting budget reconstructing an unavailable foundation model.

Primary references:
- DreamLifting: https://arxiv.org/abs/2509.07435
- DiffSplat: https://arxiv.org/abs/2501.16764 and https://github.com/chenguolin/DiffSplat
- SplatFlow: https://arxiv.org/abs/2411.16443 (the multiview rectified-flow paper, not the similarly named driving method)
- Stochastic interpolants: https://arxiv.org/abs/2303.08797
- Data-dependent couplings: https://arxiv.org/abs/2310.03725
- SDEdit, an important coarse-start control: https://arxiv.org/abs/2108.01073

## Shared data, state and decoder

Use the smallest existing licensed synthetic multiview PBR dataset available locally, with RGB, depth/normals, albedo, roughness, metallic, camera matrices and lighting. If none exists, prepare an asymmetric procedural-object pilot with known material parameters and at least two evaluation lights. Use neutral physically meaningful materials, not arbitrary random image textures masquerading as albedo. Keep object identities disjoint. Suggested first pilot: 64 training, 16 validation and 32 test objects. Record the actual counts and limitations. Use 8 training objects only for a separate debugging overfit test; never present that as held-out generalization.

Use a fixed input-camera set and 8 held-out target cameras per object where available. Also reserve unseen environment maps/light directions. Keep scene normalization, backgrounds, camera conventions, BRDF, exposure handling and evaluation masks identical across models. Do not use held-out target RGB/depth to construct the inference-time source, confidence or conditions. Full-view teachers are permitted for training objects, and must be labeled pseudo-ground truth on real data.

Start at 128 px, 3 input views, batch 1 with accumulation and 1,024 anchors. Increase to 256 px or 2,048 anchors only after profiling. Cache frozen view features. Use a small set/transformer trunk, targeting approximately 5M trainable parameters, local attention where needed and camera-aware cross-attention. Report actual parameter counts.

Use one deterministic decoder D shared and frozen across the controlled comparisons once its reconstruction quality is established. Its outputs include positions, valid positive covariance coordinates, opacity, albedo, roughness and metallic. A 2DGS implementation needs an appropriate surface/normal parameterization; a 3DGS fallback must be labeled accordingly. Do not change the representation across variants. RGB is rendered from the physical state and known lighting, rather than generated independently with baked-in shadows.

Use aligned Gaussian anchors or a persistent canonical latent. Independently fitted splat files have arbitrary order and cannot be paired token by token. Construct parent/anchor correspondence from a deterministic shared representation. Standardize geometry and material channel scales using training data only. Demonstrate the decoder reconstruction ceiling before judging a generator. Include decoder and feature-cache preparation costs separately.

## Experiment A: single-flow feasibility in a joint geometry/PBR architecture

Implement these three variants with the same conditioner, decoder, training identities, losses and comparable generative capacity:

- **A0 Direct:** deterministic residual/reconstruction head. This establishes whether iterative generation adds value.
- **A1 Diffusion:** one conditional noise-prediction or clean-state-prediction diffusion core for the joint latent.
- **A2 Flow:** one conditional rectified-flow vector field v_theta(z,t,c). Draw z0 from N(0,I), z1 from the aligned target latent, sample t uniformly, form zt=(1-t)z0+t*z1 and train against z1-z0.

For A2 use L_FM = E[||v_theta(zt,t,c)-(z1-z0)||^2], with declared channel normalization. When adding rendering/depth/material supervision, apply it to an explicitly defined clean endpoint estimate, not raw zt as if it were clean. Use the analogous auxiliary targets and schedule in A1. For example, a forward-time endpoint estimate z1_hat=zt+(1-t)*v_theta may be used as a training surrogate, but state that it is approximate for a learned marginal field. Run a basic latent-loss-only control if the renderer term dominates.

For a fair meeting comparison use matching initialization where architectures permit, equal optimizer updates and the same main training budget. Record any checkpoint/pretraining advantage. Do not compare a large pretrained diffusion model with a tiny scratch flow and attribute differences solely to the objective.

Evaluate A1/A2 at 4, 8 and 16 actual network evaluations, with a 1/2-step diagnostic only if affordable. Count Heun's two evaluations per step and classifier-free guidance evaluations correctly. Count renderer/backward guidance work separately. Profile end-to-end latency, including conditioning/decoding, as well as cached-head latency. Synchronize GPU timings and report median/p95 over repeated runs after warm-up.

Required meeting evidence:
1. One held-out table: A0/A1/A2, PSNR/SSIM/LPIPS, surface/depth error, albedo error, roughness/metallic error, held-out-light RGB error, total/trainable parameters, peak VRAM, training GPU-hours and inference latency.
2. A quality-versus-latency/NFE plot and training curves.
3. The same 6 held-out objects/cameras/lights visualized for all variants, including failures. Include normal/depth/albedo/roughness maps and relighting, not only attractive RGB renders.
4. A claim supported by the actual scope: e.g. a single flow can drive this prototype's joint decoder with a measured quality/cost tradeoff. Do not claim a faithful DreamLifting weight swap unless it really was performed.

## Experiment B: Hypothesis 3, rendering-anchored stochastic interpolant refinement

Reuse Experiment A's data, conditioner, decoder and trunk size. Keep Gaussian count fixed. Do not also add jump events or Hypothesis 1's anisotropic covariance, because they would confound this test.

Build X0 solely from the posed input views using the fixed coarse reconstructor, plus small source jitter calibrated on validation. X1 is the aligned training target. Freeze this source construction across all B variants.

Implement:
- **B0 Coarse residual flow:** X_t=(1-t)X0+t*X1, target velocity X1-X0, deterministic ODE sampling. This is the most important control for initialization and learned refinement.
- **B1 Stochastic interpolant:** X_t=(1-t)X0+t*X1+gamma(t)*xi, xi~N(0,I), gamma(t)=sigma*sin(pi*t). Train a velocity head against X1-X0+gamma'(t)*xi and a score head on the same trunk. At interior times, the paired Gaussian-kernel score target is -xi/gamma(t). Avoid singular endpoints with a declared time cutoff and schedule.
- **B2 Anchored interpolant:** B1 plus a soft rendering-stability loss on sampled intermediate states relative to the coarse input-view fit. Use L_anchor=E_t[max(0,E_obs(D(X_t))-E_obs(D(X0))-tau(t))]. Define E_obs and its RGB/depth weights. Begin with tau=0 as a diagnostic, then select a small tolerance/weight on validation. Never assume intermediate latent states are valid surfaces simply because the decoder emits valid covariance values.

For B1/B2, the matched state-independent-noise sampler is dX=[b_theta+a(t)*s_theta]dt+sqrt(2*a(t))*dW. Implement the score correction. Arbitrarily adding noise to B0 is not this sampler. If you also report ODE sampling, distinguish it from the SDE. Include the score head or an equal-capacity unused head in matched controls and count separate evaluations if you do not share computation.

Keep source jitter, total expected noise energy, sampling evaluations and rendering-gradient budget matched where applicable. Sweep sigma only over a small declared validation grid, such as {0.02,0.05,0.1} in standardized latent units. Sweep the anchor weight over at most three values. Do not tune on test objects. Compare the shortlisted B2 with A2 and a coarse-start A1/SDEdit-style baseline using the same coarse source, so the benefit cannot be attributed only to a warm start or a new renderer loss. If that control cannot finish, explicitly mark the hypothesis conclusion incomplete.

Evaluate 8 samples per held-out condition where affordable. Report sample-mean error and diversity under input consistency; use fixed seed sets across models. Never silently report best-of-8. Separate visible, disoccluded and invalid pixels using input geometry and independently defined ground-truth visibility. Include floaters/thin-structure failure examples. For uncertainty report 50/80/95% depth/material interval coverage plus interval widths or CRPS. Do not call Gaussian primitive covariance posterior uncertainty. A single target per exact condition cannot by itself establish a multimodal conditional posterior; include synthetic ambiguous-view examples or restrict the claim to refinement/calibration on the available evaluation protocol.

The hypothesis is supported only if B2 improves the completion/quality/cost tradeoff over B0, B1 and the coarse-start diffusion control without unacceptable visible-region damage. Report paired per-object differences and bootstrap intervals. Provisional meeting margins: at least 5% relative disoccluded LPIPS improvement, visible PSNR lower confidence bound above -0.2 dB, and latency no more than 1.5x the matched stochastic baseline. These are screening targets, not universal significance standards. Lock them after validation. Report negative results honestly.

## Execution order and compute discipline

First verify rendering, camera conventions, covariance validity, endpoint schedules, finite loss/gradients and correct score signs. Then overfit 8 objects as a debugging check. Profile 200 training updates and estimate the remaining cost. Run the 6 main A/B variants with one seed at the pilot resolution, then replicate only the most important baseline/proposal pair with two additional training seeds. Reuse finished runs instead of retraining the same baseline.

Use an initial 8 GPU-hour total cap for the first meeting pass, excluding any already-existing preprocessing, and record preparation cost separately. Scale updates by profiling so the strongest comparisons complete within that cap. This is a planning cap, not a promise of complete PBR convergence. If insufficient, finish A0/A2 and B0/B2 first, preserve A1/B1/coarse-start controls as explicit pending runs, and state that the hypothesis is preliminary. On Colab save resumable checkpoints often. If no GPU is attached, build/test and provide a queue; do not claim training was completed.

## Required repository outputs

Create isolated configs and runnable commands for every variant. Use existing project conventions or add:
- `experiments/meeting/README.md`: exact data preparation, training, resuming and evaluation commands.
- `experiments/meeting/PLAN.md`: scope, model-count definition, frozen/trainable modules, splits, budget and claim boundaries.
- A per-run manifest with config, commit, dataset object IDs, seeds, hardware, wall time, NFE, renderer-gradient count and output paths.
- `outputs/meeting/results_per_object.csv`, aggregated tables with units and paired differences, plots and fixed qualitative panels.
- `outputs/meeting/MEETING_RESULTS.md`: what completed, key numbers, uncertainty intervals, failures, pending controls and defensible conclusions.
- Checkpoints and at least one novel-view/relighting turntable for each shortlisted model.

Finish by giving me the actual results and exact rerun commands. Keep completed measurements, debugging results and planned runs visibly separate. Do not implement an unrelated new architecture or all three hypotheses at once.
