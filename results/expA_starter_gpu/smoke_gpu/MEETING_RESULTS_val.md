# Meeting results: val

**DEBUG / PIPELINE CHECK ONLY**

These are actual local run measurements. Synthetic Gaussian-family targets use the same approximate renderer as inference. No claim about real assets or a pretrained LGAA is supported.

Means below average per-object, sample-mean metrics. No best-of-K selection is used. Missing metrics remain blank.

| Variant | Seed | NFE | K | Objects | PSNR dB | SSIM | LPIPS | Disoccluded RGB MAE | Relit RGB MAE | Pipeline ms median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0 | 1 | 2 | 2 | 37.2086 | 0.8509 | — | 0.0150 | 0.0064 | 5.3059 |
| A1 | 0 | 4 | 2 | 2 | 32.5626 | 0.7730 | — | 0.0179 | 0.0131 | 5.9105 |
| A2 | 0 | 4 | 2 | 2 | 35.4827 | 0.8111 | — | 0.0155 | 0.0077 | 5.7656 |
| B0_anchor | 0 | 4 | 2 | 2 | 36.6211 | 0.8305 | — | 0.0143 | 0.0071 | 5.9821 |
| B0 | 0 | 4 | 2 | 2 | 36.6192 | 0.8304 | — | 0.0143 | 0.0071 | 4.8152 |
| B1 | 0 | 4 | 2 | 2 | 36.9011 | 0.8393 | — | 0.0146 | 0.0069 | 4.7177 |
| B2 | 0 | 4 | 2 | 2 | 36.9024 | 0.8393 | — | 0.0146 | 0.0069 | 5.9235 |
| B3_anchor | 0 | 4 | 2 | 2 | 35.8801 | 0.8228 | — | 0.0133 | 0.0069 | 6.2771 |
| B3 | 0 | 4 | 2 | 2 | 35.6133 | 0.8342 | — | 0.0127 | 0.0066 | 4.8781 |

## Paired B2 differences

95% percentile bootstrap across object identities, within each training seed and matched NFE/K. This is not a substitute for training-seed replication. Negative error differences favor B2.

| Control | Seed | NFE | Metric | Mean B2 − control | 95% interval | Paired objects |
|---|---:|---:|---|---:|---|---:|
| B0 | 0 | 4 | disoccluded_rgb_mae | 0.00038 | [0.00026, 0.00051] | 2 |
| B0 | 0 | 4 | visible_psnr_db | 0.21298 | [-0.08365, 0.50962] | 2 |
| B0 | 0 | 4 | heldout_light_rgb_mae | -0.00027 | [-0.00074, 0.00020] | 2 |
| B0 | 0 | 4 | depth_mae | 0.00839 | [-0.00702, 0.02381] | 2 |
| B0 | 0 | 4 | latency_median_ms | 1.10836 | [0.04174, 2.17497] | 2 |
| B1 | 0 | 4 | disoccluded_rgb_mae | -0.00000 | [-0.00000, 0.00000] | 2 |
| B1 | 0 | 4 | visible_psnr_db | -0.00003 | [-0.00025, 0.00018] | 2 |
| B1 | 0 | 4 | heldout_light_rgb_mae | -0.00000 | [-0.00000, -0.00000] | 2 |
| B1 | 0 | 4 | depth_mae | -0.00003 | [-0.00006, -0.00000] | 2 |
| B1 | 0 | 4 | latency_median_ms | 1.20579 | [0.11201, 2.29957] | 2 |
| B3 | 0 | 4 | disoccluded_rgb_mae | 0.00199 | [0.00189, 0.00209] | 2 |
| B3 | 0 | 4 | visible_psnr_db | 0.99439 | [0.98315, 1.00564] | 2 |
| B3 | 0 | 4 | heldout_light_rgb_mae | 0.00029 | [-0.00050, 0.00108] | 2 |
| B3 | 0 | 4 | depth_mae | -0.01021 | [-0.03189, 0.01148] | 2 |
| B3 | 0 | 4 | latency_median_ms | 1.04546 | [-0.03845, 2.12938] | 2 |
| B0_anchor | 0 | 4 | disoccluded_rgb_mae | 0.00038 | [0.00026, 0.00051] | 2 |
| B0_anchor | 0 | 4 | visible_psnr_db | 0.21253 | [-0.08413, 0.50920] | 2 |
| B0_anchor | 0 | 4 | heldout_light_rgb_mae | -0.00026 | [-0.00073, 0.00021] | 2 |
| B0_anchor | 0 | 4 | depth_mae | 0.00840 | [-0.00703, 0.02384] | 2 |
| B0_anchor | 0 | 4 | latency_median_ms | -0.05859 | [-0.28115, 0.16396] | 2 |
| B3_anchor | 0 | 4 | disoccluded_rgb_mae | 0.00137 | [0.00081, 0.00192] | 2 |
| B3_anchor | 0 | 4 | visible_psnr_db | 1.26435 | [0.58972, 1.93898] | 2 |
| B3_anchor | 0 | 4 | heldout_light_rgb_mae | -0.00003 | [-0.00028, 0.00022] | 2 |
| B3_anchor | 0 | 4 | depth_mae | -0.04419 | [-0.12006, 0.03169] | 2 |
| B3_anchor | 0 | 4 | latency_median_ms | -0.35353 | [-0.53587, -0.17119] | 2 |
| A2 | 0 | 4 | disoccluded_rgb_mae | -0.00081 | [-0.00171, 0.00010] | 2 |
| A2 | 0 | 4 | visible_psnr_db | 1.45877 | [0.60727, 2.31028] | 2 |
| A2 | 0 | 4 | heldout_light_rgb_mae | -0.00085 | [-0.00209, 0.00038] | 2 |
| A2 | 0 | 4 | depth_mae | -0.12814 | [-0.37189, 0.11560] | 2 |
| A2 | 0 | 4 | latency_median_ms | 0.15791 | [0.11953, 0.19629] | 2 |

## Training and pending work

| Run | Updates | Parameters allocated | Parameters with gradient | Wall hours | GPU hours | Peak VRAM GB | Backward-render views | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 seed 0 | 12 | 26714 | 26285 | 0.00048 | 0.0005 | 0.0186 | 36 | completed |
| A1 seed 0 | 12 | 26714 | 26285 | 0.00031 | 0.0003 | 0.0189 | 36 | completed |
| A2 seed 0 | 12 | 26714 | 26285 | 0.00028 | 0.0003 | 0.0189 | 36 | completed |
| B0_anchor seed 0 | 12 | 26714 | 26285 | 0.00038 | 0.0004 | 0.0197 | 72 | completed |
| B0 seed 0 | 12 | 26714 | 26285 | 0.00037 | 0.0004 | 0.0197 | 72 | completed |
| B1 seed 0 | 12 | 26714 | 26714 | 0.00039 | 0.0004 | 0.0197 | 72 | completed |
| B2 seed 0 | 12 | 26714 | 26714 | 0.00038 | 0.0004 | 0.0197 | 72 | completed |
| B3_anchor seed 0 | 12 | 26714 | 26285 | 0.00038 | 0.0004 | 0.0197 | 72 | completed |

Anchor activity diagnostics:

- B0_anchor seed 0: anchor penalty positive on 58.3% of logged updates. If zero throughout, this run did not exercise an active anchor; inspect tolerance/path length on validation before interpreting it.
- B2 seed 0: anchor penalty positive on 58.3% of logged updates. If zero throughout, this run did not exercise an active anchor; inspect tolerance/path length on validation before interpreting it.
- B3_anchor seed 0: anchor penalty positive on 75.0% of logged updates. If zero throughout, this run did not exercise an active anchor; inspect tolerance/path length on validation before interpreting it.

Pending evaluated variants: none at this split/NFE inventory; check matched rows above.
Training seeds represented: [0].

## Interpretation limits

- Smoke configurations validate code execution only. A short loss curve or synthetic overfit does not establish the hypothesis.
- B2 anchors learned rollouts. A penalty on the fixed paired bridge alone would have zero parameter gradient; that earlier formulation was corrected here.
- B0_anchor and B3_anchor test whether the rendering penalty helps competing paths equally. Keep them pending explicitly if unfinished.
- B3 reuses A1's checkpoint and uses a coarse-start DDIM sampler. It is a SDEdit-style warm-start control, not a faithful SDEdit implementation.
- Pixelwise ensemble intervals describe empirical coverage on this dataset, not a proven multimodal posterior.
- The original disoccluded-LPIPS screening margin is NOT tested by disoccluded RGB MAE. Add a validated region perceptual protocol before claiming it is met.
- Test-set tables never choose hyperparameters. Use the validation split for sigma/anchor-weight selection first.
- A publication claim needs nonprocedural data, the project's real decoder/renderer, geometry surface metrics and multiple training seeds.
