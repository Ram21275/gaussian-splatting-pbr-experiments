# Experiment 0: do geometry and PBR features share a representation space?

**Status:** completed on procedural factorial objects (data v2), 2026-11-03.
- The decision split is val. Test was evaluated once, after the rule was locked.
- Plan, locked rule and the dated amendment are in `experiments/meeting/EXP0_PLAN.md`.
- The v1 run is void and archived as `*_v1_crossed.*`: it had only 3 training material palettes.

**Scope**
- 80/16/32 train/val/test shapes, with 3 unique material draws each, in two regimes:
  - part-aligned: one material per primitive part;
  - partition: random-plane materials that cut across parts.
- 3 posed views at 512 px.
- Frozen extractors on identical RGB: Marigold v1-1 normals (geometry), depth (geometry–geometry reference) and IID-appearance (PBR), run with 1-step inference.
- Features are UNet up-block activations on a 16×16 patch grid.
- GT maps are used only as targets.

## Key numbers (val | test)

| quantity | val | test |
|---|---|---|
| **Extractor sanity** | | |
| Marigold normal error, mean (deg) | 25.3 | 23.1 |
| IID vs GT Pearson: albedo / roughness / metallic | 0.47 / 0.40 / 0.06 | 0.53 / 0.38 / 0.04 |
| **Natural alignment (no adapter)** | | |
| CKA up2, normals~IID: true / same-shape shuffle / any shuffle | 0.497 / 0.384 / 0.028 | 0.494 / 0.388 / 0.024 |
| CKA up2, normals~depth (reference) | 0.501 / 0.413 / 0.034 | 0.506 / 0.425 / 0.028 |
| Retrieval up1, raw normals→IID, R@1: all (chance) | 0.81 (0.010) | 0.83 (0.005) |
| Retrieval up1, raw normals→IID, R@1: same shape (chance 0.33) | 0.83 | 0.85 |
| Retrieval up1, ridge adapter, R@1: all / same shape | 0.78 / 0.80 | 0.80 / 0.80 |
| **Entanglement** | | |
| Probe: geometry features → material of paired object, R² (true / same-shape / any) | 0.23 / −0.04 / −0.04 | 0.22 / −0.02 / −0.04 |
| Probe: PBR features → own geometry, R² | 0.68 | 0.67 |
| Material-driven variance fraction, up2: normals / IID | 0.18 / 0.28 | 0.16 / 0.28 |
| **Matched-capacity bottleneck, k = 8, Marigold features, true pairs** | | |
| separate: normal° / albedo / roughness | 12.85 / 0.095 / 0.167 | 12.49 / 0.105 / 0.165 |
| shared: normal° / albedo / roughness | 12.90 / 0.086 / 0.156 | 12.55 / 0.094 / 0.152 |
| shared+private: normal° / albedo / roughness | 12.85 / 0.097 / 0.169 | 12.68 / 0.098 / 0.175 |
| mean predictor: normal° / albedo / roughness | 38.25 / 0.182 / 0.181 | 37.18 / 0.177 / 0.177 |
| shared vs separate composite, geo / mat (true pairs) | −1.8% / −8.2% | −0.3% / −9.2% |
| same, same-shape shuffle | −2.6% / −4.4% | −1.7% / −5.8% |
| same, any-object shuffle | +10.3% / +2.4% | +8.9% / +0.1% |
| **Locked rule → Experiment A representation** | **shared** | **shared** (confirmed) |

## What this establishes
1. Frozen geometry and PBR features of the same images are **mutually informative and object-specific**.
   - Alignment collapses under any-object shuffles and drops under same-shape material swaps.
   - Geometry features carry object-specific material information, i.e. appearance leaks into geometry features.
   - PBR features are geometry-dominated.
2. At matched latent size, **one shared code preserves geometry and material as well as separate codes**. On materials it does slightly better, but only with true pairing, so the gain depends on correspondence.
3. Experiment A keeps a **single joint latent**.

## What it does not establish (open)
- **That the two feature sets are the same space.**
  - CKA is invariant to rotation and scaling, so it does not show one coordinate system.
  - Both models see the same image and are fine-tuned from the same SD2 checkpoint. The common-cause nulls (randomly initialised UNet, base SD2 UNet) are **not yet run**.
  - Model-stitching tests are **not yet run**.
- **The shared bottleneck's advantage is confounded.**
  - Its encoder reads both inputs, while each separate encoder reads only one. A "separate codes, both inputs" baseline is **not yet run**.
  - Encoders are not parameter-matched: about 402k separate, 338k shared and 600k shared+private at k = 8.
- **Extractor limitations.** Metallic is not identifiable from these renders (Pearson ≈ 0.05), and Marigold-IID shows a domain gap on isolated objects.
- **Data scope.** Procedural objects and a single model family (Stable Diffusion based) only. No real-data claim is made.

## Figures (`figures_val/`, `figures_test/`)
| file | content |
|---|---|
| fig1_examples_gt_vs_extractors | same shape, 4 material draws: GT vs Marigold normal/albedo/roughness, material ids |
| fig2_feature_pca_patches | top-3 PCA of normals vs IID features, same shape/view across material draws |
| fig3_material_variance_fraction | how much each extractor's features move with material only |
| fig4_cka_up1 / fig4_cka_up2 | natural alignment with shuffled controls and 95% CIs |
| fig5_retrieval | raw vs adapter retrieval, all vs same-shape candidates |
| fig6_cross_probes | cross-modal probes under true/same-shape/any pairing |
| fig7_bottleneck_tradeoff | error vs k for separate/shared/shared+private, Marigold and GT-oracle inputs |
| fig8_shared_gap_vs_pairing | shared-vs-separate gap by pairing control, with the +5% rule line |
| fig9_bottleneck_reconstructions | decoded patch maps, separate vs shared at k = 8 |

## Rerun
```bash
./py -m gsm.exp0_extract --split {train,val,test}        # or experiments/meeting/exp0_extract.sh
./py -m gsm.exp0_analyze --train_split train --eval_split {val,test}
./py -m gsm.exp0_report  --split {val,test}
./py -m gsm.exp0_figures --split {val,test}
```
