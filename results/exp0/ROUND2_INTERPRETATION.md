# Experiment 0, controls round (Amendment 2): interpretation

The validation reading below was written **before** the test split was evaluated for this round. Percentages are shared relative to separate_both, (A − B)/B. Intervals are 95% shape-grouped bootstrap (2,000 replicates, 16 val shapes). Each variant is the mean of 3 seeds, with total parameters matched within 0.2%.

## (1) Shared code vs separate codes with both inputs (val)

| k = 8 | normal° | depth | albedo | roughness | metallic* |
|---|---|---|---|---|---|
| shared − separate_both | +2.7% [+1.5, +3.8] | −0.4% [−1.8, +1.2] | −3.2% [−5.1, −1.2] | −5.4% [−7.8, −3.0] | +7.8% [+0.9, +15.8] |
| shared − separate_own (v2-style contrast) | −1.1% [−3.3, +1.1] | −5.5% [−7.6, −3.6] | −9.9% [−12.1, −7.5] | −1.6% [−3.5, +0.4] | +0.9% [−4.1, +6.7] |
| separate_both − separate_own (input-access effect) | −3.7% [−5.7, −1.7] | −5.1% [−7.3, −3.4] | −6.9% [−8.9, −4.9] | +4.0% [+0.9, +7.1] | −6.4% [−12.5, −0.4] |

\*Metallic is not recovered by the extractor (part 3), so it is not interpretable.

**Reading against the pre-registered rule at k = 8**
- **Within ±5%, i.e. "a shared code is sufficient":** normal (CI entirely inside +5%), depth (CI includes 0), albedo (CI inside ±5%).
- **Roughness: not covered by the rule.** The CI is [−7.8, −3.0]: it crosses −5% without lying entirely beyond it. The rule as written has no outcome for this case, so I am reporting it as **"inconclusive, leaning toward a small shared benefit"** and not forcing it into a category.
- **Metallic: also inconclusive** (CI [+0.9, +15.8] crosses +5%), and it is uninterpretable regardless.

**Conclusion (val)**
- Most of v2's apparent "shared beats separate" gain on albedo (−9.9%) and depth (−5.5%) came from the shared encoder seeing both inputs.
- With separate codes given the same inputs, the shared code is roughly **equivalent**: within about ±5% on geometry and albedo, with at most a small roughness benefit.
- A shared code is **sufficient**, but there is no evidence it is **better** than separate codes.

**Capacity sweep**
- At **k = 4**, shared reallocates capacity: normal error is −33% [−36, −29] lower, but albedo is +16% [+12, +19] higher than separate_both. This is the bottleneck's tradeoff, not evidence of a shared space.
- At **k = 16**, every geometry and albedo/roughness contrast is within ±2.3%.

**Shared+private ablation**
- Zeroing either the shared code or the private codes collapses geometry: at k = 8, normal error becomes 33° (private zeroed) or 35° (shared zeroed), against a 38° mean predictor.
- The model does not learn a clean split between shared and modality-specific information. This doesn't support or rule out a factorised representation.

## (3) Material-extractor checks (24 val objects; full table in `iid_checks/IID_CHECKS.md`)

**Conventions confirmed**
- Albedo is sRGB-encoded: treating it as linear gives MAE 0.50, against 0.32 for sRGB→linear.
- Material channel 0 is roughness: Pearson 0.54 with GT roughness, against ≤ 0.12 for the other channels.

**Metallic is not recovered in any setting.** Channel 1's correlation with GT metallic is between −0.24 and −0.10 across all settings, and the channel mean is the same on metal and dielectric surfaces (0.32 vs 0.35).

**Background is the largest domain-gap factor.** A mid-grey background improves albedo substantially:

| | Black (main) | Grey |
|---|---|---|
| Albedo MAE | 0.322 | 0.196 |
| Albedo Pearson | 0.44 | 0.73 |
| Gain-aligned albedo MAE | 0.102 | 0.080 |

Roughness is roughly unchanged (MAE 0.147 → 0.153). 4 steps, 5× ensembling and 768 px do not help.

**Implication.** Every Experiment 0 result so far uses the **black-background** setting, and its PBR features are therefore weaker than they could be. Re-extracting with a grey background is a candidate change. **Not applied:** per the pre-registered rule it needs its own amendment and a full re-run of the earlier analyses.

## (2) Alignment controls (val | test; full tables in `controls/CONTROLS_{val,test}.md`)

**Audit:** the SD2 VAE and the Marigold VAE have identical weights (hash `356f0647e091d2d7`). Every model receives the identical image latent, so a shared input encoding is a confirmed common-cause pathway.

**Linear CKA, true pairs** (shape-bootstrap 95% CI)

| pair | up1 val | up1 test | up2 val | up2 test |
|---|---|---|---|---|
| normals~IID (Marigold) | 0.762 | 0.756 | 0.497 | 0.494 |
| normals~base SD2 (t=261) | 0.697 | 0.698 | 0.551 | 0.538 |
| IID~base SD2 (t=261) | 0.719 | 0.718 | 0.628 | 0.631 |
| random-init N(seed0)~random-init I(seed1) | 0.688 | 0.675 | **0.823** | **0.807** |

**Pre-registered reading**
- At **up2**, normals~IID lies *below* the random-network floor and below both models' alignment to base SD2. It is **not beyond common cause**.
- At **up1**, it clears both floors only marginally, with interval edges touching: val CI lower bound 0.759 against an upper bound of 0.755.
- **CKA cannot separate trained-model alignment from what same-image input through a UNet produces.** A random UNet on the same image reaches CKA above 0.8, mostly through low-level image structure. The random networks' CKA collapses under same-shape material swaps (0.82 → 0.27 at up2), while the trained models' CKA falls much less (0.50 → 0.38).

**Raw retrieval at up1** (needs coordinate-level correspondence; R@1 among all objects)

| | val | test |
|---|---|---|
| normals→IID | 0.81 [0.75, 0.88] | 0.83 [0.79, 0.88] |
| normals→base261 | 0.56 [0.46, 0.65] | 0.60 [0.53, 0.66] |
| IID→base261 | 0.72 [0.64, 0.80] | 0.73 [0.67, 0.80] |
| random N→random I, normals→random | 0.00 | 0.00–0.005 |

**What this shows**
- Retrieval works only between networks from the same SD2 initialisation. Random networks score zero.
- The two task-tuned models retrieve each other at least as well as each retrieves the base. They keep the SD2 coordinate system and drift in correlated ways.
- This is consistent with **inherited, co-adapted coordinates**. It is **not** evidence of a common geometry/PBR embedding space, and the plan forbids drawing that conclusion from CKA or retrieval.

## Test-split confirmation of (1): run once, after the val reading above was written

| k = 8, test | normal° | depth | albedo | roughness | metallic* |
|---|---|---|---|---|---|
| shared − separate_both | +2.7% [+1.8, +3.6] | +1.6% [+0.2, +3.1] | −4.3% [−6.6, −2.4] | −6.2% [−7.7, −4.9] | −0.7% [−3.9, +2.8] |
| shared − separate_own | −0.2% [−2.0, +1.7] | −4.2% [−6.3, −2.2] | −11.9% [−14.5, −9.1] | −3.3% [−5.3, −1.4] | −8.8% [−13.6, −3.6] |
| separate_both − separate_own | −2.8% [−4.1, −1.4] | −5.7% [−7.1, −4.3] | −7.9% [−9.9, −6.0] | +3.1% [+1.3, +4.8] | −8.1% [−11.8, −4.5] |

Test reproduces val.
- **Geometry:** equivalent, within ±5% (normal +2.7%, depth +1.6%).
- **Materials:** a small shared-code benefit on albedo (−4.3%) and roughness (−6.2%), with CIs straddling the −5% line, so inconclusive under the rule but leaning toward a modest benefit.
- **The v1/v2 "shared wins" story was mostly input access** (separate_both − separate_own: −8% albedo, −6% depth).

## Overall status after the controls round
- **Supported:** at matched latent size and parameter count, a shared code is sufficient for geometry and albedo/roughness, with a possible small benefit on materials.
- **Not supported:** any claim that frozen geometry and PBR features share an embedding space. CKA is explained by common cause, and retrieval by the shared SD2 initialisation.
- **Next, per plan:** bidirectional stitching with a UNet skip-connection audit. A grey-background re-extraction is a candidate amendment.
