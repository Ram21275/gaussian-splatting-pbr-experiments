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

## (2) Alignment controls
*(Filled in once the analysis completes. See `controls/CONTROLS_val.md`.)*

**Audit:** the SD2 VAE and the Marigold VAE have identical weights (hash `356f0647e091d2d7`). Every model receives the identical image latent, so a shared input encoding is a confirmed common-cause pathway.
