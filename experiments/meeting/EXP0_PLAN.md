# Experiment 0: do geometry and PBR features share a representation space, or can they?

Written 2026-11-02, **before** any Experiment 0 validation result was inspected. The decision rule in section 6 is locked after validation and before the test split is evaluated.

## Question
1. **Does (natural alignment):** do frozen geometry and PBR features, extracted from the *same* posed multiview RGB, already share structure without any learned adapter?
2. **Can (learned sharing):** at matched latent and decoder capacity, can one compact shared bottleneck preserve both geometry and materials as well as separate bottlenecks? Does a shared-plus-private split do better?

Single-generator feasibility, which is Experiment A, is treated as an existing architectural direction, not as our novelty claim. Stochastic-interpolant experiments (B) stay queued.

## 1. Data: factorial procedural objects (`gsm/scene.py`)
- Geometry depends only on `shape_id`. Materials depend only on `(material_seed, regime)`. Shape × material is fully crossed, so the same material seed gives the same palette and partition on every shape.
- **Regime `part`:** one material per primitive part. Material boundaries coincide with geometric part boundaries, a correlation built in by the generator.
- **Regime `partition`:** 4 materials assigned by 2 random planes that cut across parts. Material layout is independent of part structure. This control separates "the features share structure" from "our generator coupled them".
- **Splits:**
  - train: 80 shapes × material seeds {0,1,2}
  - val: 16 held-out shapes × held-out seeds {100,101,102}
  - test: 32 held-out shapes × held-out seeds {200,201,202}
  - Both regimes are used in every split, giving 768 objects × 3 views.
- **Rendering:** 3 fixed posed input cameras, 512 px, train light L0, black background. Local GGX shading with no cast shadows. GT normal, depth, albedo, roughness, metallic, part id and material id are rendered by an analytic SDF ray-marcher and used **only as targets**.

## 2. Frozen extractors (same RGB into each)
| role | model | output |
|---|---|---|
| geometry | prs-eth/marigold-normals-v1-1 | camera-space normals |
| geometry positive control | prs-eth/marigold-depth-v1-1 | affine-invariant depth |
| PBR | prs-eth/marigold-iid-appearance-v1-1 | albedo, roughness, metallicity |

- **Features:** UNet `up_blocks[0..2]` outputs, average-pooled to a 16×16 patch grid per view. The hook runs at the single denoising step (1-step v1-1 inference, fixed seed). Analysis uses foreground patches (mask > 0.5) only.
- **Caveat:** all three models are fine-tuned from the same Stable Diffusion 2 checkpoint, so some raw similarity can come from that shared starting point. The shuffled-pair and same-shape controls are what separate correspondence from this.
- **Reference input:** GT maps used as "oracle features", to separate data-level sharing from extractor-level sharing.

## 3. Analyses
0. **Extractor sanity on held-out objects:** normal angular error (convention measured, not assumed); albedo, roughness and metallic error versus GT.
1. **Factor-partitioned variance:** at fixed (shape, view, patch), the variance across material seeds is material-driven. The material-driven fraction of total variance is reported per extractor and layer. If geometry features vary with material, that is shading or appearance leakage.
2. **Natural alignment, no adapter:**
   - Linear CKA between same-layer normals and IID patch features, comparing true pairs with shuffled pairs (patches from other objects).
   - References: CKA(normals, depth) as geometry–geometry, and CKA of each model with the GT geometry and GT material targets.
   - Object-level raw-cosine retrieval from geometry to IID across all test objects, and among same-shape candidates with a different material, where chance is 1/3 per regime.
3. **Learned alignment (adapter, labelled as such):** a ridge or CCA map fit on train only, with the same retrieval protocol on held-out data.
4. **Bottlenecks:**
   - Patch-level MLP autoencoders: inputs are PCA-64 features fit on train; targets are GT geometry (camera normal, depth) and GT material (albedo, roughness, metallic).
   - **Separate:** E_g(G)→k/2, E_m(M)→k/2.
   - **Shared:** E([G,M])→k, with both decoders reading the full k.
   - **Shared+private:** k_s = k/2, k_g = k_m = k/4.
   - Encoder and decoder hidden widths are identical, and parameter counts are reported. k ∈ {4, 8, 16, 32}, with 3 seeds.
   - Every model is also trained on **shuffled pairs**: M features from a different object, with targets following their source. The real-minus-shuffled change in the shared-vs-separate gap is the evidence of exploited correspondence.
   - For shared+private, the shared code and the private codes are each zeroed at test time.
5. **Probes (linear and small MLP):**
   - Geometry features → material values.
   - IID features → normals.
   - Same-material affinity of adjacent patches from geometry features, and same-part affinity from IID features.
   - Each probe is reported per regime against a mean or chance baseline and a shuffled control.

## 4. What does and does not count as evidence
- Similarity scores alone do not establish a common embedding space.
- "Naturally aligned" means raw features with no fitted map. Anything that uses a fitted map is "adapter-learned alignment".
- A positive result in the `part` regime alone could be generator coupling. Claims of shared *structure* need the `partition` regime too.

## 5. Budget
Extraction is about 2,300 images × 3 one-step models; analysis is minutes per model. Target is under 1 GPU-hour total, logged separately as preparation cost.

## 6. Locked decision rule for Experiment A's representation (fixed before validation)
Use the val split, averaging both regimes and 3 seeds, at the smallest k where separate bottlenecks are within 10% of their k = 32 error.
- **Shared** (one joint latent, as currently in A): pick this if the shared bottleneck's geometry error and material error are each ≤ 5% relatively worse than separate at matched k.
- **Shared+private:** pick this if shared is > 5% worse on either modality, shared+private is within 5% of separate on both, and zeroing its private code raises that modality's error by > 10%.
- **Separate per-modality latent channels** inside the single generator (separate token channel groups, one trunk): pick this otherwise.

The test split is evaluated once, after this choice, with the same rule reported.

### 6a. Exact error composites, fixed before any result was seen
- **Geometry error** = mean of (normal angular error in deg, depth MAE). Each is divided by the same quantity for the *separate* bottleneck at k = 32 (same input source, same pairing, mean over seeds).
- **Material error** = mean of (albedo MAE, roughness MAE, metallic MAE), normalised the same way.
- **Inputs:** Marigold PCA-64 features at layer `up1`, true pairing, both regimes pooled, mean over 3 seeds.
- **k\*** = the smallest k with mean(separate geometry, separate material) ≤ 1.10.
- At k\*, the "≤ 5% relatively worse" test compares shared/separate composite ratios.
- The private-code ablation compares shared+private with private zeroed against shared+private intact.

## Amendment 1 (2026-11-03). Made after the v1 validation run; the test split has never been evaluated
**Why.** In the v1 validation run, every bottleneck's material error with Marigold inputs was at or above the mean predictor: albedo about 0.215 vs 0.188, roughness about 0.156 vs 0.151. Its "shared" verdict was decided on geometry alone. **That verdict is void and is not a result.** v1 files are archived as `*_v1_crossed.*`.

**Diagnosis (train data only, no val or test).**
- **Cause:** v1 crossed only 3 material seeds over all 80 training shapes, so training saw 3 palettes. Validation used unseen palettes.
- **Check:** within the v1 training split, I fit on shapes 0–59 and evaluated on shapes 60–79, which share those 3 palettes.
- **Result:** IID features reach clearly positive material R²: 0.55 at up2/PCA-256, with albedo about 0.70. So the features carry material information, and v1 failed on palette coverage.
- **Side finding:** the same check showed that PCA-64 at up1 discards much of that signal. At up1 with PCA-64, IID gets 0.19.

**Changes (fixed before any v2 result)**
1. **Data v2:** every object has its own material seed, 3 draws per shape. Seed ranges are disjoint per split: 480 / 96 / 192 distinct palettes. Shape × material crossing is dropped. The material-variance, same-shape-shuffle and same-shape-retrieval analyses need only several draws per shape, which v2 keeps.
2. **Bottleneck and probe inputs:** layer `up2`, PCA-256 per modality, fit on train, for both extractors identically. This was chosen on the train-only diagnostic above. CKA and retrieval still report all layers.
3. **Probes:** cross-object probes are added: geometry features → material of the M-side object, and M features → geometry of the G-side object. In v1 every probe read targets from its own feature object, so the shuffle controls were inert.
4. **Decision rule precondition:**
   - The rule in §6 applies only if, for each modality, the separate bottleneck at k = 32 beats the mean predictor on that modality's composite by ≥ 10%.
   - Otherwise the outcome is "not decidable".
   - The rule itself is unchanged.
5. **Material composite = albedo + roughness.** Metallic is reported separately. Measured extractor sanity on v1 validation: metallic Pearson 0.08. Metals under two directional lights with low ambient render as dark diffuse surfaces, so metallic is not identifiable from these inputs. MAE on a near-binary target also rewards predicting the majority class.

**Known non-blocking imbalances (reported, not corrected)**
- **Encoders are not parameter-matched:** at k=8, separate has about 303k, shared 239k and shared+private 403k.
- Decoders are matched.
- This disadvantages shared and flatters shared+private.

## Amendment 2 (2026-11-03). Controls round, pre-registered before any of these analyses ran
This round answers the open confounds from the v2 run. Stitching runs only after this round is reported. CCA, independent-model replication and generator training stay queued.

**Splits and uncertainty**
- Splits are grouped by shape: train, val and test share no shape, and all material draws of a shape, in both regimes, stay in one split. A test asserts this.
- All intervals are a **shape-grouped bootstrap**: resample shapes with replacement, carrying every material variant and regime of each resampled shape. 2,000 replicates, 95% percentile intervals.
- Metrics are reported individually: normal error (deg), depth MAE, albedo MAE, roughness MAE and metallic MAE. Composites are secondary.

**(1) Bottleneck capacity control**
- **Variants:**
  - `separate_own`: E_g(G)→k/2 and E_m(M)→k/2, as in v2.
  - `separate_both`: E_g([G,M])→k/2 and E_m([G,M])→k/2. This is new: separate codes with both inputs.
  - `shared`: E([G,M])→k.
  - `shared_private`: E_s([G,M])→k/2, E_g(G)→k/4 and E_m(M)→k/4.
- **Matching:** every variant has the same latent total k and the same **total parameter count** within ±2%. Hidden widths are solved per variant, and decoders are identical.
- **Sweep:** k ∈ {4, 8, 16}, 3 seeds, the same up2 / PCA-256 inputs as v2.
- **Primary contrast:** shared − separate_both, per metric, with shape-bootstrap CIs on val. Test is evaluated once, after this section's interpretation is written.
- **Interpretation, fixed now:**
  - If the 95% CI of shared − separate_both includes 0, or lies within ±5% of separate_both, for every metric at k = 8: "a shared code is sufficient; the v2 advantage came from access to both inputs".
  - If the CI is negative beyond −5% on a metric: "a shared code helps on that metric beyond input access".
  - If positive beyond +5%: "sharing the code costs on that metric".

**(2) Alignment controls for the common cause (same image, same initialisation)**
- **Feature sources on the same 512 px images, same hooked blocks (`up_blocks[0..2]`), same 16×16 pooling:**
  - (a) Marigold normals, depth and IID, from v2.
  - (b) **Random-init** UNets: the normals architecture (8-channel input) and the IID architecture (12-channel input), seeds 0 and 1, fed exactly the Marigold pipeline inputs.
  - (c) The **untouched base SD2 UNet** (`sd2-community/stable-diffusion-2`, a community mirror of stabilityai/stable-diffusion-2; weight hashes recorded) on the VAE latent of the same image. It runs with the empty-prompt embedding, at t = 999 (matching Marigold's 1-step timestep) and at t = 261 (the conventional feature-extraction timestep), with fixed noise.
- **Reported:** CKA (true / same-shape / any-object pairing) and raw retrieval, with shape-bootstrap CIs, for:
  - normals~IID;
  - each Marigold model ~ base (does task fine-tuning keep the base space?);
  - random~random and random~Marigold (architecture-and-input floor).
- **Reading, fixed now:**
  - normals~IID alignment counts as **"beyond common cause"** only if its CI lies above both the random-network floor and the matched base-model alignment for that layer.
  - Even then, the strongest wording is "linearly related, more than explained by shared input and initialisation". **No "same embedding space" conclusion is drawn from CKA or retrieval.**

**(3) Material-extractor accuracy and convention checks**
- On a fixed subset of 24 val objects (first 4 shapes × 3 draws × 2 regimes), measure IID albedo, roughness and metallic against GT for:
  - colour space: albedo treated as sRGB vs linear;
  - channel order: correlation matrix of predicted material channels vs GT roughness and metallic;
  - background: black vs mid-grey;
  - inference: 1 step vs 4 steps vs 4 steps × ensemble of 5;
  - processing resolution: 512 vs 768.
- **Reported:** MAE, Pearson and gain-aligned albedo MAE (evaluation-only alignment).
- If a different setting is clearly better, the main analysis is **not** silently changed. The finding is reported, and any re-extraction gets its own amendment.

**Queued after this round:** bidirectional stitching with a UNet skip-connection audit, then CCA, independent-model replication and generator training.
