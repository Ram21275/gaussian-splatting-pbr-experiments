# Dataset adapter contract

The default data requires no external downloads. For the real experiment, the agent should reuse a licensed locally available multiview/PBR dataset and its renderer. Preserve this interface or implement an explicit adapter with equivalent checks.

## Directory and normalization

```
data/<dataset>/
  splits.json
  stats.npz
  dataset.json
  <object_id>.npz
```

`splits.json` is `{"train": ["id_a", ...], "val": [...], "test": [...]}`. Identity sets must be disjoint; different views or materials of the same source mesh should not leak across splits. `stats.npz` contains `mean` and `std`, both float32 shape [13], fitted on TRAIN identities only. Std must be positive (the procedural adapter floors it at 0.1). Root NPZ bytes and splits/statistics are SHA-256 fingerprinted; changing any one invalidates existing checkpoint provenance. `dataset.json` should record dataset name/license, object/mesh identities, source files, preprocessing version, camera conventions, rendering approximations, material units and teacher provenance.

All NPZ arrays are numeric float32 with `allow_pickle=False`. State-channel order is defined in `scene.CHANNELS`. Shapes below use N anchors, V input views (default 3), Q held-out target views (pilot 8), L light directions (default 2), H=W resolution.

| Key | Shape | Meaning / access |
|---|---|---|
| `state` | [N,13] | Teacher raw joint state before train-only standardization; TRAIN supervision / evaluation only |
| `input_rgb` | [V,H,W,3] | Observed linear RGB [0,1]; available at inference |
| `input_cameras` | [V,4,3] | Camera axes + eye, below; inference input |
| `input_lights` | [V,3] | Known world-space light directions for input images; inference input |
| `target_cameras` | [Q,4,3] | Held-out cameras, not conditioning features |
| `target_lights` | [L,3] | Light 0 matches input light; light 1 absent from input images |
| `target_rgb` | [L,Q,H,W,3] | Linear RGB reference images |
| `target_alpha` | [L,Q,H,W] | Reference accumulated opacity |
| `target_depth` | [L,Q,H,W,1] | Reference opacity-weighted camera-space depth |
| `target_albedo` | [L,Q,H,W,3] | Reference composited intrinsic albedo |
| `target_roughness` | [L,Q,H,W,1] | Reference composited roughness |
| `target_metallic` | [L,Q,H,W,1] | Reference composited metallic |
| `visible_fraction` | [Q,H,W] | Independent teacher-defined fraction visible from input camera set |

The reference renderer's camera matrix stores unit `right`, unit `up`, unit `forward` as rows 0–2, and eye position as row 3. A world point transforms as `(point-eye) @ axes.T`. Positive z is forward, image y points downward, vertical FOV is 45°, principal point is ((W-1)/2,(H-1)/2). Light vectors point from the surface toward a directional light. Units use an object centered within roughly unit radius and camera radius 3. All transforms must be applied consistently to positions, normals, cameras and lights.

## Replacing the representation

Raw teacher channels are **not physical values pasted together**. `scene.decode` maps bounded raw channels using tanh/sigmoid. External physical assets need an inverse mapping into this restricted representation, or a replacement decoder and matching channel schema. Arbitrarily ordered independently fitted Gaussians are not aligned training pairs. Build deterministic anchor/parent correspondence or use a persistent teacher latent. Never silently nearest-neighbor match using held-out target geometry at inference.

For an existing project decoder, replace `decode()` and `render()` consistently for targets, loss, predictions and evaluation; change model channel dimensions and normalization schema explicitly. Keep 2DGS versus 3DGS representation labels accurate. Replacing the dense renderer with gsplat/another CUDA implementation also changes footprint, sorting or visibility behavior: verify decoder ceiling and gradients before comparing old/new numbers. The provided code has no CUDA extension installation assumptions.

`conditions(d)` deliberately reads only `input_rgb`, `input_cameras`, `input_lights`. `coarse_prediction` uses those features, never teacher `state` or target RGB/depth. The target arrays may be loaded for evaluation, but are not passed to model conditioning. The leakage test perturbs every target field and checks conditioning is unchanged.

## Relighting and uncertainty interpretation

The default pilot supervises materials explicitly and holds out a light direction from the input view set. The same held-out light family is rendered for training-object targets but not used as conditioning or RGB training loss. This tests new illumination from the input-light observations, not unseen environment-map generalization. Add truly unseen light/environment distributions for a stronger claim.

If unknown lighting is estimated on real data, document the estimator and freeze it across comparators. RGB-only data supports RGB/geometry claims unless trustworthy intrinsic-material targets or a separately justified inverse-rendering protocol are supplied. Real-data full-view fitted teachers are pseudo-ground truth.

The visible/disoccluded masks must be defined from independent geometry/visibility, never the evaluated prediction. The reference renderer culls back-facing anchor normals as an explicit one-sided surface approximation. In the procedural adapter, an anchor is input-visible if its maximum input compositing weight exceeds 0.04; its target-view visible weight fraction defines a >=0.5 visible mask within alpha>0.1 foreground. This threshold/culling convention should be audited on the real renderer; it is not ordinary two-sided 3DGS. `audit-data` records the actual disoccluded fraction and the suite blocks a non-debug completion pilot with empty masks. Interval coverage requires multiple generated scenes; primitive spatial covariance is not posterior uncertainty.
