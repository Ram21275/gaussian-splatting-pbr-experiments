# Experiment 0 results (val split)

**Status:** measured on procedural factorial objects. The frozen extractors are Marigold v1-1 (normals, depth, IID-appearance) with 1-step inference. These are not real-asset results.

## 0. Extractor sanity (vs GT, 128 px)

| metric | value |
|---|---:|
| normals_angular_err_deg_mean | 26.0699 |
| normals_angular_err_deg_median | 19.3554 |
| depth_absrel_affine_aligned | 0.0406 |
| albedo_mae_all | 0.3120 |
| roughness_mae_all | 0.1379 |
| metallic_mae_all | 0.4294 |
| albedo_mae_part | 0.3060 |
| roughness_mae_part | 0.1504 |
| metallic_mae_part | 0.4403 |
| albedo_mae_partition | 0.3181 |
| roughness_mae_partition | 0.1255 |
| metallic_mae_partition | 0.4186 |
| albedo_mae_meanbaseline | 0.1983 |
| roughness_mae_meanbaseline | 0.1572 |
| metallic_mae_meanbaseline | 0.4246 |
## 3. Object retrieval (R@1). `raw` is natural, with no fitted map; `adapter` is ridge fitted on train (learned alignment)

| layer | query→candidates | R@1 all (chance) | R@1 same-shape (chance) |
|---|---|---|---|
| up0 | raw_normals_to_iid | 0.365 (0.010) | 0.542 (0.333) |
| up0 | raw_normals_to_depth | 0.531 (0.010) | 0.708 (0.333) |
| up0 | raw_iid_to_iid_selfcheck | 1.000 (0.010) | 1.000 (0.333) |
| up0 | adapter_normals_to_iid | 1.000 (0.010) | 1.000 (0.333) |
| up1 | raw_normals_to_iid | 0.771 (0.010) | 0.833 (0.333) |
| up1 | raw_normals_to_depth | 0.927 (0.010) | 0.948 (0.333) |
| up1 | raw_iid_to_iid_selfcheck | 1.000 (0.010) | 1.000 (0.333) |
| up1 | adapter_normals_to_iid | 1.000 (0.010) | 1.000 (0.333) |
| up2 | raw_normals_to_iid | 0.365 (0.010) | 0.521 (0.333) |
| up2 | raw_normals_to_depth | 0.594 (0.010) | 0.719 (0.333) |
| up2 | raw_iid_to_iid_selfcheck | 1.000 (0.010) | 1.000 (0.333) |
| up2 | adapter_normals_to_iid | 1.000 (0.010) | 1.000 (0.333) |

## 4. Cross-prediction probes (ridge on PCA-64 patch features; held-out R²)

| probe | pairing | all | part regime | partition regime |
|---|---|---:|---:|---:|
| G->mat_of_G_object | true | 0.138 | 0.049 | 0.212 |
| M->geo_of_M_object | true | 0.705 | 0.734 | 0.676 |
| G->geo_of_G_object | true | 0.727 | 0.748 | 0.705 |
| M->mat_of_M_object | true | 0.267 | 0.193 | 0.327 |
| G->mat_of_G_object | same_shape | 0.138 | 0.049 | 0.212 |
| M->geo_of_M_object | same_shape | 0.705 | 0.734 | 0.676 |
| G->geo_of_G_object | same_shape | 0.727 | 0.748 | 0.705 |
| M->mat_of_M_object | same_shape | 0.267 | 0.193 | 0.327 |
| G->mat_of_G_object | any | 0.138 | 0.050 | 0.208 |
| M->geo_of_M_object | any | 0.700 | 0.733 | 0.666 |
| G->geo_of_G_object | any | 0.721 | 0.745 | 0.695 |
| M->mat_of_M_object | any | 0.271 | 0.209 | 0.317 |

### Adjacent-patch affinity (logistic on |f_a − f_b|; AUC, 0.5 = chance)

| probe | part regime | partition regime | positive rate |
|---|---:|---:|---:|
| G->same_matid | 0.871 | 0.752 | 0.871 |
| G->same_part | 0.884 | 0.833 | 0.890 |
| M->same_matid | 0.871 | 0.759 | 0.871 |
| M->same_part | 0.886 | 0.832 | 0.890 |

## 5. Bottlenecks at matched latent k (mean ± sd over seeds, regime = all)

| input | pairing | k | kind | zeroed | normal° | depth | albedo | rough | metal | params |
|---|---|---:|---|---|---|---|---|---|---|---:|
| gt_oracle |     true
Name: 0, dtype: object |     8
Name: 0, dtype: object |     separate
Name: 0, dtype: object |     
Name: 0, dtype: object | 4.3400±nan | 0.0030±nan | 0.0292±nan | 0.0668±nan | 0.0080±nan | 272913 |
| gt_oracle |     true
Name: 1, dtype: object |     8
Name: 1, dtype: object |     shared
Name: 1, dtype: object |     
Name: 1, dtype: object | 4.3585±nan | 0.0033±nan | 0.0539±nan | 0.1047±nan | 0.1262±nan | 208913 |
| gt_oracle |     true
Name: 2, dtype: object |     8
Name: 2, dtype: object |     shared_private
Name: 2, dtype: object |     
Name: 2, dtype: object | 4.2977±nan | 0.0048±nan | 0.0407±nan | 0.0566±nan | 0.0261±nan | 342289 |
| gt_oracle |     true
Name: 3, dtype: object |     8
Name: 3, dtype: object |     shared_private
Name: 3, dtype: object |     private
Name: 3, dtype: object | 45.2782±nan | 1.5791±nan | 0.2424±nan | 0.1465±nan | 0.3564±nan | 342289 |
| gt_oracle |     true
Name: 4, dtype: object |     8
Name: 4, dtype: object |     shared_private
Name: 4, dtype: object |     shared
Name: 4, dtype: object | 38.9430±nan | 0.4862±nan | 0.0957±nan | 0.0743±nan | 0.0478±nan | 342289 |
| marigold |     any
Name: 5, dtype: object |     8
Name: 5, dtype: object |     separate
Name: 5, dtype: object |     
Name: 5, dtype: object | 12.7159±nan | 0.0706±nan | 0.0777±nan | 0.0830±nan | 0.0943±nan | 303377 |
| marigold |     any
Name: 6, dtype: object |     8
Name: 6, dtype: object |     shared
Name: 6, dtype: object |     
Name: 6, dtype: object | 13.1850±nan | 0.0648±nan | 0.1318±nan | 0.1284±nan | 0.2209±nan | 239377 |
| marigold |     any
Name: 7, dtype: object |     8
Name: 7, dtype: object |     shared_private
Name: 7, dtype: object |     
Name: 7, dtype: object | 11.9633±nan | 0.0626±nan | 0.0966±nan | 0.0990±nan | 0.1216±nan | 403217 |
| marigold |     any
Name: 8, dtype: object |     8
Name: 8, dtype: object |     shared_private
Name: 8, dtype: object |     private
Name: 8, dtype: object | 28.4466±nan | 1.0556±nan | 0.1844±nan | 0.1523±nan | 0.3354±nan | 403217 |
| marigold |     any
Name: 9, dtype: object |     8
Name: 9, dtype: object |     shared_private
Name: 9, dtype: object |     shared
Name: 9, dtype: object | 38.0098±nan | 1.0411±nan | 0.1749±nan | 0.3063±nan | 0.3540±nan | 403217 |
| marigold |     same_shape
Name: 10, dtype: object |     8
Name: 10, dtype: object |     separate
Name: 10, dtype: object |     
Name: 10, dtype: object | 12.9129±nan | 0.0719±nan | 0.0878±nan | 0.0871±nan | 0.1188±nan | 303377 |
| marigold |     same_shape
Name: 11, dtype: object |     8
Name: 11, dtype: object |     shared
Name: 11, dtype: object |     
Name: 11, dtype: object | 11.3138±nan | 0.0614±nan | 0.1392±nan | 0.1286±nan | 0.2269±nan | 239377 |
| marigold |     same_shape
Name: 12, dtype: object |     8
Name: 12, dtype: object |     shared_private
Name: 12, dtype: object |     
Name: 12, dtype: object | 10.7204±nan | 0.0601±nan | 0.1044±nan | 0.1036±nan | 0.1407±nan | 403217 |
| marigold |     same_shape
Name: 13, dtype: object |     8
Name: 13, dtype: object |     shared_private
Name: 13, dtype: object |     private
Name: 13, dtype: object | 24.1869±nan | 0.9749±nan | 0.2034±nan | 0.1555±nan | 0.3348±nan | 403217 |
| marigold |     same_shape
Name: 14, dtype: object |     8
Name: 14, dtype: object |     shared_private
Name: 14, dtype: object |     shared
Name: 14, dtype: object | 35.9265±nan | 1.0763±nan | 0.1655±nan | 0.2569±nan | 0.3676±nan | 403217 |
| marigold |     true
Name: 15, dtype: object |     8
Name: 15, dtype: object |     separate
Name: 15, dtype: object |     
Name: 15, dtype: object | 12.9129±nan | 0.0719±nan | 0.0897±nan | 0.0879±nan | 0.1182±nan | 303377 |
| marigold |     true
Name: 16, dtype: object |     8
Name: 16, dtype: object |     shared
Name: 16, dtype: object |     
Name: 16, dtype: object | 11.3512±nan | 0.0616±nan | 0.1371±nan | 0.1256±nan | 0.2170±nan | 239377 |
| marigold |     true
Name: 17, dtype: object |     8
Name: 17, dtype: object |     shared_private
Name: 17, dtype: object |     
Name: 17, dtype: object | 10.8909±nan | 0.0606±nan | 0.1024±nan | 0.1001±nan | 0.1366±nan | 403217 |
| marigold |     true
Name: 18, dtype: object |     8
Name: 18, dtype: object |     shared_private
Name: 18, dtype: object |     private
Name: 18, dtype: object | 23.9892±nan | 1.0979±nan | 0.2168±nan | 0.1570±nan | 0.3325±nan | 403217 |
| marigold |     true
Name: 19, dtype: object |     8
Name: 19, dtype: object |     shared_private
Name: 19, dtype: object |     shared
Name: 19, dtype: object | 36.3899±nan | 0.9982±nan | 0.1559±nan | 0.2321±nan | 0.3508±nan | 403217 |

### Shared-minus-separate composite gap: true vs shuffled pairing (Marigold, regime = all)

| pairing | k | Δgeo (shared/sep − 1) | Δmat |
|---|---:|---:|---:|
| true | 8 | -0.132 | +0.598 |
| same_shape | 8 | -0.135 | +0.657 |
| any | 8 | -0.023 | +0.862 |

## 6. Locked decision rule → Experiment A representation

```
{
 "k_star": 8,
 "shared_vs_separate": {
  "geo": -0.1321252634814719,
  "mat": 0.5976418475757175
 },
 "sharedprivate_vs_separate": {
  "geo": -0.15648222922893662,
  "mat": 0.14503860782685218
 },
 "private_zeroing_cost": {
  "geo": 9.15635949716068,
  "mat": 1.0422766096008838
 },
 "choice": "separate"
}
```
