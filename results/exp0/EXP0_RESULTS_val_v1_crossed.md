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
| albedo_mae_gain_aligned_evalonly | 0.1151 |
| albedo_pearson | 0.4653 |
| roughness_pearson | 0.4577 |
| metallic_pearson | 0.0801 |
| albedo_mae_meanbaseline | 0.1983 |
| roughness_mae_meanbaseline | 0.1572 |
| metallic_mae_meanbaseline | 0.4246 |

## 1. Material-driven variance fraction (fixed shape, view and patch; vary material seed)

0 = invariant to material; 1 = all variance is material-driven.

| features | fraction |
|---|---:|
| normals/up0 | 0.112 |
| normals/up1 | 0.191 |
| normals/up2 | 0.188 |
| depth/up0 | 0.105 |
| depth/up1 | 0.197 |
| depth/up2 | 0.176 |
| iid/up0 | 0.107 |
| iid/up1 | 0.235 |
| iid/up2 | 0.292 |
| gt_geo | 0.000 |
| gt_mat | 0.784 |

## 2. Natural alignment: linear CKA, no adapter (object-bootstrap 95% CI)

### layer up0

| pair | true | same-shape shuffle | any shuffle |
|---|---|---|---|
| depth~gt_geo | 0.378 [0.355,0.397] | 0.378 [0.355,0.397] | 0.299 [0.270,0.320] |
| depth~gt_mat | 0.012 [0.011,0.021] | 0.011 [0.010,0.019] | 0.010 [0.010,0.019] |
| depth~iid | 0.840 [0.833,0.849] | 0.826 [0.820,0.836] | 0.717 [0.709,0.730] |
| gt_geo~gt_mat | 0.011 [0.006,0.020] | 0.010 [0.005,0.018] | 0.010 [0.006,0.017] |
| iid~gt_geo | 0.388 [0.367,0.402] | 0.388 [0.367,0.402] | 0.326 [0.293,0.351] |
| iid~gt_mat | 0.015 [0.013,0.026] | 0.013 [0.010,0.021] | 0.012 [0.011,0.023] |
| normals~depth | 0.811 [0.808,0.819] | 0.790 [0.787,0.797] | 0.651 [0.642,0.666] |
| normals~gt_geo | 0.362 [0.343,0.379] | 0.362 [0.343,0.379] | 0.299 [0.270,0.322] |
| normals~gt_mat | 0.015 [0.015,0.025] | 0.012 [0.012,0.023] | 0.012 [0.011,0.023] |
| normals~iid | 0.808 [0.805,0.816] | 0.796 [0.791,0.804] | 0.701 [0.693,0.713] |

### layer up1

| pair | true | same-shape shuffle | any shuffle |
|---|---|---|---|
| depth~gt_geo | 0.380 [0.360,0.391] | 0.380 [0.360,0.391] | 0.201 [0.175,0.220] |
| depth~gt_mat | 0.027 [0.026,0.043] | 0.017 [0.019,0.032] | 0.016 [0.016,0.029] |
| depth~iid | 0.778 [0.776,0.790] | 0.667 [0.657,0.695] | 0.197 [0.202,0.232] |
| gt_geo~gt_mat | 0.011 [0.006,0.020] | 0.010 [0.005,0.018] | 0.010 [0.006,0.017] |
| iid~gt_geo | 0.326 [0.309,0.336] | 0.326 [0.309,0.336] | 0.178 [0.155,0.196] |
| iid~gt_mat | 0.037 [0.035,0.057] | 0.027 [0.028,0.042] | 0.019 [0.016,0.034] |
| normals~depth | 0.817 [0.816,0.825] | 0.707 [0.696,0.735] | 0.226 [0.230,0.262] |
| normals~gt_geo | 0.387 [0.367,0.399] | 0.387 [0.367,0.399] | 0.214 [0.188,0.229] |
| normals~gt_mat | 0.023 [0.023,0.042] | 0.017 [0.019,0.031] | 0.015 [0.015,0.029] |
| normals~iid | 0.761 [0.759,0.774] | 0.652 [0.642,0.682] | 0.205 [0.213,0.237] |

### layer up2

| pair | true | same-shape shuffle | any shuffle |
|---|---|---|---|
| depth~gt_geo | 0.202 [0.193,0.210] | 0.202 [0.193,0.210] | 0.071 [0.061,0.084] |
| depth~gt_mat | 0.018 [0.014,0.032] | 0.012 [0.012,0.021] | 0.009 [0.007,0.023] |
| depth~iid | 0.636 [0.621,0.657] | 0.508 [0.489,0.534] | 0.036 [0.034,0.052] |
| gt_geo~gt_mat | 0.011 [0.006,0.020] | 0.010 [0.005,0.018] | 0.010 [0.006,0.017] |
| iid~gt_geo | 0.115 [0.109,0.124] | 0.115 [0.109,0.124] | 0.048 [0.042,0.057] |
| iid~gt_mat | 0.076 [0.070,0.099] | 0.041 [0.036,0.063] | 0.016 [0.013,0.027] |
| normals~depth | 0.517 [0.507,0.534] | 0.414 [0.397,0.443] | 0.035 [0.035,0.047] |
| normals~gt_geo | 0.363 [0.334,0.388] | 0.363 [0.334,0.388] | 0.083 [0.073,0.095] |
| normals~gt_mat | 0.012 [0.012,0.022] | 0.008 [0.009,0.015] | 0.007 [0.006,0.014] |
| normals~iid | 0.518 [0.505,0.537] | 0.378 [0.361,0.405] | 0.030 [0.031,0.043] |

## 3. Object retrieval (R@1). `raw` is natural, with no fitted map; `adapter` is ridge fitted on train (learned alignment)

| layer | query→candidates | R@1 all (chance) | R@1 same-shape (chance) |
|---|---|---|---|
| up0 | raw_normals_to_iid | 0.365 (0.010) | 0.542 (0.333) |
| up0 | raw_normals_to_depth | 0.531 (0.010) | 0.708 (0.333) |
| up0 | raw_iid_to_iid_selfcheck | 1.000 (0.010) | 1.000 (0.333) |
| up0 | adapter_normals_to_iid | 0.448 (0.010) | 0.552 (0.333) |
| up1 | raw_normals_to_iid | 0.771 (0.010) | 0.833 (0.333) |
| up1 | raw_normals_to_depth | 0.927 (0.010) | 0.948 (0.333) |
| up1 | raw_iid_to_iid_selfcheck | 1.000 (0.010) | 1.000 (0.333) |
| up1 | adapter_normals_to_iid | 0.792 (0.010) | 0.802 (0.333) |
| up2 | raw_normals_to_iid | 0.365 (0.010) | 0.521 (0.333) |
| up2 | raw_normals_to_depth | 0.594 (0.010) | 0.719 (0.333) |
| up2 | raw_iid_to_iid_selfcheck | 1.000 (0.010) | 1.000 (0.333) |
| up2 | adapter_normals_to_iid | 0.458 (0.010) | 0.719 (0.333) |

## 4. Cross-prediction probes (ridge on PCA-64 patch features; held-out R²)

| probe | pairing | all | part regime | partition regime |
|---|---|---:|---:|---:|
| G->mat_of_G_object | true | -0.052 | -0.080 | -0.075 |
| M->geo_of_M_object | true | 0.660 | 0.690 | 0.630 |
| G->geo_of_G_object | true | 0.688 | 0.711 | 0.665 |
| M->mat_of_M_object | true | -0.012 | 0.051 | -0.161 |
| G->mat_of_G_object | same_shape | -0.052 | -0.080 | -0.075 |
| M->geo_of_M_object | same_shape | 0.660 | 0.690 | 0.630 |
| G->geo_of_G_object | same_shape | 0.688 | 0.711 | 0.665 |
| M->mat_of_M_object | same_shape | -0.012 | 0.051 | -0.161 |
| G->mat_of_G_object | any | -0.044 | -0.073 | -0.074 |
| M->geo_of_M_object | any | 0.650 | 0.682 | 0.618 |
| G->geo_of_G_object | any | 0.681 | 0.708 | 0.654 |
| M->mat_of_M_object | any | -0.017 | 0.066 | -0.199 |

### Adjacent-patch affinity (logistic on |f_a − f_b|; AUC, 0.5 = chance)

| probe | part regime | partition regime | positive rate |
|---|---:|---:|---:|
| G->same_matid | 0.865 | 0.727 | 0.871 |
| G->same_part | 0.868 | 0.809 | 0.890 |
| M->same_matid | 0.872 | 0.742 | 0.871 |
| M->same_part | 0.886 | 0.822 | 0.890 |

## 5. Bottlenecks at matched latent k (mean ± sd over seeds, regime = all)

| input | pairing | k | kind | zeroed | normal° | depth | albedo | rough | metal | params |
|---|---|---:|---|---|---|---|---|---|---|---:|
| gt_oracle |     true
Name: 0, dtype: object |     4
Name: 0, dtype: object |     separate
Name: 0, dtype: object |     
Name: 0, dtype: object | 13.5638±0.1754 | 0.0007±0.0001 | 0.1062±0.0206 | 0.0897±0.0170 | 0.0205±0.0012 | 270861 |
| gt_oracle |     true
Name: 1, dtype: object |     4
Name: 1, dtype: object |     shared
Name: 1, dtype: object |     
Name: 1, dtype: object | 0.7564±0.0878 | 0.0033±0.0008 | 0.0997±0.0047 | 0.1161±0.0023 | 0.0281±0.0047 | 205837 |
| gt_oracle |     true
Name: 2, dtype: object |     4
Name: 2, dtype: object |     shared_private
Name: 2, dtype: object |     
Name: 2, dtype: object | 0.5580±0.0626 | 0.0020±0.0010 | 0.1089±0.0060 | 0.1111±0.0032 | 0.0214±0.0032 | 339725 |
| gt_oracle |     true
Name: 3, dtype: object |     4
Name: 3, dtype: object |     shared_private
Name: 3, dtype: object |     private
Name: 3, dtype: object | 55.9262±28.0414 | 0.6782±0.7202 | 0.2583±0.0603 | 0.1609±0.0129 | 0.3138±0.0030 | 339725 |
| gt_oracle |     true
Name: 4, dtype: object |     4
Name: 4, dtype: object |     shared_private
Name: 4, dtype: object |     shared
Name: 4, dtype: object | 61.8463±25.7459 | 0.9105±0.5170 | 0.1101±0.0070 | 0.0963±0.0073 | 0.0320±0.0088 | 339725 |
| gt_oracle |     true
Name: 5, dtype: object |     8
Name: 5, dtype: object |     separate
Name: 5, dtype: object |     
Name: 5, dtype: object | 0.4203±0.0646 | 0.0005±0.0000 | 0.0489±0.0010 | 0.0027±0.0015 | 0.0048±0.0018 | 272913 |
| gt_oracle |     true
Name: 6, dtype: object |     8
Name: 6, dtype: object |     shared
Name: 6, dtype: object |     
Name: 6, dtype: object | 0.9205±0.1633 | 0.0032±0.0010 | 0.0714±0.0265 | 0.0037±0.0008 | 0.0068±0.0034 | 208913 |
| gt_oracle |     true
Name: 7, dtype: object |     8
Name: 7, dtype: object |     shared_private
Name: 7, dtype: object |     
Name: 7, dtype: object | 0.5202±0.0566 | 0.0011±0.0003 | 0.1033±0.0136 | 0.0049±0.0013 | 0.0045±0.0012 | 342289 |
| gt_oracle |     true
Name: 8, dtype: object |     8
Name: 8, dtype: object |     shared_private
Name: 8, dtype: object |     private
Name: 8, dtype: object | 57.5013±6.6048 | 1.4138±0.2861 | 0.1965±0.0146 | 0.0516±0.0160 | 0.3543±0.0328 | 342289 |
| gt_oracle |     true
Name: 9, dtype: object |     8
Name: 9, dtype: object |     shared_private
Name: 9, dtype: object |     shared
Name: 9, dtype: object | 51.7954±19.7772 | 0.3193±0.1164 | 0.1157±0.0175 | 0.2158±0.0739 | 0.0245±0.0231 | 342289 |
| gt_oracle |     true
Name: 10, dtype: object |     16
Name: 10, dtype: object |     separate
Name: 10, dtype: object |     
Name: 10, dtype: object | 0.3127±0.0523 | 0.0004±0.0001 | 0.0054±0.0010 | 0.0021±0.0004 | 0.0062±0.0018 | 277017 |
| gt_oracle |     true
Name: 11, dtype: object |     16
Name: 11, dtype: object |     shared
Name: 11, dtype: object |     
Name: 11, dtype: object | 0.6910±0.0480 | 0.0028±0.0004 | 0.0268±0.0141 | 0.0019±0.0004 | 0.0070±0.0019 | 215065 |
| gt_oracle |     true
Name: 12, dtype: object |     16
Name: 12, dtype: object |     shared_private
Name: 12, dtype: object |     
Name: 12, dtype: object | 0.4766±0.1304 | 0.0009±0.0002 | 0.0076±0.0003 | 0.0022±0.0005 | 0.0067±0.0022 | 347417 |
| gt_oracle |     true
Name: 13, dtype: object |     16
Name: 13, dtype: object |     shared_private
Name: 13, dtype: object |     private
Name: 13, dtype: object | 33.3507±9.8951 | 1.6020±0.2460 | 0.1916±0.0109 | 0.1586±0.0108 | 0.3402±0.0169 | 347417 |
| gt_oracle |     true
Name: 14, dtype: object |     16
Name: 14, dtype: object |     shared_private
Name: 14, dtype: object |     shared
Name: 14, dtype: object | 29.1770±6.8141 | 0.3049±0.1793 | 0.0583±0.0133 | 0.0255±0.0104 | 0.0127±0.0027 | 347417 |
| gt_oracle |     true
Name: 15, dtype: object |     32
Name: 15, dtype: object |     separate
Name: 15, dtype: object |     
Name: 15, dtype: object | 0.3010±0.0451 | 0.0003±0.0001 | 0.0047±0.0014 | 0.0027±0.0013 | 0.0052±0.0009 | 285225 |
| gt_oracle |     true
Name: 16, dtype: object |     32
Name: 16, dtype: object |     shared
Name: 16, dtype: object |     
Name: 16, dtype: object | 0.5726±0.0420 | 0.0021±0.0006 | 0.0076±0.0017 | 0.0027±0.0007 | 0.0069±0.0010 | 227369 |
| gt_oracle |     true
Name: 17, dtype: object |     32
Name: 17, dtype: object |     shared_private
Name: 17, dtype: object |     
Name: 17, dtype: object | 0.3141±0.0450 | 0.0008±0.0002 | 0.0044±0.0005 | 0.0026±0.0003 | 0.0059±0.0014 | 357673 |
| gt_oracle |     true
Name: 18, dtype: object |     32
Name: 18, dtype: object |     shared_private
Name: 18, dtype: object |     private
Name: 18, dtype: object | 28.1263±3.2152 | 1.5439±0.0811 | 0.2117±0.0075 | 0.1602±0.0095 | 0.3492±0.0063 | 357673 |
| gt_oracle |     true
Name: 19, dtype: object |     32
Name: 19, dtype: object |     shared_private
Name: 19, dtype: object |     shared
Name: 19, dtype: object | 23.0144±11.1254 | 0.3458±0.0791 | 0.0107±0.0028 | 0.0289±0.0061 | 0.0183±0.0086 | 357673 |
| marigold |     any
Name: 20, dtype: object |     4
Name: 20, dtype: object |     separate
Name: 20, dtype: object |     
Name: 20, dtype: object | 20.7577±0.5385 | 0.1133±0.0011 | 0.2197±0.0013 | 0.1666±0.0020 | 0.3153±0.0095 | 301325 |
| marigold |     any
Name: 21, dtype: object |     4
Name: 21, dtype: object |     shared
Name: 21, dtype: object |     
Name: 21, dtype: object | 15.3347±0.0725 | 0.1151±0.0008 | 0.2288±0.0017 | 0.1683±0.0005 | 0.3317±0.0082 | 236301 |
| marigold |     any
Name: 22, dtype: object |     4
Name: 22, dtype: object |     shared_private
Name: 22, dtype: object |     
Name: 22, dtype: object | 16.4019±0.2439 | 0.1159±0.0009 | 0.2262±0.0008 | 0.1654±0.0013 | 0.3159±0.0037 | 400653 |
| marigold |     any
Name: 23, dtype: object |     4
Name: 23, dtype: object |     shared_private
Name: 23, dtype: object |     private
Name: 23, dtype: object | 27.9010±3.6940 | 0.6718±0.7860 | 0.2338±0.0642 | 0.1702±0.0172 | 0.2977±0.0004 | 400653 |
| marigold |     any
Name: 24, dtype: object |     4
Name: 24, dtype: object |     shared_private
Name: 24, dtype: object |     shared
Name: 24, dtype: object | 39.5634±7.7418 | 0.7043±0.6604 | 0.2135±0.0137 | 0.1815±0.0286 | 0.3330±0.0158 | 400653 |
| marigold |     any
Name: 25, dtype: object |     8
Name: 25, dtype: object |     separate
Name: 25, dtype: object |     
Name: 25, dtype: object | 14.7835±0.0580 | 0.1122±0.0007 | 0.2148±0.0011 | 0.1648±0.0006 | 0.3182±0.0034 | 303377 |
| marigold |     any
Name: 26, dtype: object |     8
Name: 26, dtype: object |     shared
Name: 26, dtype: object |     
Name: 26, dtype: object | 15.1758±0.0888 | 0.1160±0.0014 | 0.2211±0.0012 | 0.1705±0.0021 | 0.3520±0.0061 | 239377 |
| marigold |     any
Name: 27, dtype: object |     8
Name: 27, dtype: object |     shared_private
Name: 27, dtype: object |     
Name: 27, dtype: object | 15.2985±0.2326 | 0.1147±0.0013 | 0.2174±0.0015 | 0.1646±0.0002 | 0.3120±0.0036 | 403217 |
| marigold |     any
Name: 28, dtype: object |     8
Name: 28, dtype: object |     shared_private
Name: 28, dtype: object |     private
Name: 28, dtype: object | 31.4690±2.7934 | 0.2462±0.0979 | 0.2433±0.0112 | 0.1932±0.0152 | 0.3041±0.0111 | 403217 |
| marigold |     any
Name: 29, dtype: object |     8
Name: 29, dtype: object |     shared_private
Name: 29, dtype: object |     shared
Name: 29, dtype: object | 35.4270±10.9699 | 1.2912±0.5302 | 0.2115±0.0067 | 0.2251±0.0494 | 0.3207±0.0157 | 403217 |
| marigold |     any
Name: 30, dtype: object |     16
Name: 30, dtype: object |     separate
Name: 30, dtype: object |     
Name: 30, dtype: object | 14.4806±0.0904 | 0.1109±0.0015 | 0.2143±0.0006 | 0.1632±0.0024 | 0.3153±0.0036 | 307481 |
| marigold |     any
Name: 31, dtype: object |     16
Name: 31, dtype: object |     shared
Name: 31, dtype: object |     
Name: 31, dtype: object | 15.1551±0.1397 | 0.1166±0.0010 | 0.2218±0.0010 | 0.1693±0.0009 | 0.3453±0.0026 | 245529 |
| marigold |     any
Name: 32, dtype: object |     16
Name: 32, dtype: object |     shared_private
Name: 32, dtype: object |     
Name: 32, dtype: object | 14.8111±0.2090 | 0.1148±0.0015 | 0.2150±0.0008 | 0.1646±0.0013 | 0.3198±0.0050 | 408345 |
| marigold |     any
Name: 33, dtype: object |     16
Name: 33, dtype: object |     shared_private
Name: 33, dtype: object |     private
Name: 33, dtype: object | 34.3398±3.4716 | 0.6401±0.6521 | 0.2154±0.0070 | 0.1784±0.0111 | 0.3222±0.0164 | 408345 |
| marigold |     any
Name: 34, dtype: object |     16
Name: 34, dtype: object |     shared_private
Name: 34, dtype: object |     shared
Name: 34, dtype: object | 23.3939±5.9685 | 1.0880±0.7253 | 0.2259±0.0140 | 0.2532±0.0706 | 0.3198±0.0074 | 408345 |
| marigold |     any
Name: 35, dtype: object |     32
Name: 35, dtype: object |     separate
Name: 35, dtype: object |     
Name: 35, dtype: object | 14.3423±0.0776 | 0.1111±0.0006 | 0.2151±0.0012 | 0.1636±0.0016 | 0.3198±0.0109 | 315689 |
| marigold |     any
Name: 36, dtype: object |     32
Name: 36, dtype: object |     shared
Name: 36, dtype: object |     
Name: 36, dtype: object | 14.9135±0.1073 | 0.1152±0.0019 | 0.2226±0.0025 | 0.1702±0.0015 | 0.3458±0.0022 | 257833 |
| marigold |     any
Name: 37, dtype: object |     32
Name: 37, dtype: object |     shared_private
Name: 37, dtype: object |     
Name: 37, dtype: object | 14.7170±0.0861 | 0.1140±0.0003 | 0.2146±0.0013 | 0.1643±0.0015 | 0.3218±0.0011 | 418601 |
| marigold |     any
Name: 38, dtype: object |     32
Name: 38, dtype: object |     shared_private
Name: 38, dtype: object |     private
Name: 38, dtype: object | 35.7075±2.6450 | 0.7423±0.5304 | 0.1987±0.0101 | 0.1641±0.0104 | 0.3950±0.0882 | 418601 |
| marigold |     any
Name: 39, dtype: object |     32
Name: 39, dtype: object |     shared_private
Name: 39, dtype: object |     shared
Name: 39, dtype: object | 18.1931±0.8341 | 1.0169±0.5091 | 0.2152±0.0119 | 0.1905±0.0167 | 0.3122±0.0082 | 418601 |
| marigold |     same_shape
Name: 40, dtype: object |     4
Name: 40, dtype: object |     separate
Name: 40, dtype: object |     
Name: 40, dtype: object | 19.4331±0.2279 | 0.1149±0.0021 | 0.2216±0.0005 | 0.1627±0.0014 | 0.3122±0.0016 | 301325 |
| marigold |     same_shape
Name: 41, dtype: object |     4
Name: 41, dtype: object |     shared
Name: 41, dtype: object |     
Name: 41, dtype: object | 13.6983±0.0789 | 0.1056±0.0005 | 0.2315±0.0013 | 0.1619±0.0004 | 0.3098±0.0054 | 236301 |
| marigold |     same_shape
Name: 42, dtype: object |     4
Name: 42, dtype: object |     shared_private
Name: 42, dtype: object |     
Name: 42, dtype: object | 14.7318±0.1686 | 0.1081±0.0017 | 0.2271±0.0007 | 0.1626±0.0006 | 0.3175±0.0012 | 400653 |
| marigold |     same_shape
Name: 43, dtype: object |     4
Name: 43, dtype: object |     shared_private
Name: 43, dtype: object |     private
Name: 43, dtype: object | 25.5562±6.1931 | 0.6502±0.9078 | 0.2479±0.0209 | 0.1732±0.0156 | 0.3113±0.0015 | 400653 |
| marigold |     same_shape
Name: 44, dtype: object |     4
Name: 44, dtype: object |     shared_private
Name: 44, dtype: object |     shared
Name: 44, dtype: object | 34.0997±4.0146 | 0.9519±0.6934 | 0.2098±0.0159 | 0.1982±0.0277 | 0.3500±0.0220 | 400653 |
| marigold |     same_shape
Name: 45, dtype: object |     8
Name: 45, dtype: object |     separate
Name: 45, dtype: object |     
Name: 45, dtype: object | 14.2512±0.1155 | 0.1103±0.0006 | 0.2174±0.0001 | 0.1604±0.0010 | 0.3179±0.0045 | 303377 |
| marigold |     same_shape
Name: 46, dtype: object |     8
Name: 46, dtype: object |     shared
Name: 46, dtype: object |     
Name: 46, dtype: object | 13.2525±0.1800 | 0.1037±0.0016 | 0.2235±0.0012 | 0.1591±0.0002 | 0.3170±0.0064 | 239377 |
| marigold |     same_shape
Name: 47, dtype: object |     8
Name: 47, dtype: object |     shared_private
Name: 47, dtype: object |     
Name: 47, dtype: object | 13.8055±0.3029 | 0.1058±0.0020 | 0.2191±0.0008 | 0.1596±0.0009 | 0.3154±0.0043 | 403217 |
| marigold |     same_shape
Name: 48, dtype: object |     8
Name: 48, dtype: object |     shared_private
Name: 48, dtype: object |     private
Name: 48, dtype: object | 27.9498±3.4398 | 0.2627±0.1674 | 0.2277±0.0189 | 0.1804±0.0097 | 0.3374±0.0266 | 403217 |
| marigold |     same_shape
Name: 49, dtype: object |     8
Name: 49, dtype: object |     shared_private
Name: 49, dtype: object |     shared
Name: 49, dtype: object | 32.5651±3.2702 | 1.1938±0.5838 | 0.2147±0.0075 | 0.2149±0.0474 | 0.3274±0.0286 | 403217 |
| marigold |     same_shape
Name: 50, dtype: object |     16
Name: 50, dtype: object |     separate
Name: 50, dtype: object |     
Name: 50, dtype: object | 14.1402±0.0408 | 0.1104±0.0018 | 0.2171±0.0001 | 0.1609±0.0007 | 0.3182±0.0041 | 307481 |
| marigold |     same_shape
Name: 51, dtype: object |     16
Name: 51, dtype: object |     shared
Name: 51, dtype: object |     
Name: 51, dtype: object | 13.1835±0.1189 | 0.1014±0.0005 | 0.2241±0.0002 | 0.1592±0.0004 | 0.3152±0.0027 | 245529 |
| marigold |     same_shape
Name: 52, dtype: object |     16
Name: 52, dtype: object |     shared_private
Name: 52, dtype: object |     
Name: 52, dtype: object | 13.4074±0.1733 | 0.1036±0.0012 | 0.2164±0.0012 | 0.1593±0.0019 | 0.3174±0.0061 | 408345 |
| marigold |     same_shape
Name: 53, dtype: object |     16
Name: 53, dtype: object |     shared_private
Name: 53, dtype: object |     private
Name: 53, dtype: object | 28.2910±11.2799 | 0.7622±0.6522 | 0.2283±0.0029 | 0.1747±0.0023 | 0.3773±0.0236 | 408345 |
| marigold |     same_shape
Name: 54, dtype: object |     16
Name: 54, dtype: object |     shared_private
Name: 54, dtype: object |     shared
Name: 54, dtype: object | 28.0028±3.8744 | 0.8638±0.6775 | 0.2291±0.0147 | 0.2341±0.1011 | 0.3208±0.0065 | 408345 |
| marigold |     same_shape
Name: 55, dtype: object |     32
Name: 55, dtype: object |     separate
Name: 55, dtype: object |     
Name: 55, dtype: object | 14.0901±0.1995 | 0.1103±0.0013 | 0.2183±0.0009 | 0.1620±0.0014 | 0.3191±0.0043 | 315689 |
| marigold |     same_shape
Name: 56, dtype: object |     32
Name: 56, dtype: object |     shared
Name: 56, dtype: object |     
Name: 56, dtype: object | 13.0805±0.1772 | 0.1028±0.0022 | 0.2252±0.0021 | 0.1597±0.0017 | 0.3146±0.0063 | 257833 |
| marigold |     same_shape
Name: 57, dtype: object |     32
Name: 57, dtype: object |     shared_private
Name: 57, dtype: object |     
Name: 57, dtype: object | 13.1836±0.0963 | 0.1035±0.0002 | 0.2170±0.0014 | 0.1591±0.0007 | 0.3183±0.0003 | 418601 |
| marigold |     same_shape
Name: 58, dtype: object |     32
Name: 58, dtype: object |     shared_private
Name: 58, dtype: object |     private
Name: 58, dtype: object | 24.6450±0.4604 | 0.7139±0.4647 | 0.2237±0.0099 | 0.1737±0.0078 | 0.3866±0.0470 | 418601 |
| marigold |     same_shape
Name: 59, dtype: object |     32
Name: 59, dtype: object |     shared_private
Name: 59, dtype: object |     shared
Name: 59, dtype: object | 23.4453±1.1314 | 0.9075±0.4236 | 0.2255±0.0203 | 0.1891±0.0095 | 0.3166±0.0086 | 418601 |
| marigold |     true
Name: 60, dtype: object |     4
Name: 60, dtype: object |     separate
Name: 60, dtype: object |     
Name: 60, dtype: object | 19.4331±0.2279 | 0.1149±0.0021 | 0.2199±0.0015 | 0.1621±0.0008 | 0.3163±0.0051 | 301325 |
| marigold |     true
Name: 61, dtype: object |     4
Name: 61, dtype: object |     shared
Name: 61, dtype: object |     
Name: 61, dtype: object | 14.1241±0.2585 | 0.1058±0.0013 | 0.2256±0.0013 | 0.1591±0.0008 | 0.3095±0.0030 | 236301 |
| marigold |     true
Name: 62, dtype: object |     4
Name: 62, dtype: object |     shared_private
Name: 62, dtype: object |     
Name: 62, dtype: object | 15.0376±0.0225 | 0.1088±0.0031 | 0.2276±0.0007 | 0.1624±0.0006 | 0.3153±0.0016 | 400653 |
| marigold |     true
Name: 63, dtype: object |     4
Name: 63, dtype: object |     shared_private
Name: 63, dtype: object |     private
Name: 63, dtype: object | 27.2661±6.5300 | 0.6898±0.9742 | 0.2511±0.0294 | 0.1717±0.0173 | 0.3107±0.0007 | 400653 |
| marigold |     true
Name: 64, dtype: object |     4
Name: 64, dtype: object |     shared_private
Name: 64, dtype: object |     shared
Name: 64, dtype: object | 35.2958±4.4849 | 0.8628±0.6088 | 0.2135±0.0119 | 0.1886±0.0245 | 0.3395±0.0147 | 400653 |
| marigold |     true
Name: 65, dtype: object |     8
Name: 65, dtype: object |     separate
Name: 65, dtype: object |     
Name: 65, dtype: object | 14.2512±0.1155 | 0.1103±0.0006 | 0.2183±0.0007 | 0.1615±0.0009 | 0.3219±0.0040 | 303377 |
| marigold |     true
Name: 66, dtype: object |     8
Name: 66, dtype: object |     shared
Name: 66, dtype: object |     
Name: 66, dtype: object | 13.6904±0.1125 | 0.1044±0.0008 | 0.2143±0.0008 | 0.1553±0.0021 | 0.3122±0.0046 | 239377 |
| marigold |     true
Name: 67, dtype: object |     8
Name: 67, dtype: object |     shared_private
Name: 67, dtype: object |     
Name: 67, dtype: object | 14.1653±0.3446 | 0.1055±0.0022 | 0.2194±0.0011 | 0.1593±0.0016 | 0.3133±0.0018 | 403217 |
| marigold |     true
Name: 68, dtype: object |     8
Name: 68, dtype: object |     shared_private
Name: 68, dtype: object |     private
Name: 68, dtype: object | 28.8297±1.1030 | 0.2919±0.1360 | 0.2149±0.0093 | 0.1828±0.0166 | 0.3689±0.0692 | 403217 |
| marigold |     true
Name: 69, dtype: object |     8
Name: 69, dtype: object |     shared_private
Name: 69, dtype: object |     shared
Name: 69, dtype: object | 35.6607±4.8518 | 1.0527±0.5590 | 0.2169±0.0133 | 0.1988±0.0466 | 0.3179±0.0302 | 403217 |
| marigold |     true
Name: 70, dtype: object |     16
Name: 70, dtype: object |     separate
Name: 70, dtype: object |     
Name: 70, dtype: object | 14.1402±0.0408 | 0.1104±0.0018 | 0.2163±0.0009 | 0.1601±0.0007 | 0.3207±0.0060 | 307481 |
| marigold |     true
Name: 71, dtype: object |     16
Name: 71, dtype: object |     shared
Name: 71, dtype: object |     
Name: 71, dtype: object | 13.5266±0.1096 | 0.1035±0.0003 | 0.2161±0.0026 | 0.1556±0.0014 | 0.3110±0.0023 | 245529 |
| marigold |     true
Name: 72, dtype: object |     16
Name: 72, dtype: object |     shared_private
Name: 72, dtype: object |     
Name: 72, dtype: object | 13.8230±0.1411 | 0.1038±0.0005 | 0.2157±0.0012 | 0.1587±0.0009 | 0.3197±0.0050 | 408345 |
| marigold |     true
Name: 73, dtype: object |     16
Name: 73, dtype: object |     shared_private
Name: 73, dtype: object |     private
Name: 73, dtype: object | 32.3278±15.7131 | 0.7570±0.6484 | 0.2331±0.0117 | 0.1671±0.0123 | 0.3390±0.0209 | 408345 |
| marigold |     true
Name: 74, dtype: object |     16
Name: 74, dtype: object |     shared_private
Name: 74, dtype: object |     shared
Name: 74, dtype: object | 27.2622±4.2859 | 0.8657±0.7066 | 0.2325±0.0075 | 0.2278±0.0650 | 0.3233±0.0068 | 408345 |
| marigold |     true
Name: 75, dtype: object |     32
Name: 75, dtype: object |     separate
Name: 75, dtype: object |     
Name: 75, dtype: object | 14.0901±0.1995 | 0.1103±0.0013 | 0.2180±0.0005 | 0.1612±0.0013 | 0.3178±0.0031 | 315689 |
| marigold |     true
Name: 76, dtype: object |     32
Name: 76, dtype: object |     shared
Name: 76, dtype: object |     
Name: 76, dtype: object | 13.4675±0.1893 | 0.1035±0.0012 | 0.2154±0.0018 | 0.1556±0.0010 | 0.3107±0.0053 | 257833 |
| marigold |     true
Name: 77, dtype: object |     32
Name: 77, dtype: object |     shared_private
Name: 77, dtype: object |     
Name: 77, dtype: object | 13.6786±0.1952 | 0.1037±0.0009 | 0.2141±0.0014 | 0.1600±0.0001 | 0.3193±0.0010 | 418601 |
| marigold |     true
Name: 78, dtype: object |     32
Name: 78, dtype: object |     shared_private
Name: 78, dtype: object |     private
Name: 78, dtype: object | 26.6474±0.6593 | 0.7380±0.4657 | 0.2126±0.0036 | 0.1717±0.0127 | 0.4153±0.0803 | 418601 |
| marigold |     true
Name: 79, dtype: object |     32
Name: 79, dtype: object |     shared_private
Name: 79, dtype: object |     shared
Name: 79, dtype: object | 22.9814±2.7330 | 0.9007±0.4333 | 0.2209±0.0164 | 0.1981±0.0286 | 0.3166±0.0147 | 418601 |

### Shared-minus-separate composite gap: true vs shuffled pairing (Marigold, regime = all)

| pairing | k | Δgeo (shared/sep − 1) | Δmat |
|---|---:|---:|---:|
| true | 4 | -0.190 | -0.005 |
| true | 8 | -0.046 | -0.029 |
| true | 16 | -0.053 | -0.020 |
| true | 32 | -0.053 | -0.023 |
| same_shape | 4 | -0.203 | +0.011 |
| same_shape | 8 | -0.065 | +0.006 |
| same_shape | 16 | -0.075 | +0.004 |
| same_shape | 32 | -0.070 | +0.001 |
| any | 4 | -0.147 | +0.034 |
| any | 8 | +0.030 | +0.057 |
| any | 16 | +0.049 | +0.056 |
| any | 32 | +0.038 | +0.052 |

## 6. Locked decision rule → Experiment A representation

```
{
 "k_star": 8,
 "shared_vs_separate": {
  "geo": -0.046461577278825716,
  "mat": -0.028750312599500227
 },
 "sharedprivate_vs_separate": {
  "geo": -0.025001679457738013,
  "mat": -0.011636282735177361
 },
 "private_zeroing_cost": {
  "geo": 1.3921639407492101,
  "mat": 0.10069433515011839
 },
 "choice": "shared"
}
```
