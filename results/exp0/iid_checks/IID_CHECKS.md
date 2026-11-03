# Marigold-IID accuracy/convention checks (24 val objects, 72 views; Amendment 2, part 3)

Mean-predictor baselines: {"albedo_mae": 0.1792, "rough_mae": 0.1856, "metal_mae": 0.4511, "metal_fraction": 0.3436}

| setting | albedo MAE sRGB→lin / as-lin | albedo r | gain-aligned albedo MAE | rough MAE (ch0) | metal MAE (ch1) | ch0~rough r | ch1~metal r | ch1 on metal / dielectric |
|---|---|---|---|---|---|---|---|---|
| main(black,512,1step) | 0.322 / 0.496 | 0.444 | 0.102 | 0.147 | 0.463 | 0.542 | -0.111 | 0.32 / 0.35 |
| grey_background | 0.196 / 0.388 | 0.729 | 0.080 | 0.153 | 0.518 | 0.508 | -0.153 | 0.43 / 0.49 |
| 4_steps | 0.360 / 0.525 | 0.443 | 0.104 | 0.159 | 0.462 | 0.441 | -0.096 | 0.32 / 0.35 |
| 4_steps_ens5 | 0.352 / 0.521 | 0.509 | 0.103 | 0.158 | 0.462 | 0.446 | -0.108 | 0.32 / 0.35 |
| proc_res_768 | 0.294 / 0.477 | 0.454 | 0.105 | 0.152 | 0.474 | 0.498 | -0.238 | 0.29 / 0.35 |

Channel-order matrix (Pearson of predicted material channel vs GT), main setting:

```
{
 "ch0": {
  "rough": 0.5419265031814575,
  "metal": -0.20933327078819275
 },
 "ch1": {
  "rough": 0.003855776274576783,
  "metal": -0.1106543317437172
 },
 "ch2": {
  "rough": 0.1248069480061531,
  "metal": -0.06707974523305893
 }
}
```
