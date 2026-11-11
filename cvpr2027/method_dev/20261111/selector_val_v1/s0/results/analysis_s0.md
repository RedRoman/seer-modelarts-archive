# Correction selector: exploratory BVI VAL16

Exploratory BVI VAL16, eight scenes, one trained seed; not TEST/SOTA/equivalence evidence

| Method | PSNR-RGB ↑ | PSNR-Y ↑ | LPIPS ↓ | tLP gap ↓ |
|---|---:|---:|---:|---:|
| base | 28.105014 | 29.127795 | 0.15323244 | 0.00646416 |
| fixed_average | 28.108504 | 29.036977 | 0.16379399 | 0.00585179 |
| initial_thirds | 28.233553 | 29.186139 | 0.15628407 | 0.00658828 |
| risk_full | 28.522144 | 29.537995 | 0.14674371 | 0.00627140 |
| direct_mse | 28.549199 | 29.578138 | 0.14502269 | 0.00585173 |
| direct_mse_normalized | 28.548655 | 29.577998 | 0.14520149 | 0.00587568 |

Differences are left minus right. Intervals resample eight whole scenes (10,000 draws, RNG0).
PSNR positive is favorable; LPIPS/tLP gap negative is favorable. Intervals are unadjusted and conditional on one training seed.

| Contrast | Metric | Difference | Scene 95% CI | Better sequences / 16 |
|---|---|---:|---|---:|
| risk_full_minus_base | psnr_rgb | +0.41712922 | [+0.13227332, +0.85430120] | 15 |
| risk_full_minus_base | psnr_y | +0.41020057 | [+0.09996376, +0.87235974] | 13 |
| risk_full_minus_base | lpips | -0.00648873 | [-0.02197727, +0.00350436] | 8 |
| risk_full_minus_base | tlp_gap | -0.00019277 | [-0.00206417, +0.00160632] | 8 |
| direct_mse_minus_base | psnr_rgb | +0.44418450 | [+0.15716582, +0.89219904] | 15 |
| direct_mse_minus_base | psnr_y | +0.45034342 | [+0.13289834, +0.93443059] | 13 |
| direct_mse_minus_base | lpips | -0.00820975 | [-0.02499563, +0.00257977] | 8 |
| direct_mse_minus_base | tlp_gap | -0.00061243 | [-0.00267156, +0.00134179] | 9 |
| direct_mse_normalized_minus_base | psnr_rgb | +0.44364073 | [+0.16296645, +0.88451624] | 15 |
| direct_mse_normalized_minus_base | psnr_y | +0.45020355 | [+0.14086432, +0.92327697] | 13 |
| direct_mse_normalized_minus_base | lpips | -0.00803095 | [-0.02482618, +0.00273533] | 8 |
| direct_mse_normalized_minus_base | tlp_gap | -0.00058848 | [-0.00262231, +0.00135477] | 9 |
| fixed_average_minus_base | psnr_rgb | +0.00348993 | [-0.33868072, +0.41099395] | 7 |
| fixed_average_minus_base | psnr_y | -0.09081812 | [-0.45190961, +0.34764220] | 7 |
| fixed_average_minus_base | lpips | +0.01056155 | [-0.00549874, +0.02311064] | 2 |
| fixed_average_minus_base | tlp_gap | -0.00061237 | [-0.00378835, +0.00213643] | 7 |
| risk_full_minus_direct_mse | psnr_rgb | -0.02705527 | [-0.04771630, -0.00799828] | 8 |
| risk_full_minus_direct_mse | psnr_y | -0.04014285 | [-0.07142897, -0.01611667] | 7 |
| risk_full_minus_direct_mse | lpips | +0.00172102 | [+0.00048024, +0.00318816] | 4 |
| risk_full_minus_direct_mse | tlp_gap | +0.00041966 | [+0.00000999, +0.00093325] | 6 |
| risk_full_minus_direct_mse_normalized | psnr_rgb | -0.02651151 | [-0.05065173, -0.00505605] | 7 |
| risk_full_minus_direct_mse_normalized | psnr_y | -0.04000298 | [-0.06854074, -0.01509674] | 6 |
| risk_full_minus_direct_mse_normalized | lpips | +0.00154221 | [+0.00032944, +0.00299439] | 6 |
| risk_full_minus_direct_mse_normalized | tlp_gap | +0.00039572 | [+0.00000891, +0.00086460] | 7 |

No oracle is scored on VAL. The earlier six-scene selector holdout was internal to TRAIN; this report uses the separate frozen VAL scenes.
The temporal gap is abs(tLP(pred)-tLP(GT)); a smaller value alone does not establish better fidelity or rule out oversmoothing.
Raw per-sequence/scene effects and input hashes are retained in the adjacent JSON.
