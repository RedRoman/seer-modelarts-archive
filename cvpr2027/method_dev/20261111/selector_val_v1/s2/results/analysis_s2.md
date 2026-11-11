# Correction selector: exploratory BVI VAL16

Exploratory BVI VAL16, eight scenes, one trained seed; not TEST/SOTA/equivalence evidence

| Method | PSNR-RGB ↑ | PSNR-Y ↑ | LPIPS ↓ | tLP gap ↓ |
|---|---:|---:|---:|---:|
| base | 28.105014 | 29.127795 | 0.15323244 | 0.00646416 |
| fixed_average | 28.108541 | 29.037028 | 0.16379341 | 0.00585246 |
| initial_thirds | 28.233574 | 29.186168 | 0.15628402 | 0.00659018 |
| risk_full | 28.512510 | 29.531962 | 0.14827944 | 0.00638263 |
| direct_mse | 28.552926 | 29.587902 | 0.14503824 | 0.00582229 |
| direct_mse_normalized | 28.560812 | 29.593137 | 0.14473429 | 0.00564310 |

Differences are left minus right. Intervals resample eight whole scenes (10,000 draws, RNG0).
PSNR positive is favorable; LPIPS/tLP gap negative is favorable. Intervals are unadjusted and conditional on one training seed.

| Contrast | Metric | Difference | Scene 95% CI | Better sequences / 16 |
|---|---|---:|---|---:|
| risk_full_minus_base | psnr_rgb | +0.40749557 | [+0.12984531, +0.82696466] | 14 |
| risk_full_minus_base | psnr_y | +0.40416668 | [+0.09993683, +0.85305068] | 12 |
| risk_full_minus_base | lpips | -0.00495300 | [-0.01995583, +0.00479776] | 8 |
| risk_full_minus_base | tlp_gap | -0.00008154 | [-0.00176351, +0.00156114] | 8 |
| direct_mse_minus_base | psnr_rgb | +0.44791181 | [+0.15541979, +0.89959786] | 15 |
| direct_mse_minus_base | psnr_y | +0.46010663 | [+0.13345929, +0.94772428] | 13 |
| direct_mse_minus_base | lpips | -0.00819420 | [-0.02501941, +0.00259865] | 8 |
| direct_mse_minus_base | tlp_gap | -0.00064187 | [-0.00246938, +0.00119045] | 10 |
| direct_mse_normalized_minus_base | psnr_rgb | +0.45579795 | [+0.15947131, +0.91709261] | 15 |
| direct_mse_normalized_minus_base | psnr_y | +0.46534170 | [+0.13851281, +0.95896729] | 14 |
| direct_mse_normalized_minus_base | lpips | -0.00849815 | [-0.02568196, +0.00251335] | 8 |
| direct_mse_normalized_minus_base | tlp_gap | -0.00082106 | [-0.00287083, +0.00106630] | 10 |
| fixed_average_minus_base | psnr_rgb | +0.00352677 | [-0.33863632, +0.41096147] | 7 |
| fixed_average_minus_base | psnr_y | -0.09076670 | [-0.45187672, +0.34774221] | 7 |
| fixed_average_minus_base | lpips | +0.01056097 | [-0.00549859, +0.02310584] | 2 |
| fixed_average_minus_base | tlp_gap | -0.00061170 | [-0.00378932, +0.00213589] | 7 |
| risk_full_minus_direct_mse | psnr_rgb | -0.04041624 | [-0.07597943, -0.01423927] | 6 |
| risk_full_minus_direct_mse | psnr_y | -0.05593995 | [-0.09899329, -0.02298864] | 6 |
| risk_full_minus_direct_mse | lpips | +0.00324120 | [+0.00169198, +0.00529233] | 1 |
| risk_full_minus_direct_mse | tlp_gap | +0.00056033 | [-0.00001970, +0.00134344] | 6 |
| risk_full_minus_direct_mse_normalized | psnr_rgb | -0.04830239 | [-0.09753196, -0.01501152] | 6 |
| risk_full_minus_direct_mse_normalized | psnr_y | -0.06117501 | [-0.11097282, -0.02460407] | 6 |
| risk_full_minus_direct_mse_normalized | lpips | +0.00354515 | [+0.00175871, +0.00600030] | 2 |
| risk_full_minus_direct_mse_normalized | tlp_gap | +0.00073952 | [+0.00001464, +0.00181741] | 3 |

No oracle is scored on VAL. The earlier six-scene selector holdout was internal to TRAIN; this report uses the separate frozen VAL scenes.
The temporal gap is abs(tLP(pred)-tLP(GT)); a smaller value alone does not establish better fidelity or rule out oversmoothing.
Raw per-sequence/scene effects and input hashes are retained in the adjacent JSON.
