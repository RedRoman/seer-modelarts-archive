# Correction selector: exploratory BVI VAL16

Exploratory BVI VAL16, eight scenes, one trained seed; not TEST/SOTA/equivalence evidence

| Method | PSNR-RGB ↑ | PSNR-Y ↑ | LPIPS ↓ | tLP gap ↓ |
|---|---:|---:|---:|---:|
| base | 28.105014 | 29.127795 | 0.15323244 | 0.00646416 |
| fixed_average | 28.108507 | 29.036974 | 0.16379354 | 0.00585217 |
| initial_thirds | 28.233546 | 29.186129 | 0.15628222 | 0.00658859 |
| risk_full | 28.500640 | 29.523008 | 0.14814764 | 0.00658738 |
| direct_mse | 28.562123 | 29.593887 | 0.14475512 | 0.00584914 |
| direct_mse_normalized | 28.556859 | 29.586517 | 0.14478429 | 0.00578971 |

Differences are left minus right. Intervals resample eight whole scenes (10,000 draws, RNG0).
PSNR positive is favorable; LPIPS/tLP gap negative is favorable. Intervals are unadjusted and conditional on one training seed.

| Contrast | Metric | Difference | Scene 95% CI | Better sequences / 16 |
|---|---|---:|---|---:|
| risk_full_minus_base | psnr_rgb | +0.39562598 | [+0.11585044, +0.82506829] | 14 |
| risk_full_minus_base | psnr_y | +0.39521322 | [+0.08399578, +0.86123152] | 12 |
| risk_full_minus_base | lpips | -0.00508480 | [-0.02084273, +0.00518996] | 8 |
| risk_full_minus_base | tlp_gap | +0.00012322 | [-0.00175020, +0.00185478] | 7 |
| direct_mse_minus_base | psnr_rgb | +0.45710855 | [+0.16725207, +0.91634748] | 15 |
| direct_mse_minus_base | psnr_y | +0.46609210 | [+0.14355172, +0.95676143] | 14 |
| direct_mse_minus_base | lpips | -0.00847732 | [-0.02595951, +0.00259808] | 8 |
| direct_mse_minus_base | tlp_gap | -0.00061502 | [-0.00299932, +0.00153847] | 10 |
| direct_mse_normalized_minus_base | psnr_rgb | +0.45184434 | [+0.16326450, +0.90933054] | 15 |
| direct_mse_normalized_minus_base | psnr_y | +0.45872187 | [+0.13903569, +0.94758529] | 14 |
| direct_mse_normalized_minus_base | lpips | -0.00844815 | [-0.02601944, +0.00271392] | 8 |
| direct_mse_normalized_minus_base | tlp_gap | -0.00067445 | [-0.00309620, +0.00149495] | 9 |
| fixed_average_minus_base | psnr_rgb | +0.00349248 | [-0.33867072, +0.41095681] | 7 |
| fixed_average_minus_base | psnr_y | -0.09082114 | [-0.45192900, +0.34762147] | 7 |
| fixed_average_minus_base | lpips | +0.01056110 | [-0.00549735, +0.02310772] | 2 |
| fixed_average_minus_base | tlp_gap | -0.00061199 | [-0.00378581, +0.00213470] | 7 |
| risk_full_minus_direct_mse | psnr_rgb | -0.06148257 | [-0.10420424, -0.02529279] | 3 |
| risk_full_minus_direct_mse | psnr_y | -0.07087888 | [-0.10961166, -0.03602285] | 4 |
| risk_full_minus_direct_mse | lpips | +0.00339253 | [+0.00169755, +0.00546674] | 2 |
| risk_full_minus_direct_mse | tlp_gap | +0.00073824 | [-0.00012743, +0.00207377] | 7 |
| risk_full_minus_direct_mse_normalized | psnr_rgb | -0.05621836 | [-0.09334976, -0.02687960] | 4 |
| risk_full_minus_direct_mse_normalized | psnr_y | -0.06350865 | [-0.09422486, -0.03679479] | 4 |
| risk_full_minus_direct_mse_normalized | lpips | +0.00336336 | [+0.00167686, +0.00547924] | 2 |
| risk_full_minus_direct_mse_normalized | tlp_gap | +0.00079767 | [-0.00008792, +0.00221953] | 7 |

No oracle is scored on VAL. The earlier six-scene selector holdout was internal to TRAIN; this report uses the separate frozen VAL scenes.
The temporal gap is abs(tLP(pred)-tLP(GT)); a smaller value alone does not establish better fidelity or rule out oversmoothing.
Raw per-sequence/scene effects and input hashes are retained in the adjacent JSON.
