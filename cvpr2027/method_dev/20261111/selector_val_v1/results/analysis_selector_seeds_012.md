# Correction selector: three selector seeds on BVI VAL16

Exploratory frozen BVI VAL16; selector seeds0/1/2; one fixed E12 base seed0

The base remains E12 seed0. These are three selector fits, not three independent base trainings.

| Method | Metric | Mean | Range across selector seeds |
|---|---|---:|---|
| base | psnr_rgb | 28.10501439 | [28.10501439, 28.10501439] |
| base | psnr_y | 29.12779492 | [29.12779492, 29.12779492] |
| base | lpips | 0.15323244 | [0.15323244, 0.15323244] |
| base | tlp_gap | 0.00646416 | [0.00646416, 0.00646416] |
| fixed_average | psnr_rgb | 28.10851745 | [28.10850432, 28.10854116] |
| fixed_average | psnr_y | 29.03699294 | [29.03697379, 29.03702822] |
| fixed_average | lpips | 0.16379365 | [0.16379341, 0.16379399] |
| fixed_average | tlp_gap | 0.00585214 | [0.00585179, 0.00585246] |
| initial_thirds | psnr_rgb | 28.23355742 | [28.23354581, 28.23357353] |
| initial_thirds | psnr_y | 29.18614543 | [29.18612888, 29.18616798] |
| initial_thirds | lpips | 0.15628344 | [0.15628222, 0.15628407] |
| initial_thirds | tlp_gap | 0.00658902 | [0.00658828, 0.00659018] |
| risk_full | psnr_rgb | 28.51176465 | [28.50064037, 28.52214362] |
| risk_full | psnr_y | 29.53098841 | [29.52300814, 29.53799549] |
| risk_full | lpips | 0.14772360 | [0.14674371, 0.14827944] |
| risk_full | tlp_gap | 0.00641380 | [0.00627140, 0.00658738] |
| direct_mse | psnr_rgb | 28.55474934 | [28.54919889, 28.56212294] |
| direct_mse | psnr_y | 29.58664231 | [29.57813834, 29.59388702] |
| direct_mse | lpips | 0.14493868 | [0.14475512, 0.14503824] |
| direct_mse | tlp_gap | 0.00584105 | [0.00582229, 0.00585173] |
| direct_mse_normalized | psnr_rgb | 28.55544207 | [28.54865512, 28.56081235] |
| direct_mse_normalized | psnr_y | 29.58588396 | [29.57799847, 29.59313662] |
| direct_mse_normalized | lpips | 0.14490669 | [0.14473429, 0.14520149] |
| direct_mse_normalized | tlp_gap | 0.00576950 | [0.00564310, 0.00587568] |

| Contrast | Metric | Mean difference | Scene 95% CI | Favorable selector seeds / 3 |
|---|---|---:|---|---:|
| risk_full_minus_base | psnr_rgb | +0.40675025 | [+0.12815571, +0.83613002] | 3 |
| risk_full_minus_base | psnr_y | +0.40319349 | [+0.09488482, +0.86065455] | 3 |
| risk_full_minus_base | lpips | -0.00550884 | [-0.02092676, +0.00445539] | 3 |
| risk_full_minus_base | tlp_gap | -0.00005036 | [-0.00182629, +0.00166788] | 2 |
| direct_mse_minus_base | psnr_rgb | +0.44973495 | [+0.16020956, +0.90152769] | 3 |
| direct_mse_minus_base | psnr_y | +0.45884738 | [+0.13643933, +0.94815887] | 3 |
| direct_mse_minus_base | lpips | -0.00829376 | [-0.02528465, +0.00258485] | 3 |
| direct_mse_minus_base | tlp_gap | -0.00062311 | [-0.00270583, +0.00136168] | 3 |
| direct_mse_normalized_minus_base | psnr_rgb | +0.45042767 | [+0.16232993, +0.90332690] | 3 |
| direct_mse_normalized_minus_base | psnr_y | +0.45808904 | [+0.14056056, +0.94198065] | 3 |
| direct_mse_normalized_minus_base | lpips | -0.00832575 | [-0.02546862, +0.00263823] | 3 |
| direct_mse_normalized_minus_base | tlp_gap | -0.00069467 | [-0.00284593, +0.00132441] | 3 |
| fixed_average_minus_base | psnr_rgb | +0.00350306 | [-0.33866259, +0.41097075] | 3 |
| fixed_average_minus_base | psnr_y | -0.09080198 | [-0.45190511, +0.34766863] | 0 |
| fixed_average_minus_base | lpips | +0.01056121 | [-0.00549823, +0.02310807] | 0 |
| fixed_average_minus_base | tlp_gap | -0.00061202 | [-0.00378749, +0.00213494] | 3 |
| risk_full_minus_direct_mse | psnr_rgb | -0.04298469 | [-0.07450073, -0.01820569] | 0 |
| risk_full_minus_direct_mse | psnr_y | -0.05565389 | [-0.09080350, -0.02919014] | 0 |
| risk_full_minus_direct_mse | lpips | +0.00278492 | [+0.00135256, +0.00461505] | 0 |
| risk_full_minus_direct_mse | tlp_gap | +0.00057275 | [-0.00003185, +0.00144480] | 0 |
| risk_full_minus_direct_mse_normalized | psnr_rgb | -0.04367742 | [-0.07661975, -0.01836441] | 0 |
| risk_full_minus_direct_mse_normalized | psnr_y | -0.05489555 | [-0.08834987, -0.02945103] | 0 |
| risk_full_minus_direct_mse_normalized | lpips | +0.00281691 | [+0.00132133, +0.00478155] | 0 |
| risk_full_minus_direct_mse_normalized | tlp_gap | +0.00064430 | [-0.00000971, +0.00163473] | 0 |

| Fixed control | Metric | Maximum absolute drift vs seed0 | Mean absolute drift (seeds1/2 vs seed0) |
|---|---|---:|---:|
| base | psnr_rgb | 0.00000000 | 0.00000000 |
| base | psnr_y | 0.00000000 | 0.00000000 |
| base | lpips | 0.00000000 | 0.00000000 |
| base | tlp_gap | 0.00000000 | 0.00000000 |
| fixed_average | psnr_rgb | 0.00023557 | 0.00006912 |
| fixed_average | psnr_y | 0.00032138 | 0.00009013 |
| fixed_average | lpips | 0.00001904 | 0.00000696 |
| fixed_average | tlp_gap | 0.00003619 | 0.00000754 |
| initial_thirds | psnr_rgb | 0.00019163 | 0.00005042 |
| initial_thirds | psnr_y | 0.00022065 | 0.00006391 |
| initial_thirds | lpips | 0.00002164 | 0.00000465 |
| initial_thirds | tlp_gap | 0.00000866 | 0.00000364 |

| Fixed control | Metric | Max drift / direct-vs-base gain | Mean drift / gain |
|---|---|---:|---:|
| fixed_average | psnr_rgb | 0.052380% | 0.015370% |
| fixed_average | psnr_y | 0.070041% | 0.019642% |
| initial_thirds | psnr_rgb | 0.042609% | 0.011211% |
| initial_thirds | psnr_y | 0.048088% | 0.013928% |

First average each sequence's paired effect across selector seeds, then resample eight complete scenes (10,000 draws, RNG0).
Intervals are unadjusted and conditional on these three selectors, one base and the actual candidate computations. Repeated sequences are not counted as 48 independent observations.
Fixed-average/initial-thirds metric drift is reported, not assumed zero. Candidates are shared within each run; cross-run changes are not attributable solely to selector initialization. Their numerical source is not isolated by this analysis.
PSNR positive is favorable; LPIPS/tLP gap negative is favorable. VAL selection, base-seed uncertainty and SOTA claims are outside this analysis.
