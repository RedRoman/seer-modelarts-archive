# Source-trained temporal correction on BVI VAL16

Exploratory SDSD-only restoration/selector training to BVI VAL16; one base seed0, three selector seeds

| Arm | PSNR-RGB | PSNR-Y | LPIPS | tLP gap |
|---|---:|---:|---:|---:|
| base | 18.35735052 | 19.28493130 | 0.24482940 | 0.00523702 |
| restored/fixed_average | 18.34248221 | 19.25630854 | 0.25348093 | 0.00471710 |
| restored/initial_thirds | 18.36039852 | 19.27894864 | 0.24786317 | 0.00549989 |
| restored_detail/fixed_average | 18.34345655 | 19.26261612 | 0.25396539 | 0.00452385 |
| restored_detail/initial_thirds | 18.35392376 | 19.27634421 | 0.25099401 | 0.00459407 |
| raw_detail/fixed_average | 18.33470973 | 19.29912292 | 0.20783515 | 0.02323663 |
| raw_detail/initial_thirds | 18.42674187 | 19.38691442 | 0.18783542 | 0.01327217 |
| restored/direct_mse | 18.29150318 | 19.19735559 | 0.24744069 | 0.00428472 |
| restored/direct_mse_normalized | 18.29325696 | 19.19960057 | 0.24740097 | 0.00426200 |
| restored_detail/direct_mse | 18.33885549 | 19.26154954 | 0.24774693 | 0.00384183 |
| restored_detail/direct_mse_normalized | 18.33964102 | 19.26259059 | 0.24760914 | 0.00388740 |
| raw_detail/direct_mse | 18.43302889 | 19.38255456 | 0.20255497 | 0.00685211 |
| raw_detail/direct_mse_normalized | 18.43252469 | 19.38135521 | 0.20320398 | 0.00650054 |

| Paired contrast | Metric | Mean difference | Scene 95% CI | Favorable seeds / 3 |
|---|---|---:|---|---:|
| restored_minus_base/direct_mse | psnr_rgb | -0.06584734 | [-0.11716671,-0.02674177] | 0 |
| restored_minus_base/direct_mse | psnr_y | -0.08757571 | [-0.14006159,-0.04546980] | 0 |
| restored_minus_base/direct_mse | lpips | +0.00261129 | [-0.02076729,+0.01880201] | 0 |
| restored_minus_base/direct_mse | tlp_gap | -0.00095230 | [-0.00274498,+0.00105357] | 3 |
| restored_minus_base/direct_mse_normalized | psnr_rgb | -0.06409355 | [-0.11250760,-0.02611800] | 0 |
| restored_minus_base/direct_mse_normalized | psnr_y | -0.08533073 | [-0.13489181,-0.04462356] | 0 |
| restored_minus_base/direct_mse_normalized | lpips | +0.00257158 | [-0.02104626,+0.01891299] | 0 |
| restored_minus_base/direct_mse_normalized | tlp_gap | -0.00097502 | [-0.00280838,+0.00108204] | 3 |
| restored_detail_minus_base/direct_mse | psnr_rgb | -0.01849503 | [-0.06687469,+0.01826685] | 0 |
| restored_detail_minus_base/direct_mse | psnr_y | -0.02338176 | [-0.07873740,+0.01825046] | 0 |
| restored_detail_minus_base/direct_mse | lpips | +0.00291754 | [-0.02187518,+0.01977131] | 0 |
| restored_detail_minus_base/direct_mse | tlp_gap | -0.00139519 | [-0.00288238,+0.00024401] | 3 |
| restored_detail_minus_base/direct_mse_normalized | psnr_rgb | -0.01770949 | [-0.06532226,+0.01847554] | 0 |
| restored_detail_minus_base/direct_mse_normalized | psnr_y | -0.02234071 | [-0.07695144,+0.01857876] | 0 |
| restored_detail_minus_base/direct_mse_normalized | lpips | +0.00277975 | [-0.02193161,+0.01955776] | 0 |
| restored_detail_minus_base/direct_mse_normalized | tlp_gap | -0.00134961 | [-0.00283340,+0.00027886] | 3 |
| raw_detail_minus_base/direct_mse | psnr_rgb | +0.07567837 | [+0.03738844,+0.14072885] | 3 |
| raw_detail_minus_base/direct_mse | psnr_y | +0.09762326 | [+0.04200681,+0.17368558] | 3 |
| raw_detail_minus_base/direct_mse | lpips | -0.04227443 | [-0.04887322,-0.03520548] | 3 |
| raw_detail_minus_base/direct_mse | tlp_gap | +0.00161509 | [+0.00052818,+0.00270214] | 0 |
| raw_detail_minus_base/direct_mse_normalized | psnr_rgb | +0.07517417 | [+0.03680213,+0.14075183] | 3 |
| raw_detail_minus_base/direct_mse_normalized | psnr_y | +0.09642391 | [+0.04076753,+0.17303513] | 3 |
| raw_detail_minus_base/direct_mse_normalized | lpips | -0.04162542 | [-0.04847959,-0.03460398] | 3 |
| raw_detail_minus_base/direct_mse_normalized | tlp_gap | +0.00126353 | [+0.00030345,+0.00224503] | 0 |
| restored_detail_minus_restored/direct_mse | psnr_rgb | +0.04735231 | [+0.03168635,+0.06278629] | 3 |
| restored_detail_minus_restored/direct_mse | psnr_y | +0.06419395 | [+0.04311409,+0.08453491] | 3 |
| restored_detail_minus_restored/direct_mse | lpips | +0.00030625 | [-0.00207154,+0.00255118] | 1 |
| restored_detail_minus_restored/direct_mse | tlp_gap | -0.00044289 | [-0.00087690,+0.00001572] | 3 |
| restored_detail_minus_restored/direct_mse_normalized | psnr_rgb | +0.04638406 | [+0.03148485,+0.06133304] | 3 |
| restored_detail_minus_restored/direct_mse_normalized | psnr_y | +0.06299002 | [+0.04196263,+0.08391766] | 3 |
| restored_detail_minus_restored/direct_mse_normalized | lpips | +0.00020817 | [-0.00204838,+0.00229060] | 2 |
| restored_detail_minus_restored/direct_mse_normalized | tlp_gap | -0.00037460 | [-0.00087230,+0.00011725] | 3 |
| raw_detail_minus_restored/direct_mse | psnr_rgb | +0.14152571 | [+0.07514290,+0.22733682] | 3 |
| raw_detail_minus_restored/direct_mse | psnr_y | +0.18519897 | [+0.11011748,+0.27758375] | 3 |
| raw_detail_minus_restored/direct_mse | lpips | -0.04488572 | [-0.06514874,-0.02060377] | 3 |
| raw_detail_minus_restored/direct_mse | tlp_gap | +0.00256739 | [-0.00046905,+0.00541985] | 0 |
| raw_detail_minus_restored/direct_mse_normalized | psnr_rgb | +0.13926773 | [+0.07348974,+0.22497642] | 3 |
| raw_detail_minus_restored/direct_mse_normalized | psnr_y | +0.18175464 | [+0.10678601,+0.27426777] | 3 |
| raw_detail_minus_restored/direct_mse_normalized | lpips | -0.04419699 | [-0.06517900,-0.01878400] | 3 |
| raw_detail_minus_restored/direct_mse_normalized | tlp_gap | +0.00223854 | [-0.00070103,+0.00500886] | 0 |
| raw_detail_minus_restored_detail/direct_mse | psnr_rgb | +0.09417340 | [+0.02733722,+0.18526150] | 3 |
| raw_detail_minus_restored_detail/direct_mse | psnr_y | +0.12100502 | [+0.04434517,+0.22052010] | 3 |
| raw_detail_minus_restored_detail/direct_mse | lpips | -0.04519197 | [-0.06673326,-0.01919216] | 3 |
| raw_detail_minus_restored_detail/direct_mse | tlp_gap | +0.00301028 | [+0.00033477,+0.00545592] | 0 |
| raw_detail_minus_restored_detail/direct_mse_normalized | psnr_rgb | +0.09288367 | [+0.02576079,+0.18372501] | 3 |
| raw_detail_minus_restored_detail/direct_mse_normalized | psnr_y | +0.11876462 | [+0.04217399,+0.21842886] | 3 |
| raw_detail_minus_restored_detail/direct_mse_normalized | lpips | -0.04440516 | [-0.06645703,-0.01752083] | 3 |
| raw_detail_minus_restored_detail/direct_mse_normalized | tlp_gap | +0.00261314 | [+0.00011001,+0.00495188] | 0 |
| restored_minus_fixed_average/direct_mse | psnr_rgb | -0.05097903 | [-0.08405393,-0.02449253] | 0 |
| restored_minus_fixed_average/direct_mse | psnr_y | -0.05895295 | [-0.09419370,-0.02955171] | 0 |
| restored_minus_fixed_average/direct_mse | lpips | -0.00604024 | [-0.00826316,-0.00414858] | 3 |
| restored_minus_fixed_average/direct_mse | tlp_gap | -0.00043238 | [-0.00122075,+0.00022082] | 3 |
| restored_minus_fixed_average/direct_mse_normalized | psnr_rgb | -0.04922525 | [-0.07985152,-0.02353006] | 0 |
| restored_minus_fixed_average/direct_mse_normalized | psnr_y | -0.05670797 | [-0.08965695,-0.02870126] | 0 |
| restored_minus_fixed_average/direct_mse_normalized | lpips | -0.00607996 | [-0.00844920,-0.00415287] | 3 |
| restored_minus_fixed_average/direct_mse_normalized | tlp_gap | -0.00045510 | [-0.00126895,+0.00024696] | 3 |
| restored_minus_initial_thirds/direct_mse | psnr_rgb | -0.06889534 | [-0.10897253,-0.03824072] | 0 |
| restored_minus_initial_thirds/direct_mse | psnr_y | -0.08159305 | [-0.12188522,-0.04896641] | 0 |
| restored_minus_initial_thirds/direct_mse | lpips | -0.00042248 | [-0.00640257,+0.00327843] | 3 |
| restored_minus_initial_thirds/direct_mse | tlp_gap | -0.00121517 | [-0.00245056,-0.00035948] | 3 |
| restored_minus_initial_thirds/direct_mse_normalized | psnr_rgb | -0.06714156 | [-0.10443500,-0.03768148] | 0 |
| restored_minus_initial_thirds/direct_mse_normalized | psnr_y | -0.07934806 | [-0.11675620,-0.04773312] | 0 |
| restored_minus_initial_thirds/direct_mse_normalized | lpips | -0.00046219 | [-0.00666680,+0.00339166] | 3 |
| restored_minus_initial_thirds/direct_mse_normalized | tlp_gap | -0.00123789 | [-0.00247081,-0.00041037] | 3 |
| restored_detail_minus_fixed_average/direct_mse | psnr_rgb | -0.00460107 | [-0.02865461,+0.01093960] | 0 |
| restored_detail_minus_fixed_average/direct_mse | psnr_y | -0.00106658 | [-0.02901506,+0.01621438] | 1 |
| restored_detail_minus_fixed_average/direct_mse | lpips | -0.00621845 | [-0.00925003,-0.00347902] | 3 |
| restored_detail_minus_fixed_average/direct_mse | tlp_gap | -0.00068202 | [-0.00157699,+0.00003302] | 3 |
| restored_detail_minus_fixed_average/direct_mse_normalized | psnr_rgb | -0.00381553 | [-0.02725527,+0.01163530] | 0 |
| restored_detail_minus_fixed_average/direct_mse_normalized | psnr_y | -0.00002553 | [-0.02734340,+0.01703743] | 1 |
| restored_detail_minus_fixed_average/direct_mse_normalized | lpips | -0.00635624 | [-0.00934117,-0.00365881] | 3 |
| restored_detail_minus_fixed_average/direct_mse_normalized | tlp_gap | -0.00063644 | [-0.00154060,+0.00008045] | 3 |
| restored_detail_minus_initial_thirds/direct_mse | psnr_rgb | -0.01506827 | [-0.04559739,+0.00605143] | 0 |
| restored_detail_minus_initial_thirds/direct_mse | psnr_y | -0.01479467 | [-0.04951123,+0.00860703] | 0 |
| restored_detail_minus_initial_thirds/direct_mse | lpips | -0.00324708 | [-0.01055844,+0.00146715] | 3 |
| restored_detail_minus_initial_thirds/direct_mse | tlp_gap | -0.00075224 | [-0.00202321,+0.00035208] | 3 |
| restored_detail_minus_initial_thirds/direct_mse_normalized | psnr_rgb | -0.01428273 | [-0.04404347,+0.00635132] | 0 |
| restored_detail_minus_initial_thirds/direct_mse_normalized | psnr_y | -0.01375362 | [-0.04763349,+0.00892877] | 0 |
| restored_detail_minus_initial_thirds/direct_mse_normalized | lpips | -0.00338487 | [-0.01059758,+0.00128603] | 3 |
| restored_detail_minus_initial_thirds/direct_mse_normalized | tlp_gap | -0.00070666 | [-0.00200304,+0.00039651] | 3 |
| raw_detail_minus_fixed_average/direct_mse | psnr_rgb | +0.09831916 | [+0.04703186,+0.16110248] | 3 |
| raw_detail_minus_fixed_average/direct_mse | psnr_y | +0.08343163 | [+0.02616751,+0.16423333] | 3 |
| raw_detail_minus_fixed_average/direct_mse | lpips | -0.00528019 | [-0.02067226,+0.01126792] | 3 |
| raw_detail_minus_fixed_average/direct_mse | tlp_gap | -0.01638452 | [-0.02086110,-0.01241680] | 3 |
| raw_detail_minus_fixed_average/direct_mse_normalized | psnr_rgb | +0.09781496 | [+0.04539982,+0.16147937] | 3 |
| raw_detail_minus_fixed_average/direct_mse_normalized | psnr_y | +0.08223228 | [+0.02398729,+0.16429197] | 3 |
| raw_detail_minus_fixed_average/direct_mse_normalized | lpips | -0.00463117 | [-0.02043315,+0.01236530] | 3 |
| raw_detail_minus_fixed_average/direct_mse_normalized | tlp_gap | -0.01673609 | [-0.02119106,-0.01269690] | 3 |
| raw_detail_minus_initial_thirds/direct_mse | psnr_rgb | +0.00628702 | [-0.01681016,+0.02655133] | 3 |
| raw_detail_minus_initial_thirds/direct_mse | psnr_y | -0.00435986 | [-0.03260358,+0.02334449] | 0 |
| raw_detail_minus_initial_thirds/direct_mse | lpips | +0.01471955 | [+0.00275980,+0.02805020] | 0 |
| raw_detail_minus_initial_thirds/direct_mse | tlp_gap | -0.00642006 | [-0.00896416,-0.00419739] | 3 |
| raw_detail_minus_initial_thirds/direct_mse_normalized | psnr_rgb | +0.00578282 | [-0.01745447,+0.02640784] | 3 |
| raw_detail_minus_initial_thirds/direct_mse_normalized | psnr_y | -0.00555921 | [-0.03363624,+0.02256842] | 0 |
| raw_detail_minus_initial_thirds/direct_mse_normalized | lpips | +0.01536856 | [+0.00300551,+0.02927003] | 0 |
| raw_detail_minus_initial_thirds/direct_mse_normalized | tlp_gap | -0.00677163 | [-0.00918735,-0.00450853] | 3 |

Average three selector effects per sequence, then resample eight complete two-condition scenes,10k,RNG0; unadjusted percentile95; conditional on fitted models

No target training/normalization or oracle. BVI VAL informed prior development; no target-blind-history or SOTA claim.
