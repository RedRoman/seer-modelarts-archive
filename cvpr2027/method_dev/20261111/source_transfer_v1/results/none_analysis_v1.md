# Source-trained temporal correction on BVI VAL16

Exploratory SDSD-only restoration/selector training to BVI VAL16; one base seed0, three selector seeds

| Arm | PSNR-RGB | PSNR-Y | LPIPS | tLP gap |
|---|---:|---:|---:|---:|
| base | 17.36560231 | 18.54647743 | 0.26487936 | 0.00586282 |
| restored/fixed_average | 17.34925625 | 18.50539520 | 0.27958041 | 0.00575659 |
| restored/initial_thirds | 17.36918794 | 18.53810443 | 0.27315078 | 0.00649936 |
| restored_detail/fixed_average | 17.35328728 | 18.51882175 | 0.27890700 | 0.00549895 |
| restored_detail/initial_thirds | 17.36324102 | 18.53581506 | 0.27580483 | 0.00523943 |
| raw_detail/fixed_average | 17.38518050 | 18.65452281 | 0.23143136 | 0.02732516 |
| raw_detail/initial_thirds | 17.50619825 | 18.80985353 | 0.20436676 | 0.01653130 |
| restored/direct_mse | 17.30576399 | 18.45296668 | 0.27292250 | 0.00508780 |
| restored/direct_mse_normalized | 17.30670803 | 18.45322193 | 0.27278705 | 0.00514033 |
| restored_detail/direct_mse | 17.35134802 | 18.51956594 | 0.27217335 | 0.00443335 |
| restored_detail/direct_mse_normalized | 17.35233391 | 18.52079960 | 0.27209687 | 0.00443189 |
| raw_detail/direct_mse | 17.52005493 | 18.79900186 | 0.20481640 | 0.00856639 |
| raw_detail/direct_mse_normalized | 17.52012758 | 18.79914099 | 0.20497993 | 0.00856094 |

| Paired contrast | Metric | Mean difference | Scene 95% CI | Favorable seeds / 3 |
|---|---|---:|---|---:|
| restored_minus_base/direct_mse | psnr_rgb | -0.05983831 | [-0.13918969,-0.00643533] | 0 |
| restored_minus_base/direct_mse | psnr_y | -0.09351075 | [-0.19356988,-0.02695199] | 0 |
| restored_minus_base/direct_mse | lpips | +0.00804314 | [-0.00633631,+0.01938205] | 0 |
| restored_minus_base/direct_mse | tlp_gap | -0.00077503 | [-0.00307886,+0.00156092] | 3 |
| restored_minus_base/direct_mse_normalized | psnr_rgb | -0.05889428 | [-0.14081086,-0.00496243] | 0 |
| restored_minus_base/direct_mse_normalized | psnr_y | -0.09325550 | [-0.19559138,-0.02631655] | 0 |
| restored_minus_base/direct_mse_normalized | lpips | +0.00790769 | [-0.00645421,+0.01923591] | 0 |
| restored_minus_base/direct_mse_normalized | tlp_gap | -0.00072249 | [-0.00297057,+0.00156068] | 3 |
| restored_detail_minus_base/direct_mse | psnr_rgb | -0.01425429 | [-0.06274625,+0.03129233] | 0 |
| restored_detail_minus_base/direct_mse | psnr_y | -0.02691149 | [-0.08890890,+0.02185347] | 0 |
| restored_detail_minus_base/direct_mse | lpips | +0.00729399 | [-0.00911096,+0.01979749] | 0 |
| restored_detail_minus_base/direct_mse | tlp_gap | -0.00142947 | [-0.00342592,+0.00059564] | 3 |
| restored_detail_minus_base/direct_mse_normalized | psnr_rgb | -0.01326839 | [-0.06074779,+0.03175948] | 0 |
| restored_detail_minus_base/direct_mse_normalized | psnr_y | -0.02567783 | [-0.08606396,+0.02237347] | 0 |
| restored_detail_minus_base/direct_mse_normalized | lpips | +0.00721751 | [-0.00915308,+0.01967312] | 0 |
| restored_detail_minus_base/direct_mse_normalized | tlp_gap | -0.00143094 | [-0.00342960,+0.00059007] | 3 |
| raw_detail_minus_base/direct_mse | psnr_rgb | +0.15445262 | [+0.07193826,+0.28485306] | 3 |
| raw_detail_minus_base/direct_mse | psnr_y | +0.25252443 | [+0.10412010,+0.43456015] | 3 |
| raw_detail_minus_base/direct_mse | lpips | -0.06006296 | [-0.07098903,-0.04955942] | 3 |
| raw_detail_minus_base/direct_mse | tlp_gap | +0.00270357 | [-0.00006032,+0.00540836] | 0 |
| raw_detail_minus_base/direct_mse_normalized | psnr_rgb | +0.15452527 | [+0.07206692,+0.28465308] | 3 |
| raw_detail_minus_base/direct_mse_normalized | psnr_y | +0.25266356 | [+0.10442692,+0.43433162] | 3 |
| raw_detail_minus_base/direct_mse_normalized | lpips | -0.05989943 | [-0.07074253,-0.04941191] | 3 |
| raw_detail_minus_base/direct_mse_normalized | tlp_gap | +0.00269812 | [-0.00006248,+0.00540730] | 0 |
| restored_detail_minus_restored/direct_mse | psnr_rgb | +0.04558403 | [+0.01679303,+0.08288789] | 3 |
| restored_detail_minus_restored/direct_mse | psnr_y | +0.06659926 | [+0.02874574,+0.11203994] | 3 |
| restored_detail_minus_restored/direct_mse | lpips | -0.00074915 | [-0.00358614,+0.00179572] | 3 |
| restored_detail_minus_restored/direct_mse | tlp_gap | -0.00065445 | [-0.00128843,-0.00000742] | 3 |
| restored_detail_minus_restored/direct_mse_normalized | psnr_rgb | +0.04562589 | [+0.01602808,+0.08612552] | 3 |
| restored_detail_minus_restored/direct_mse_normalized | psnr_y | +0.06757767 | [+0.02918625,+0.11714705] | 3 |
| restored_detail_minus_restored/direct_mse_normalized | lpips | -0.00069018 | [-0.00350805,+0.00180616] | 3 |
| restored_detail_minus_restored/direct_mse_normalized | tlp_gap | -0.00070845 | [-0.00133685,-0.00008530] | 3 |
| raw_detail_minus_restored/direct_mse | psnr_rgb | +0.21429094 | [+0.09086637,+0.37161761] | 3 |
| raw_detail_minus_restored/direct_mse | psnr_y | +0.34603518 | [+0.16839271,+0.55560132] | 3 |
| raw_detail_minus_restored/direct_mse | lpips | -0.06810610 | [-0.08845867,-0.04551265] | 3 |
| raw_detail_minus_restored/direct_mse | tlp_gap | +0.00347860 | [-0.00085673,+0.00767253] | 0 |
| raw_detail_minus_restored/direct_mse_normalized | psnr_rgb | +0.21341955 | [+0.08868076,+0.37301523] | 3 |
| raw_detail_minus_restored/direct_mse_normalized | psnr_y | +0.34591906 | [+0.16739476,+0.55761858] | 3 |
| raw_detail_minus_restored/direct_mse_normalized | lpips | -0.06780712 | [-0.08811046,-0.04524594] | 3 |
| raw_detail_minus_restored/direct_mse_normalized | tlp_gap | +0.00342061 | [-0.00087953,+0.00760582] | 0 |
| raw_detail_minus_restored_detail/direct_mse | psnr_rgb | +0.16870691 | [+0.05548724,+0.32308419] | 3 |
| raw_detail_minus_restored_detail/direct_mse | psnr_y | +0.27943592 | [+0.10878465,+0.48438285] | 3 |
| raw_detail_minus_restored_detail/direct_mse | lpips | -0.06735695 | [-0.08931853,-0.04240026] | 3 |
| raw_detail_minus_restored_detail/direct_mse | tlp_gap | +0.00413304 | [-0.00002004,+0.00820840] | 0 |
| raw_detail_minus_restored_detail/direct_mse_normalized | psnr_rgb | +0.16779366 | [+0.05542469,+0.32164994] | 3 |
| raw_detail_minus_restored_detail/direct_mse_normalized | psnr_y | +0.27834139 | [+0.10790911,+0.48208889] | 3 |
| raw_detail_minus_restored_detail/direct_mse_normalized | lpips | -0.06711694 | [-0.08900968,-0.04221596] | 3 |
| raw_detail_minus_restored_detail/direct_mse_normalized | tlp_gap | +0.00412906 | [-0.00005094,+0.00822596] | 0 |
| restored_minus_fixed_average/direct_mse | psnr_rgb | -0.04349225 | [-0.09670753,-0.00380872] | 0 |
| restored_minus_fixed_average/direct_mse | psnr_y | -0.05242852 | [-0.12169491,+0.00252398] | 0 |
| restored_minus_fixed_average/direct_mse | lpips | -0.00665791 | [-0.00914237,-0.00422245] | 3 |
| restored_minus_fixed_average/direct_mse | tlp_gap | -0.00066879 | [-0.00263341,+0.00059625] | 3 |
| restored_minus_fixed_average/direct_mse_normalized | psnr_rgb | -0.04254822 | [-0.09773067,-0.00259735] | 0 |
| restored_minus_fixed_average/direct_mse_normalized | psnr_y | -0.05217327 | [-0.12388575,+0.00323284] | 0 |
| restored_minus_fixed_average/direct_mse_normalized | lpips | -0.00679336 | [-0.00925380,-0.00438042] | 3 |
| restored_minus_fixed_average/direct_mse_normalized | tlp_gap | -0.00061625 | [-0.00249120,+0.00060510] | 3 |
| restored_minus_initial_thirds/direct_mse | psnr_rgb | -0.06342395 | [-0.13250669,-0.01924479] | 0 |
| restored_minus_initial_thirds/direct_mse | psnr_y | -0.08513775 | [-0.17223092,-0.02387585] | 0 |
| restored_minus_initial_thirds/direct_mse | lpips | -0.00022828 | [-0.00437754,+0.00305836] | 2 |
| restored_minus_initial_thirds/direct_mse | tlp_gap | -0.00141157 | [-0.00370291,+0.00004207] | 3 |
| restored_minus_initial_thirds/direct_mse_normalized | psnr_rgb | -0.06247992 | [-0.13337227,-0.01812782] | 0 |
| restored_minus_initial_thirds/direct_mse_normalized | psnr_y | -0.08488250 | [-0.17433500,-0.02312671] | 0 |
| restored_minus_initial_thirds/direct_mse_normalized | lpips | -0.00036373 | [-0.00447091,+0.00289911] | 2 |
| restored_minus_initial_thirds/direct_mse_normalized | tlp_gap | -0.00135903 | [-0.00356308,+0.00009931] | 3 |
| restored_detail_minus_fixed_average/direct_mse | psnr_rgb | -0.00193926 | [-0.02452107,+0.01674524] | 1 |
| restored_detail_minus_fixed_average/direct_mse | psnr_y | +0.00074419 | [-0.02852667,+0.02383680] | 2 |
| restored_detail_minus_fixed_average/direct_mse | lpips | -0.00673366 | [-0.00909201,-0.00449511] | 3 |
| restored_detail_minus_fixed_average/direct_mse | tlp_gap | -0.00106560 | [-0.00248596,-0.00014719] | 3 |
| restored_detail_minus_fixed_average/direct_mse_normalized | psnr_rgb | -0.00095337 | [-0.02233624,+0.01732661] | 2 |
| restored_detail_minus_fixed_average/direct_mse_normalized | psnr_y | +0.00197785 | [-0.02602588,+0.02466420] | 2 |
| restored_detail_minus_fixed_average/direct_mse_normalized | lpips | -0.00681014 | [-0.00916526,-0.00456502] | 3 |
| restored_detail_minus_fixed_average/direct_mse_normalized | tlp_gap | -0.00106707 | [-0.00245615,-0.00016338] | 3 |
| restored_detail_minus_initial_thirds/direct_mse | psnr_rgb | -0.01189300 | [-0.04676783,+0.01645101] | 0 |
| restored_detail_minus_initial_thirds/direct_mse | psnr_y | -0.01624912 | [-0.05933613,+0.01549629] | 0 |
| restored_detail_minus_initial_thirds/direct_mse | lpips | -0.00363148 | [-0.00851154,-0.00005877] | 3 |
| restored_detail_minus_initial_thirds/direct_mse | tlp_gap | -0.00080608 | [-0.00260026,+0.00046516] | 3 |
| restored_detail_minus_initial_thirds/direct_mse_normalized | psnr_rgb | -0.01090711 | [-0.04463544,+0.01694633] | 0 |
| restored_detail_minus_initial_thirds/direct_mse_normalized | psnr_y | -0.01501546 | [-0.05659102,+0.01598609] | 0 |
| restored_detail_minus_initial_thirds/direct_mse_normalized | lpips | -0.00370796 | [-0.00853061,-0.00021220] | 3 |
| restored_detail_minus_initial_thirds/direct_mse_normalized | tlp_gap | -0.00080755 | [-0.00256422,+0.00043982] | 3 |
| raw_detail_minus_fixed_average/direct_mse | psnr_rgb | +0.13487443 | [+0.06200219,+0.23391022] | 3 |
| raw_detail_minus_fixed_average/direct_mse | psnr_y | +0.14447905 | [+0.02623218,+0.29405288] | 3 |
| raw_detail_minus_fixed_average/direct_mse | lpips | -0.02661496 | [-0.03996599,-0.01222743] | 3 |
| raw_detail_minus_fixed_average/direct_mse | tlp_gap | -0.01875876 | [-0.02431566,-0.01360070] | 3 |
| raw_detail_minus_fixed_average/direct_mse_normalized | psnr_rgb | +0.13494708 | [+0.06205180,+0.23402732] | 3 |
| raw_detail_minus_fixed_average/direct_mse_normalized | psnr_y | +0.14461818 | [+0.02622586,+0.29430853] | 3 |
| raw_detail_minus_fixed_average/direct_mse_normalized | lpips | -0.02645143 | [-0.03990069,-0.01195137] | 3 |
| raw_detail_minus_fixed_average/direct_mse_normalized | tlp_gap | -0.01876421 | [-0.02430918,-0.01359706] | 3 |
| raw_detail_minus_initial_thirds/direct_mse | psnr_rgb | +0.01385668 | [-0.02203862,+0.04084578] | 3 |
| raw_detail_minus_initial_thirds/direct_mse | psnr_y | -0.01085167 | [-0.06747467,+0.03748962] | 0 |
| raw_detail_minus_initial_thirds/direct_mse | lpips | +0.00044964 | [-0.00905797,+0.01150995] | 1 |
| raw_detail_minus_initial_thirds/direct_mse | tlp_gap | -0.00796490 | [-0.01157373,-0.00455482] | 3 |
| raw_detail_minus_initial_thirds/direct_mse_normalized | psnr_rgb | +0.01392932 | [-0.02216425,+0.04100927] | 3 |
| raw_detail_minus_initial_thirds/direct_mse_normalized | psnr_y | -0.01071254 | [-0.06768348,+0.03779615] | 0 |
| raw_detail_minus_initial_thirds/direct_mse_normalized | lpips | +0.00061317 | [-0.00895844,+0.01174481] | 1 |
| raw_detail_minus_initial_thirds/direct_mse_normalized | tlp_gap | -0.00797035 | [-0.01156976,-0.00458364] | 3 |

Average three selector effects per sequence, then resample eight complete two-condition scenes,10k,RNG0; unadjusted percentile95; conditional on fitted models

No target training/normalization or oracle. BVI VAL informed prior development; no target-blind-history or SOTA claim.
