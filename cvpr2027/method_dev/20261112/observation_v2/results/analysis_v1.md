# Observation-v2 complete paired analysis

Exploratory source SDSD136 training to BVI VAL16; all prespecified methods/contrasts

Differences are left minus right; PSNR higher is favorable, LPIPS and absolute tLP gap lower is favorable.

## pretrained

| Method | PSNR-RGB | PSNR-Y | LPIPS | tLP gap |
|---|---:|---:|---:|---:|
| base | 18.35735052 | 19.28493130 | 0.24482940 | 0.00523702 |
| raw_detail/initial_thirds | 18.42674663 | 19.38691180 | 0.18783557 | 0.01327192 |
| raw_detail/fixed_average | 18.33472549 | 19.29913784 | 0.20783597 | 0.02323722 |
| same_frame_detail/initial_thirds | 18.36862716 | 19.34535776 | 0.22964612 | 0.02691876 |
| same_frame_detail/fixed_average | 18.19290308 | 19.17463408 | 0.26380640 | 0.03760520 |
| mismatched_detail/initial_thirds | 18.00485464 | 18.86585063 | 0.28402428 | 0.01195084 |
| mismatched_detail/fixed_average | 17.62569947 | 18.42918307 | 0.30333144 | 0.02205601 |
| raw_node | 18.42265527 | 19.36563181 | 0.21120790 | 0.00542664 |
| same_frame_node | 18.40525892 | 19.34287027 | 0.22890164 | 0.00803964 |
| mismatched_node | 18.35108985 | 19.27192712 | 0.24626589 | 0.00491753 |
| trajectory | 18.42261354 | 19.35857752 | 0.22850431 | 0.00402833 |
| trajectory_fixed_edge | 18.42641734 | 19.36419709 | 0.22276145 | 0.00399281 |
| trajectory_no_cross | 18.42278354 | 19.35881743 | 0.22818118 | 0.00402926 |
| trajectory_scrambled_cross | 18.42256304 | 19.35852791 | 0.22852606 | 0.00402908 |
| v1_raw_direct | 18.43302902 | 19.38255152 | 0.20255610 | 0.00685283 |

| Contrast | Metric | Difference | Scene95%CI | Favorable head seeds /3 |
|---|---|---:|---|---:|
| raw_node_minus_same_frame_node | psnr_rgb | +0.01739635 | [+0.00472015,+0.02993546] | 3 |
| raw_node_minus_same_frame_node | psnr_y | +0.02276153 | [+0.00332942,+0.04429086] | 3 |
| raw_node_minus_same_frame_node | lpips | -0.01769374 | [-0.02576503,-0.01067754] | 3 |
| raw_node_minus_same_frame_node | tlp_gap | -0.00261300 | [-0.00465382,-0.00089845] | 3 |
| raw_node_minus_mismatched_node | psnr_rgb | +0.07156542 | [+0.02944891,+0.14274727] | 3 |
| raw_node_minus_mismatched_node | psnr_y | +0.09370468 | [+0.03567800,+0.17484584] | 3 |
| raw_node_minus_mismatched_node | lpips | -0.03505799 | [-0.04589795,-0.02415703] | 3 |
| raw_node_minus_mismatched_node | tlp_gap | +0.00050911 | [-0.00011978,+0.00105688] | 0 |
| raw_node_minus_base | psnr_rgb | +0.06530476 | [+0.02399307,+0.13131149] | 3 |
| raw_node_minus_base | psnr_y | +0.08070051 | [+0.02794567,+0.15601959] | 3 |
| raw_node_minus_base | lpips | -0.03362150 | [-0.04310272,-0.02403353] | 3 |
| raw_node_minus_base | tlp_gap | +0.00018962 | [-0.00064719,+0.00097725] | 1 |
| raw_node_minus_raw_detail/initial_thirds | psnr_rgb | -0.00409136 | [-0.03242469,+0.02247720] | 1 |
| raw_node_minus_raw_detail/initial_thirds | psnr_y | -0.02128000 | [-0.05487205,+0.01398213] | 0 |
| raw_node_minus_raw_detail/initial_thirds | lpips | +0.02337233 | [+0.00790185,+0.04259289] | 0 |
| raw_node_minus_raw_detail/initial_thirds | tlp_gap | -0.00784528 | [-0.01041176,-0.00535620] | 3 |
| raw_node_minus_v1_raw_direct | psnr_rgb | -0.01037375 | [-0.02189480,-0.00245270] | 0 |
| raw_node_minus_v1_raw_direct | psnr_y | -0.01691972 | [-0.02877606,-0.00639645] | 0 |
| raw_node_minus_v1_raw_direct | lpips | +0.00865180 | [+0.00337072,+0.01709388] | 0 |
| raw_node_minus_v1_raw_direct | tlp_gap | -0.00142619 | [-0.00227918,-0.00055546] | 3 |
| trajectory_minus_raw_node | psnr_rgb | -0.00004174 | [-0.01873164,+0.01824363] | 1 |
| trajectory_minus_raw_node | psnr_y | -0.00705429 | [-0.02685444,+0.01283480] | 0 |
| trajectory_minus_raw_node | lpips | +0.01729641 | [-0.00754704,+0.03503816] | 0 |
| trajectory_minus_raw_node | tlp_gap | -0.00139831 | [-0.00383381,+0.00099746] | 3 |
| trajectory_minus_trajectory_fixed_edge | psnr_rgb | -0.00380380 | [-0.00849130,+0.00063094] | 0 |
| trajectory_minus_trajectory_fixed_edge | psnr_y | -0.00561957 | [-0.01054935,-0.00086261] | 0 |
| trajectory_minus_trajectory_fixed_edge | lpips | +0.00574286 | [+0.00185256,+0.00879521] | 0 |
| trajectory_minus_trajectory_fixed_edge | tlp_gap | +0.00003552 | [-0.00006094,+0.00014646] | 0 |
| trajectory_minus_trajectory_no_cross | psnr_rgb | -0.00017000 | [-0.00036104,+0.00005599] | 1 |
| trajectory_minus_trajectory_no_cross | psnr_y | -0.00023991 | [-0.00043920,-0.00000476] | 1 |
| trajectory_minus_trajectory_no_cross | lpips | +0.00032312 | [+0.00010772,+0.00050084] | 0 |
| trajectory_minus_trajectory_no_cross | tlp_gap | -0.00000094 | [-0.00000854,+0.00000427] | 2 |
| trajectory_minus_trajectory_scrambled_cross | psnr_rgb | +0.00005050 | [+0.00000620,+0.00010257] | 3 |
| trajectory_minus_trajectory_scrambled_cross | psnr_y | +0.00004960 | [-0.00000632,+0.00011156] | 3 |
| trajectory_minus_trajectory_scrambled_cross | lpips | -0.00002176 | [-0.00003363,-0.00000933] | 3 |
| trajectory_minus_trajectory_scrambled_cross | tlp_gap | -0.00000075 | [-0.00000444,+0.00000271] | 3 |
| trajectory_minus_base | psnr_rgb | +0.06526302 | [+0.03324067,+0.11653815] | 3 |
| trajectory_minus_base | psnr_y | +0.07364622 | [+0.03120765,+0.13341376] | 3 |
| trajectory_minus_base | lpips | -0.01632509 | [-0.03256473,-0.00457058] | 3 |
| trajectory_minus_base | tlp_gap | -0.00120869 | [-0.00327765,+0.00068871] | 3 |
| trajectory_minus_raw_detail/initial_thirds | psnr_rgb | -0.00413309 | [-0.04166820,+0.02263947] | 1 |
| trajectory_minus_raw_detail/initial_thirds | psnr_y | -0.02833429 | [-0.07262191,+0.00884278] | 0 |
| trajectory_minus_raw_detail/initial_thirds | lpips | +0.04066873 | [+0.02298385,+0.06124858] | 0 |
| trajectory_minus_raw_detail/initial_thirds | tlp_gap | -0.00924359 | [-0.01391648,-0.00466014] | 3 |
| trajectory_minus_v1_raw_direct | psnr_rgb | -0.01041548 | [-0.02665255,+0.00067060] | 0 |
| trajectory_minus_v1_raw_direct | psnr_y | -0.02397401 | [-0.04194498,-0.00879740] | 0 |
| trajectory_minus_v1_raw_direct | lpips | +0.02594821 | [+0.00915016,+0.03969484] | 0 |
| trajectory_minus_v1_raw_direct | tlp_gap | -0.00282450 | [-0.00590400,-0.00006620] | 3 |

## none

| Method | PSNR-RGB | PSNR-Y | LPIPS | tLP gap |
|---|---:|---:|---:|---:|
| base | 17.36560231 | 18.54647743 | 0.26487936 | 0.00586282 |
| raw_detail/initial_thirds | 17.50620913 | 18.80986344 | 0.20436239 | 0.01653206 |
| raw_detail/fixed_average | 17.38519119 | 18.65452733 | 0.23142807 | 0.02732216 |
| same_frame_detail/initial_thirds | 17.43499428 | 18.74408977 | 0.25052120 | 0.03066347 |
| same_frame_detail/fixed_average | 17.20149037 | 18.44801880 | 0.28939453 | 0.04233167 |
| mismatched_detail/initial_thirds | 16.97339218 | 18.03213902 | 0.30271259 | 0.01403767 |
| mismatched_detail/fixed_average | 16.58236242 | 17.52947402 | 0.32735419 | 0.02545646 |
| raw_node | 17.49872061 | 18.76055028 | 0.21323357 | 0.00666228 |
| same_frame_node | 17.48438872 | 18.73180656 | 0.23061363 | 0.01111647 |
| mismatched_node | 17.34108609 | 18.51884580 | 0.26546288 | 0.00553814 |
| trajectory | 17.51055614 | 18.76725400 | 0.22804982 | 0.00405616 |
| trajectory_fixed_edge | 17.51336029 | 18.77400306 | 0.22298649 | 0.00401270 |
| trajectory_no_cross | 17.51067690 | 18.76751983 | 0.22779623 | 0.00405334 |
| trajectory_scrambled_cross | 17.51059044 | 18.76730194 | 0.22804402 | 0.00405691 |
| v1_raw_direct | 17.52006034 | 18.79900466 | 0.20481500 | 0.00856504 |

| Contrast | Metric | Difference | Scene95%CI | Favorable head seeds /3 |
|---|---|---:|---|---:|
| raw_node_minus_same_frame_node | psnr_rgb | +0.01433189 | [+0.00513490,+0.02344420] | 3 |
| raw_node_minus_same_frame_node | psnr_y | +0.02874371 | [-0.00061853,+0.06884336] | 3 |
| raw_node_minus_same_frame_node | lpips | -0.01738006 | [-0.02595965,-0.00947627] | 3 |
| raw_node_minus_same_frame_node | tlp_gap | -0.00445419 | [-0.00656222,-0.00264089] | 3 |
| raw_node_minus_mismatched_node | psnr_rgb | +0.15763452 | [+0.05495197,+0.30918597] | 3 |
| raw_node_minus_mismatched_node | psnr_y | +0.24170448 | [+0.10146486,+0.43652385] | 3 |
| raw_node_minus_mismatched_node | lpips | -0.05222931 | [-0.06825686,-0.03637285] | 3 |
| raw_node_minus_mismatched_node | tlp_gap | +0.00112414 | [+0.00036496,+0.00205303] | 0 |
| raw_node_minus_base | psnr_rgb | +0.13311830 | [+0.05163689,+0.25824862] | 3 |
| raw_node_minus_base | psnr_y | +0.21407285 | [+0.07987780,+0.38174382] | 3 |
| raw_node_minus_base | lpips | -0.05164579 | [-0.06515665,-0.03772514] | 3 |
| raw_node_minus_base | tlp_gap | +0.00079946 | [-0.00125099,+0.00297247] | 0 |
| raw_node_minus_raw_detail/initial_thirds | psnr_rgb | -0.00748852 | [-0.05480236,+0.03270114] | 1 |
| raw_node_minus_raw_detail/initial_thirds | psnr_y | -0.04931317 | [-0.12636135,+0.02089071] | 0 |
| raw_node_minus_raw_detail/initial_thirds | lpips | +0.00887118 | [-0.00164662,+0.02038346] | 0 |
| raw_node_minus_raw_detail/initial_thirds | tlp_gap | -0.00986978 | [-0.01365000,-0.00620013] | 3 |
| raw_node_minus_v1_raw_direct | psnr_rgb | -0.02133973 | [-0.04484073,-0.00420491] | 0 |
| raw_node_minus_v1_raw_direct | psnr_y | -0.03845438 | [-0.07105473,-0.01120500] | 0 |
| raw_node_minus_v1_raw_direct | lpips | +0.00841857 | [+0.00493401,+0.01338881] | 0 |
| raw_node_minus_v1_raw_direct | tlp_gap | -0.00190276 | [-0.00298373,-0.00078386] | 3 |
| trajectory_minus_raw_node | psnr_rgb | +0.01183553 | [-0.01099361,+0.03941416] | 3 |
| trajectory_minus_raw_node | psnr_y | +0.00670372 | [-0.02164121,+0.03468856] | 3 |
| trajectory_minus_raw_node | lpips | +0.01481626 | [+0.00118632,+0.02597051] | 0 |
| trajectory_minus_raw_node | tlp_gap | -0.00260612 | [-0.00568099,+0.00051729] | 3 |
| trajectory_minus_trajectory_fixed_edge | psnr_rgb | -0.00280414 | [-0.00989511,+0.00490711] | 0 |
| trajectory_minus_trajectory_fixed_edge | psnr_y | -0.00674906 | [-0.01517397,+0.00223604] | 0 |
| trajectory_minus_trajectory_fixed_edge | lpips | +0.00506333 | [+0.00177459,+0.00767056] | 0 |
| trajectory_minus_trajectory_fixed_edge | tlp_gap | +0.00004346 | [-0.00008347,+0.00018793] | 0 |
| trajectory_minus_trajectory_no_cross | psnr_rgb | -0.00012075 | [-0.00044734,+0.00027632] | 0 |
| trajectory_minus_trajectory_no_cross | psnr_y | -0.00026583 | [-0.00063881,+0.00014438] | 0 |
| trajectory_minus_trajectory_no_cross | lpips | +0.00025360 | [+0.00008345,+0.00039666] | 0 |
| trajectory_minus_trajectory_no_cross | tlp_gap | +0.00000282 | [-0.00000606,+0.00001071] | 1 |
| trajectory_minus_trajectory_scrambled_cross | psnr_rgb | -0.00003429 | [-0.00009151,+0.00001145] | 1 |
| trajectory_minus_trajectory_scrambled_cross | psnr_y | -0.00004795 | [-0.00013097,+0.00001607] | 1 |
| trajectory_minus_trajectory_scrambled_cross | lpips | +0.00000580 | [-0.00000941,+0.00002268] | 1 |
| trajectory_minus_trajectory_scrambled_cross | tlp_gap | -0.00000075 | [-0.00000535,+0.00000340] | 2 |
| trajectory_minus_base | psnr_rgb | +0.14495384 | [+0.07438904,+0.25504340] | 3 |
| trajectory_minus_base | psnr_y | +0.22077657 | [+0.09789115,+0.36985901] | 3 |
| trajectory_minus_base | lpips | -0.03682953 | [-0.04440172,-0.02906560] | 3 |
| trajectory_minus_base | tlp_gap | -0.00180666 | [-0.00368365,-0.00012023] | 3 |
| trajectory_minus_raw_detail/initial_thirds | psnr_rgb | +0.00434701 | [-0.05308566,+0.04346848] | 2 |
| trajectory_minus_raw_detail/initial_thirds | psnr_y | -0.04260945 | [-0.13067048,+0.02836082] | 0 |
| trajectory_minus_raw_detail/initial_thirds | lpips | +0.02368744 | [+0.00983693,+0.04017214] | 0 |
| trajectory_minus_raw_detail/initial_thirds | tlp_gap | -0.01247590 | [-0.01887308,-0.00603662] | 3 |
| trajectory_minus_v1_raw_direct | psnr_rgb | -0.00950420 | [-0.03250219,+0.00679436] | 1 |
| trajectory_minus_v1_raw_direct | psnr_y | -0.03175066 | [-0.06543598,-0.00333084] | 0 |
| trajectory_minus_v1_raw_direct | lpips | +0.02323482 | [+0.01409083,+0.03126841] | 0 |
| trajectory_minus_v1_raw_direct | tlp_gap | -0.00450888 | [-0.00838197,-0.00060552] | 3 |

## Between historical base arms

These comparisons do not isolate a causal video-prior effect.

| Difference of arm values/effects | Metric | Difference | Scene95%CI |
|---|---|---:|---|
| absolute_pretrained_minus_none/base | psnr_rgb | +0.99174821 | [-0.36401462,+2.18102621] |
| absolute_pretrained_minus_none/base | psnr_y | +0.73845387 | [-0.86712825,+2.20801893] |
| absolute_pretrained_minus_none/base | lpips | -0.02004996 | [-0.03822804,-0.00084075] |
| absolute_pretrained_minus_none/base | tlp_gap | -0.00062581 | [-0.00153401,+0.00027390] |
| absolute_pretrained_minus_none/raw_detail/initial_thirds | psnr_rgb | +0.92053750 | [-0.47408667,+2.15041014] |
| absolute_pretrained_minus_none/raw_detail/initial_thirds | psnr_y | +0.57704836 | [-1.10741209,+2.16088042] |
| absolute_pretrained_minus_none/raw_detail/initial_thirds | lpips | -0.01652682 | [-0.03186844,+0.00094975] |
| absolute_pretrained_minus_none/raw_detail/initial_thirds | tlp_gap | -0.00326014 | [-0.00560375,-0.00088667] |
| absolute_pretrained_minus_none/raw_detail/fixed_average | psnr_rgb | +0.94953430 | [-0.41245883,+2.16786611] |
| absolute_pretrained_minus_none/raw_detail/fixed_average | psnr_y | +0.64461050 | [-0.98868327,+2.20207059] |
| absolute_pretrained_minus_none/raw_detail/fixed_average | lpips | -0.02359210 | [-0.04091159,-0.00325782] |
| absolute_pretrained_minus_none/raw_detail/fixed_average | tlp_gap | -0.00408494 | [-0.00668861,-0.00115761] |
| absolute_pretrained_minus_none/same_frame_detail/initial_thirds | psnr_rgb | +0.93363288 | [-0.45949658,+2.15579420] |
| absolute_pretrained_minus_none/same_frame_detail/initial_thirds | psnr_y | +0.60126799 | [-1.07729460,+2.17269184] |
| absolute_pretrained_minus_none/same_frame_detail/initial_thirds | lpips | -0.02087509 | [-0.04079508,+0.00090047] |
| absolute_pretrained_minus_none/same_frame_detail/initial_thirds | tlp_gap | -0.00374471 | [-0.00674890,-0.00058186] |
| absolute_pretrained_minus_none/same_frame_detail/fixed_average | psnr_rgb | +0.99141271 | [-0.35773447,+2.18971615] |
| absolute_pretrained_minus_none/same_frame_detail/fixed_average | psnr_y | +0.72661528 | [-0.85111942,+2.22671748] |
| absolute_pretrained_minus_none/same_frame_detail/fixed_average | lpips | -0.02558813 | [-0.04619559,-0.00179913] |
| absolute_pretrained_minus_none/same_frame_detail/fixed_average | tlp_gap | -0.00472647 | [-0.00831877,-0.00076001] |
| absolute_pretrained_minus_none/mismatched_detail/initial_thirds | psnr_rgb | +1.03146247 | [-0.21408228,+2.14949200] |
| absolute_pretrained_minus_none/mismatched_detail/initial_thirds | psnr_y | +0.83371162 | [-0.60352933,+2.17578522] |
| absolute_pretrained_minus_none/mismatched_detail/initial_thirds | lpips | -0.01868831 | [-0.03735486,+0.00010358] |
| absolute_pretrained_minus_none/mismatched_detail/initial_thirds | tlp_gap | -0.00208683 | [-0.00451295,+0.00010642] |
| absolute_pretrained_minus_none/mismatched_detail/fixed_average | psnr_rgb | +1.04333705 | [-0.12583574,+2.11011054] |
| absolute_pretrained_minus_none/mismatched_detail/fixed_average | psnr_y | +0.89970905 | [-0.41334366,+2.13151202] |
| absolute_pretrained_minus_none/mismatched_detail/fixed_average | lpips | -0.02402276 | [-0.04054913,-0.00538660] |
| absolute_pretrained_minus_none/mismatched_detail/fixed_average | tlp_gap | -0.00340044 | [-0.00596948,-0.00081396] |
| absolute_pretrained_minus_none/raw_node | psnr_rgb | +0.92393466 | [-0.47117955,+2.14344211] |
| absolute_pretrained_minus_none/raw_node | psnr_y | +0.60508153 | [-1.07241156,+2.15499779] |
| absolute_pretrained_minus_none/raw_node | lpips | -0.00202567 | [-0.01789858,+0.01483353] |
| absolute_pretrained_minus_none/raw_node | tlp_gap | -0.00123564 | [-0.00233431,-0.00013676] |
| absolute_pretrained_minus_none/same_frame_node | psnr_rgb | +0.92087020 | [-0.47180627,+2.12888675] |
| absolute_pretrained_minus_none/same_frame_node | psnr_y | +0.61106371 | [-1.04868609,+2.14669383] |
| absolute_pretrained_minus_none/same_frame_node | lpips | -0.00171200 | [-0.01901456,+0.01536680] |
| absolute_pretrained_minus_none/same_frame_node | tlp_gap | -0.00307683 | [-0.00518135,-0.00109804] |
| absolute_pretrained_minus_none/mismatched_node | psnr_rgb | +1.01000376 | [-0.33107251,+2.18480563] |
| absolute_pretrained_minus_none/mismatched_node | psnr_y | +0.75308132 | [-0.84919909,+2.21079266] |
| absolute_pretrained_minus_none/mismatched_node | lpips | -0.01919699 | [-0.03480961,-0.00322537] |
| absolute_pretrained_minus_none/mismatched_node | tlp_gap | -0.00062061 | [-0.00171192,+0.00041611] |
| absolute_pretrained_minus_none/trajectory | psnr_rgb | +0.91205739 | [-0.48262945,+2.13593936] |
| absolute_pretrained_minus_none/trajectory | psnr_y | +0.59132352 | [-1.08129416,+2.13997344] |
| absolute_pretrained_minus_none/trajectory | lpips | +0.00045448 | [-0.01302725,+0.01502966] |
| absolute_pretrained_minus_none/trajectory | tlp_gap | -0.00002784 | [-0.00080823,+0.00066008] |
| absolute_pretrained_minus_none/trajectory_fixed_edge | psnr_rgb | +0.91305705 | [-0.48476669,+2.13685396] |
| absolute_pretrained_minus_none/trajectory_fixed_edge | psnr_y | +0.59019403 | [-1.08758565,+2.14213413] |
| absolute_pretrained_minus_none/trajectory_fixed_edge | lpips | -0.00022504 | [-0.01382746,+0.01454627] |
| absolute_pretrained_minus_none/trajectory_fixed_edge | tlp_gap | -0.00001989 | [-0.00080453,+0.00067133] |
| absolute_pretrained_minus_none/trajectory_no_cross | psnr_rgb | +0.91210664 | [-0.48266885,+2.13595915] |
| absolute_pretrained_minus_none/trajectory_no_cross | psnr_y | +0.59129760 | [-1.08143059,+2.14002189] |
| absolute_pretrained_minus_none/trajectory_no_cross | lpips | +0.00038495 | [-0.01310782,+0.01496742] |
| absolute_pretrained_minus_none/trajectory_no_cross | tlp_gap | -0.00002408 | [-0.00080433,+0.00066131] |
| absolute_pretrained_minus_none/trajectory_scrambled_cross | psnr_rgb | +0.91197260 | [-0.48272249,+2.13585095] |
| absolute_pretrained_minus_none/trajectory_scrambled_cross | psnr_y | +0.59122597 | [-1.08141408,+2.13988288] |
| absolute_pretrained_minus_none/trajectory_scrambled_cross | lpips | +0.00048204 | [-0.01301330,+0.01505918] |
| absolute_pretrained_minus_none/trajectory_scrambled_cross | tlp_gap | -0.00002783 | [-0.00080705,+0.00065481] |
| absolute_pretrained_minus_none/v1_raw_direct | psnr_rgb | +0.91296868 | [-0.48001272,+2.13728347] |
| absolute_pretrained_minus_none/v1_raw_direct | psnr_y | +0.58354687 | [-1.10145264,+2.14854092] |
| absolute_pretrained_minus_none/v1_raw_direct | lpips | -0.00225891 | [-0.01709899,+0.01345914] |
| absolute_pretrained_minus_none/v1_raw_direct | tlp_gap | -0.00171221 | [-0.00323779,-0.00023087] |
| contrast_effect_pretrained_minus_none/raw_node_minus_same_frame_node | psnr_rgb | +0.00306447 | [-0.01111306,+0.01937848] |
| contrast_effect_pretrained_minus_none/raw_node_minus_same_frame_node | psnr_y | -0.00598218 | [-0.04078814,+0.02594429] |
| contrast_effect_pretrained_minus_none/raw_node_minus_same_frame_node | lpips | -0.00031368 | [-0.00428994,+0.00311121] |
| contrast_effect_pretrained_minus_none/raw_node_minus_same_frame_node | tlp_gap | +0.00184119 | [+0.00034418,+0.00325853] |
| contrast_effect_pretrained_minus_none/raw_node_minus_mismatched_node | psnr_rgb | -0.08606910 | [-0.17356219,-0.01890870] |
| contrast_effect_pretrained_minus_none/raw_node_minus_mismatched_node | psnr_y | -0.14799979 | [-0.27195067,-0.04326714] |
| contrast_effect_pretrained_minus_none/raw_node_minus_mismatched_node | lpips | +0.01717132 | [+0.01093632,+0.02366779] |
| contrast_effect_pretrained_minus_none/raw_node_minus_mismatched_node | tlp_gap | -0.00061503 | [-0.00156733,+0.00015504] |
| contrast_effect_pretrained_minus_none/raw_node_minus_base | psnr_rgb | -0.06781354 | [-0.13400459,-0.01863624] |
| contrast_effect_pretrained_minus_none/raw_node_minus_base | psnr_y | -0.13337234 | [-0.24231536,-0.03993725] |
| contrast_effect_pretrained_minus_none/raw_node_minus_base | lpips | +0.01802429 | [+0.01291195,+0.02290695] |
| contrast_effect_pretrained_minus_none/raw_node_minus_base | tlp_gap | -0.00060983 | [-0.00214178,+0.00080318] |
| contrast_effect_pretrained_minus_none/raw_node_minus_raw_detail/initial_thirds | psnr_rgb | +0.00339716 | [-0.01607536,+0.02606502] |
| contrast_effect_pretrained_minus_none/raw_node_minus_raw_detail/initial_thirds | psnr_y | +0.02803317 | [-0.02268856,+0.08654264] |
| contrast_effect_pretrained_minus_none/raw_node_minus_raw_detail/initial_thirds | lpips | +0.01450115 | [+0.00733653,+0.02462234] |
| contrast_effect_pretrained_minus_none/raw_node_minus_raw_detail/initial_thirds | tlp_gap | +0.00202450 | [-0.00000839,+0.00401165] |
| contrast_effect_pretrained_minus_none/raw_node_minus_v1_raw_direct | psnr_rgb | +0.01096599 | [+0.00071707,+0.02370587] |
| contrast_effect_pretrained_minus_none/raw_node_minus_v1_raw_direct | psnr_y | +0.02153466 | [-0.00074262,+0.04800845] |
| contrast_effect_pretrained_minus_none/raw_node_minus_v1_raw_direct | lpips | +0.00023324 | [-0.00266268,+0.00416143] |
| contrast_effect_pretrained_minus_none/raw_node_minus_v1_raw_direct | tlp_gap | +0.00047657 | [-0.00049805,+0.00155653] |
| contrast_effect_pretrained_minus_none/trajectory_minus_raw_node | psnr_rgb | -0.01187727 | [-0.02187196,-0.00553439] |
| contrast_effect_pretrained_minus_none/trajectory_minus_raw_node | psnr_y | -0.01375801 | [-0.02539100,-0.00259505] |
| contrast_effect_pretrained_minus_none/trajectory_minus_raw_node | lpips | +0.00248015 | [-0.00868736,+0.01024995] |
| contrast_effect_pretrained_minus_none/trajectory_minus_raw_node | tlp_gap | +0.00120781 | [+0.00027231,+0.00205522] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_fixed_edge | psnr_rgb | -0.00099966 | [-0.00463747,+0.00197263] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_fixed_edge | psnr_y | +0.00112949 | [-0.00359470,+0.00578527] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_fixed_edge | lpips | +0.00067953 | [-0.00000753,+0.00131088] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_fixed_edge | tlp_gap | -0.00000794 | [-0.00004803,+0.00003492] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_no_cross | psnr_rgb | -0.00004925 | [-0.00022765,+0.00009541] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_no_cross | psnr_y | +0.00002592 | [-0.00017916,+0.00022436] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_no_cross | lpips | +0.00006953 | [+0.00001801,+0.00011646] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_no_cross | tlp_gap | -0.00000376 | [-0.00000825,+0.00000128] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_scrambled_cross | psnr_rgb | +0.00008479 | [+0.00002711,+0.00017903] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_scrambled_cross | psnr_y | +0.00009755 | [+0.00001352,+0.00022503] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_scrambled_cross | lpips | -0.00002756 | [-0.00004358,-0.00000850] |
| contrast_effect_pretrained_minus_none/trajectory_minus_trajectory_scrambled_cross | tlp_gap | -0.00000001 | [-0.00000493,+0.00000568] |
| contrast_effect_pretrained_minus_none/trajectory_minus_base | psnr_rgb | -0.07969082 | [-0.14561354,-0.02944921] |
| contrast_effect_pretrained_minus_none/trajectory_minus_base | psnr_y | -0.14713035 | [-0.25727266,-0.05602141] |
| contrast_effect_pretrained_minus_none/trajectory_minus_base | lpips | +0.02050444 | [+0.00483190,+0.03238256] |
| contrast_effect_pretrained_minus_none/trajectory_minus_base | tlp_gap | +0.00059797 | [-0.00050912,+0.00156119] |
| contrast_effect_pretrained_minus_none/trajectory_minus_raw_detail/initial_thirds | psnr_rgb | -0.00848011 | [-0.02738944,+0.01423660] |
| contrast_effect_pretrained_minus_none/trajectory_minus_raw_detail/initial_thirds | psnr_y | +0.01427516 | [-0.03626290,+0.07075321] |
| contrast_effect_pretrained_minus_none/trajectory_minus_raw_detail/initial_thirds | lpips | +0.01698130 | [+0.01040894,+0.02319384] |
| contrast_effect_pretrained_minus_none/trajectory_minus_raw_detail/initial_thirds | tlp_gap | +0.00323230 | [+0.00059544,+0.00581729] |
| contrast_effect_pretrained_minus_none/trajectory_minus_v1_raw_direct | psnr_rgb | -0.00091128 | [-0.00784261,+0.00697794] |
| contrast_effect_pretrained_minus_none/trajectory_minus_v1_raw_direct | psnr_y | +0.00777665 | [-0.01476202,+0.02970875] |
| contrast_effect_pretrained_minus_none/trajectory_minus_v1_raw_direct | lpips | +0.00271339 | [-0.00538907,+0.00941840] |
| contrast_effect_pretrained_minus_none/trajectory_minus_v1_raw_direct | tlp_gap | +0.00168438 | [+0.00003335,+0.00333529] |

Bootstrap: 10,000 complete-scene resamples, RNG0, seed effects averaged before resampling; unadjusted percentile95.

- One historical base per arm; three added-head seeds do not replicate base training
- Base pixel initialization/training histories differ; between-arm effects are not isolated causal video-prior evidence
- Scene bootstrap intervals are exploratory and unadjusted across the full declared comparison family
- Same-frame controls remove explicit neighbour values only at construction; base_t is all-clip
- Node-only and graph modes have different initial functions; inactive edge parameters are not equal active capacity
- V1 uses an earlier frame-batch/whole-crop training recipe; comparison is a frozen reference, not a single-mechanism isolation
- Extra adjacent RAFT and CPU graph work are real costs; no equal-latency claim
- BVI VAL informed prior project development; no target-blind-history or SOTA conclusion
