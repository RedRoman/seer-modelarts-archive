# Independent observation-v2 statistical verification

Completed raw JSON only; independent Python standard-library + NumPy implementation. No imports of the main analyzer or its reference auditor. No models, datasets, training, or accelerators used.

Result: PASS, 41,544 scalar comparisons; 372 raw reported means and 408 raw-direction CI endpoints included. Maximum numerical discrepancy is 4.4408920985006262e-16. All point estimates and raw means match exactly.

All 31 individual methods per arm and all 15 fixed/seed-averaged labels are retained. Coverage: 12 within-arm contrasts × 4 metrics × 2 arms plus 27 between-arm contrasts × 4 metrics. Three head-seed effects are averaged within each sequence, then each complete two-condition scene is averaged; eight whole scenes are resampled 10,000 times with RNG0. The independent implementation converts the common draws to scene multiplicities and uses matrix multiplication rather than the main analyzer’s index-gather mean.

All per-seed effects, per-sequence and per-scene effects, favorable seed counts, and benefit-oriented signs/intervals were also checked. The P/N and TempRetinex GT temporal references are exactly equal (Temp’s prespecified tolerance was 1e-6; no relaxation).

## Numerical replay drift

Old source-transfer-v1 raw outputs were compared against the new run’s base, raw_detail fixed_average/initial_thirds, and all three v1_raw_direct references. Checkpoint and base/RAFT state identity are required; numerical predictions are not required to be bitwise identical. Both base arms’ six metrics and all methods’ GT tLP exactly match the old run. The following maxima cover all six replayed methods and all 16 sequences.

| Arm | max PSNR-Y drift (dB) | max PSNR-RGB drift (dB) | max LPIPS drift | max tLP-gap drift |
|---|---:|---:|---:|---:|
| pretrained | 0.000149706120862 | 0.000101291213163 | 2.97874212265e-05 | 4.05237078667e-05 |
| none | 0.000302284113438 | 0.000330295314065 | 2.20835208893e-05 | 3.17245721817e-05 |

These small replay differences are recorded, not attributed to learning or to head seeds. Intervals are exploratory and unadjusted, conditional on the three fitted heads and one historical base per arm; no causal video-prior isolation or SOTA conclusion follows from this check.

## Reproduction

Run `python3 verify_stats.py --help` for required actual-file arguments; `--self-test` only checks synthetic algebra/coverage and does not create experimental results. `verification_complete_v1.json` records exact raw/analysis/legacy/Temp input paths and hashes, all recomputed statistics, all comparison errors, and complete replay drift vectors. Earlier `none_raw_precheck.json` and `verification_v1.json` are interim evidence; use the complete result for archiving.

Frozen verifier SHA256: `9e1db2bf76bdeb0db7a654cfe9449aa168dcd41bea0db04faad4bc030917e2b3`.
Complete result SHA256: `8f10c57cae005df74c73891c7a785abd7076eb1351ce1ae1741eec9d629a4e4a`.
