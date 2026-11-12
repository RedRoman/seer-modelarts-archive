# HVI temporal FIT feasibility v1 — completed

The prespecified seven-row diagnostic completed on all102 clips from the51
FIT scenes. No17-scene internal-development labels or target tensors were
loaded or scored. HVI's independent no-GT export and the joint-search screen
were unchanged. No parameter was fitted, lambda selected or checkpoint chosen.

All16 frames/full256 RGB were scored in float32 squared residuals with
float64 reduction. PSNR is averaged from each clip's RGB MSE; each scene has
two clips. These are **source FIT diagnostics**, not BVI uint8 PSNR-Y results.

| Fixed diagnostic | Mean RGB MSE | Mean clip RGB PSNR | PSNR difference vs HVI |
|---|---:|---:|---:|
| HVI | 0.050160672175 | 15.414470 | 0 |
| Aligned mean, lambda0.25 | 0.049050562665 | 15.657168 | +0.242699 |
| Aligned mean, lambda0.5 | 0.048437157774 | 15.802878 | +0.388408 |
| Aligned mean, lambda1 | 0.048700461727 | 15.714912 | +0.300442 |
| Same-frame identity, lambda0.25 | 0.050160672175 | 15.414470 | 0 |
| Same-frame identity, lambda0.5 | 0.050160672175 | 15.414470 | 0 |
| Same-frame identity, lambda1 | 0.050160672175 | 15.414470 | 0 |

Using the same saved Seer-flow proposals and original-mask/geometry validity,
lambda0.25 and0.5 reduce mean MSE in all51 FIT scenes; lambda1 does so in49.
Mean pixel coverage is73.993968% for two available donors,25.349840% for one,
and0.656192% for none. Same-frame rows are arithmetic identities, not matched
causal controls; all three reproduce the HVI pixels bit-for-bit.

The fixed neighboring HVI values therefore provide FIT fusion headroom for
further module development. This does not validate a learned mechanism or
cross-domain improvement. HVI itself has poor source RGB fidelity here despite
its stronger already-seen BVI replay result; calling it a universally strong
source anchor would be incorrect. No extra lambda, oracle, GT calibration,
model output, target score or learned result was added after seeing this table.

## Verification and archive

- The frozen protocol/script passed11 synthetic CPU tests before dispatch.
  The single uploaded bundle's SHA256 is
  `a8b134ce855a8673d622116354641c2ecbedb7398f6f27b3882e7dc303866aed`.
- Production scoring used CPU4 threads; the102-clip scoring section took
  58.8800s, excluding metadata/file verification. Main computation exited0;
  final task computation-process count was0. No NPU was used.
- Independent `math.fsum`/`math.log10` checks covered2170 scalar values,
  including clip PSNR, all seven means and every scene difference. Maximum
  absolute discrepancy was1.7764e-15. NumPy validity counts matched all102
  coverage records exactly; NumPy bit comparisons independently verified all
  same-frame identities without accessing GT.
- The first verifier used NumPy's implicit int64-to-float64 coordinate
  promotion and failed its boundary-coverage check. Both versions and the
  precision note are retained. Explicit float32 grids fixed the verifier;
  the frozen scoring script and original scores were not changed or rerun.
- [Complete per-clip/per-scene results](evidence/results_v1/fit_results.json),
  [execution/source lineage](evidence/results_v1/execution_manifest.json),
  [independent verification](evidence/checks_v1/independent_verification.json)
  and [verified backup receipt](evidence/checks_v1/backup_receipt.json) are
  retained as UTF-8 text. Result SHA256:
  `1984d5db1ded47f14b9bafc4d8fd8a9e3d8d4f7fc68d7b2b689fd31faf24b0e4`.
- Exclusive SFS backup:
  `/temp/zihaohe/seer_cvpr/method_dev_20261112/hvi_temporal_fit_feasibility_v1_final_v1`;
  all19 listed files/340,512 bytes were SHA-verified. Backup-manifest SHA256:
  `8dd4d3364e46032a171f66a3bfdea996231840fd247b2389d5c8128060c751db`.
  No prediction RGB or GT array is included. The downloaded389,120-byte
  artifact has SHA256
  `b2e61fb70e7f3b188278f298d3aa58a25fc2de31a0b0fcffa02557c971e3db91`;
  local extraction reverified every listed file and UTF-8 decoding.
