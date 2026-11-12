# Canonical RGB source repair and fixed FIT diagnostic

The source NPY convention is BGR. The official archive-member/loader audit and exact fixed-clip replay establish that the historical generic reader preserved BGR while HVI expected RGB. The corrected HVI cache now converts LQ BGR to RGB before prediction; the corrected diagnostic converts source GT BGR to RGB before scoring. Old code, predictions, flows and scores remain unchanged in their original versions. This correctness repair is not a method contribution.

## Executed result

New HVI export: all 136 source clips / 68 scenes, 174.678 seconds, strict 191-key loading, unchanged model state, all 136 new output payloads verified. Its selective tensor reader opens no GT storage entries or GT pixel values; whole-file provenance hashing does read complete opaque archive bytes. Export computes no quality scores. Cache manifest SHA256: `5f25fdfcca5bc7d581c1484245fa760ef321da642f5e741077c128376616a94a`.

The following diagnostic uses only the previously fixed 102 FIT clips / 51 scenes, with unchanged legacy Seer-native flow/validity proposals. It does not recompute flow from RGB HVI. The metric is mean per-clip full16x256 float-RGB PSNR, with float32 squared residuals accumulated in float64. All three lambdas were fixed before this replay and none was selected for deployment.

| Fixed condition | Mean clip RGB PSNR | Change vs HVI | Mean RGB MSE | Scenes with lower mean MSE |
|---|---:|---:|---:|---:|
| HVI | 15.322981 | 0 | 0.0466254751 | — |
| Temporal mean, lambda 0.25 | 15.556482 | +0.233501 | 0.0455269103 | 51/51 |
| Temporal mean, lambda 0.5 | 15.695540 | +0.372559 | 0.0449316477 | 51/51 |
| Temporal mean, lambda 1 | 15.599553 | +0.276572 | 0.0452510294 | 51/51 |
| Same-frame identity, lambda 0.25 | 15.322981 | 0 | 0.0466254751 | 0/51 |
| Same-frame identity, lambda 0.5 | 15.322981 | 0 | 0.0466254751 | 0/51 |
| Same-frame identity, lambda 1 | 15.322981 | 0 | 0.0466254751 | 0/51 |

The same-frame rows are exact arithmetic identities, not a complete causal control. The positive result establishes usable temporal headroom for this frozen anchor and proposal recipe on source FIT. Fixed temporal averaging is an established operation; these scores establish neither a novel learned module nor cross-domain/SOTA performance. Absolute source HVI quality remains low under this source setup; the color repair alone does not solve that mismatch. HVI's external pretraining is unchanged and is not relabeled SDSD-only training.

No excluded-17 tensor archive, target input or target score is accessed by this diagnostic. FIT was used for earlier method development and is not a blind generalization set. No model, parameter, lambda or checkpoint fitting occurs here.

## Verification and retained artifacts

- Eight real-Torch boundary tests, eight exporter/firewall tests and ten RGB-FIT tests passed with no skips. The initial wrong-interpreter launch and the original raw-audit network timeout are retained.
- Independent verification checks 2,170 scalar/provenance conditions; maximum scalar difference is `3.55e-15`. Coverage is exactly unchanged from the historical fixed-flow diagnostic. The two original FIT input tensor hashes equal the official RGB reader outputs.
- Root independently verified downloaded result/execution hashes and 334 aggregate/identity checks, with maximum scalar difference `3.55e-15`.
- Result SHA256: `96dee7426235d1378ffae91cac9e7bbe70dca05ff01b0c7097c3455a53dda9a9`. Execution manifest SHA256: `00f43663f686025a244ba5db6d3cb1ee4a3263f38d7242ba1fc190b6af49db61`.
- Full sufficient statistics are in `results/fit_rgb_v1/`; release/launch/test/failure/backup evidence is in `checks/`. Binary outputs stay outside Git. New exclusive SFS backups are `color_contract_rgb_hvi_source_v1_final_v1` and `color_contract_rgb_fit_v1_final_v1` under `/temp/zihaohe/seer_cvpr/method_dev_20261112/`, with read-back hash verification.

## Current method decision

The requested methodology contribution is not yet established. Joint matching did not pass its matched mechanism controls; HVI plus fixed temporal fusion is a useful benchmark; a frozen predictive critic has close prior work and still requires a distinct operator and utility evidence. The next new module must beat the corrected fixed-fusion benchmark and matched ordinary-confidence/descriptor/shuffled-prior controls under a frozen source protocol before it is promoted to a core contribution. The source color fix does not by itself establish a cross-domain claim.
