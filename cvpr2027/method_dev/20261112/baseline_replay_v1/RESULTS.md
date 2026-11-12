# Strong image-baseline replay v1 — complete

All three prespecified historical checkpoints completed the full BVI VAL16
replay with the current v2/STCD/TempRetinex scoring environment. The strong
image-reference scores reproduce. They must remain in the comparison set;
the existing Seer observation modules have not established SOTA spatial quality.

| Method / fixed weight | PSNR-Y | PSNR-RGB | LPIPS | Absolute tLP gap |
|---|---:|---:|---:|---:|
| HVI-CIDNet / generalization | **24.241232** | **23.383146** | .153150 | .034285 |
| DarkIR / DarkIR_384 | 22.581697 | 21.465763 | **.151016** | .017732 |
| Retinexformer / same-SDSD-split160k | 19.881678 | 19.103253 | .203176 | **.007376** |

Higher PSNR and lower LPIPS/gap are better under these diagnostics. The table
contains mean per-sequence values, not an official full-video leaderboard.
Image methods act independently on each frame. The temporal gap alone does
not establish reconstruction quality, and the three training sources/budgets
are not matched. No SOTA ranking follows from this reference replay.

Raw results: [HVI](results/hvi_cidnet_bvi_val16.json),
[DarkIR](results/darkir_bvi_val16.json),
[Retinexformer](results/retinexformer_bvi_val16.json).
All three still exceed the existing v2 pretrained observation variants in
mean target PSNR-Y; HVI/DarkIR also have lower LPIPS but larger temporal gaps.
The comparison and corresponding tradeoffs must be reported together.

## Protocol and provenance

[PLAN.md](PLAN.md) fixed the weight identities, architecture, all16 sequences,
centre16/crop256 selection, raw prediction scoring and compatibility decisions
before new target scores. There is no target training, GT brightness fitting,
CP/alpha, GT histogram matching, weight selection, ensemble or TEST-image access.
The adapter's forward function accepts only LQ observations. Predictions are
clamped and rounded to uint8; GT enters the existing post-prediction scorer.

The unmodified upstream network files and original weights were copied to
`/cache/zihaohe/seer_baseline_replay_20261112/release_v1`.
The frozen271-file manifest has SHA256
`f2294e9a7493f88cee3455b3432b7d52366f9afe43d3da5a3f3a652c0b24164d`;
all259 locally packaged payload hashes match it. All frozen files remained
unchanged after inference. [The origins](evidence/upstream_origins.json)
identify original paths and checkpoint hashes; the protocol/adapter release
was not changed after scores appeared.

- HVI-CIDNet:191 state keys,1,975,569 parameters, weight SHA256
  `6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27`.
  The [author README](https://github.com/Fediory/HVI-CIDNet/blob/master/Readme.md)
  describes `generalization.pth` as LOLv2-Syn training with random gamma.
  The original download receipt and current public-file byte identity were
  not independently recovered; the pre-existing archived file is fixed.
- DarkIR:343 state keys,3,321,638 parameters, weight SHA256
  `61eee7d5cfb593d408fba4c716874449b324ed1f49cbff97c2a25ce2fcbe4fde`.
  It is the historical public `DarkIR_384.pt` reference; its exact training
  split is not independently verified. Author documentation describes both
  LOLBlur-only and multitask releases, so no SDSD-matched label is assigned.
- Retinexformer:122 state keys,1,605,701 parameters, weight SHA256
  `928cfa4f130914aa49268807b1f751095c5d28875051d87e3922af00017e53b6`.
  This is the archived160k-step model retrained on Seer's SDSD split, with
  architecture and training schedule different from the Seer base.

All three passed strict-load/synthetic CPU shape and batch-invariance checks.
HVI and Retinexformer also passed actual synthetic NPU checks. DarkIR's native
NPU smoke failed at the complex FFT absolute operation with a documented
`DT_COMPLEX64` error. Its complete original network therefore ran on CPU,
with the same NPU scorer. No FFT surrogate or network rewrite was introduced.
This device decision was recorded before any new score; the failed smoke
log is retained. All three model states remained unchanged.

Physical NPU7 was checked free immediately before each dispatch; other users'
cards/processes were untouched. The runtime uses the same Python environment,
Torch2.7.1, torch-npu2.7.1.post2, pyiqa0.1.13 and scoring-library SHA
`a26ff0c81f0e6fa380ca26ac87dc2197c66248d3b6945e0da1c99e5b3d3f451d`
as the current v2 reference. Evaluation elapsed105.36/310.78/89.89 seconds
for HVI/DarkIR/Retinexformer; these include loading, decoding and scores and
are not model-throughput benchmarks.

## Independent verification and preservation

The [separate NumPy verifier](verify_replay.py) passed210 scalar/hash checks,
with exactly zero aggregate error, both remotely and on the downloaded local
records. All96 comparisons of actual LQ/GT tensors (32 unique input clips,
reused by three methods) match the preceding frozen input audit. All48 GT
tLP values are exactly identical to the current v2 base. Full ordered16/16
coverage, strict loading, finite means, absolute-gap arithmetic and unchanged
states are checked in [local verification](checks/local_separate_verification.json).

The new PSNR-Y values differ from their historical values by only
+.005841/+.001713/+.002508dB for HVI/DarkIR/Retinexformer. Historical scores
were not deleted. In particular, the large advantage of the two image
references is reproduced, not explained away by the old metric environment.

The final scan found no remaining task inference process; all271 frozen
release files were unchanged. The [exclusive verified SFS backup](checks/backup_receipt.json)
contains304 files/34,832,415 bytes, including all three original fixed weights,
sources, five passed smoke reports, the failed DarkIR NPU smoke, complete
results and logs. Its manifest SHA256 is
`d5328850c490f5f6177b9a155daaaffb586d1d86c564abc19f4455739d9d03b7`.
Backup root:
`/temp/zihaohe/seer_cvpr/method_dev_20261112/baseline_replay_v1_final_v1`.

The977,871-byte text/source archive was downloaded and its SHA256
`3159387ef60dac7a230a093d20868e7855e150d307e6eb19d4396a99d5a242b1`
verified. All301 downloaded payload-file hashes were independently checked;
the three tensor checkpoints stay in the verified shared backup and are
excluded from Git. No original images or predictions were archived.

## Remaining modern video reference

[AMNET_READINESS.md](AMNET_READINESS.md) records the official RGB and NPU
source commits, inference interface and dataset-finetuned weight links.
Code/interface availability is confirmed, but the specified finetuned weight
has not been obtained: one anonymous Baidu request returned the extraction
initialization page. There is no AMNet inference or score. Keep it as a pending
modern video comparison, rather than substituting the general pretraining
checkpoint or a paper-table number. The frozen comparison registry is untouched.
