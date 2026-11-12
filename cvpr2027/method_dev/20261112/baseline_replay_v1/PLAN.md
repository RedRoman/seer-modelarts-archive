# Frozen strong-baseline replay v1

Declared before any new replay target scores. Replay exactly the historical
HVI-CIDNet `generalization.pth`, DarkIR `DarkIR_384.pt`, and same-SDSD-split
Retinexformer `net_g_160000.pth`. Do not choose another weight from target
scores. HVI/DarkIR are public-weight references with unmatched training sources;
Retinexformer shares the SDSD split but has its own 160k-step recipe.

Use all frozen BVI VAL16 sequences, eight whole scenes/two conditions, centre16
frames and centre256 RGB crops. Reuse the exact v2/STCD iterator and uint8
scorer: PSNR-RGB, PSNR-Y, LPIPS, consecutive-frame tLP, GT tLP, and the
per-sequence absolute tLP gap. No GT brightness rescaling, CP/alpha, adaptation,
optimizer, target checkpoint selection, or TEST-image reads. Source/weight
hashes, strict loading, unchanged inference state, actual input tensor hashes,
complete ordered coverage and exact GT-temporal replay are required.

The upstream network files are copied unchanged to a new exclusive runtime.
HVI uses CIDNet defaults; DarkIR uses its historical width32/2+2 middle blocks,
encoder [1,2,3], decoder [3,1,1], dilations [1,4,9], extra-depthwise configuration;
Retinexformer uses n_feat40/stage1/blocks[1,2,2]. Image models process independent
frames with batch1 and ordinary required-multiple reflect padding (none at256).
Clamp outputs to [0,1], round to uint8, and score the uncalibrated prediction.
No spatial augmentation or ensemble is added. No upstream file is modified.

Require synthetic CPU strict-load and finite-shape/batch-invariance smoke first.
Check NPU availability immediately before use; never occupy a card with another
user's process. Model NPU compatibility is established on synthetic inputs
before target evaluation. A model that cannot execute supported operations on
NPU may use CPU inference with the same frozen NPU scorer; record this decision
before its first target prediction. Preserve all failed smoke evidence.

Freeze the adapter, upstream sources, checkpoint and evaluator hashes in an
exclusive release before scoring. Runtime code and protocol are not changed
after any score. Freeze fixes in a new release if compatibility requires them,
before the affected method's target evaluation. There is one final complete
evaluation per specified method, and all scores are reported.

These common-crop development results are not official full-video leaderboard
scores or proof of SOTA. Old results are retained and compared transparently.
AMNet code/weight access is audited separately without substitutes or guessed
weights; unavailable access is recorded rather than treated as a model result.
