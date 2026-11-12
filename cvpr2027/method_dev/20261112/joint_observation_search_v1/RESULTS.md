# Joint observation search v1 — complete, source gate not passed

All five final400 searches completed. The explicit joint model reproduces the
strong frozen anchor to about five decimal places in source RGB PSNR and fails
all four prespecified primary checks. Do not advance this version to target
inference or claim a validated joint-correspondence, calibration or prior module.
This closes this implementation screen; it does not establish that all search
methods fail. Manuscript work remains paused.

## Complete source results

Full16/full256 float RGB,34 clips/17 scene pairs, mean of clip PSNR with equal
scene weights; the historical base saw all source scenes. These numbers are not
BVI uint8 Y-channel scores or independent base generalization measurements.

| Method | Mean RGB PSNR | Mean RGB MSE |
|---|---:|---:|
| Original base | 28.56486846 | .00229092395232 |
| Frozen raw-observation anchor | 28.71543365 | .00225201872108 |
| Joint / prior / calibrated | 28.71542829 | .00225202098130 |
| Unary-only / prior / calibrated | 28.71542484 | .00225202531321 |
| Joint / prior / constant variance | 28.71543365 | .00225201872108 |
| Joint / pixel / calibrated | 28.71542684 | .00225202178564 |
| Joint / shuffled prior / calibrated | 28.71542791 | .00225202135893 |

Joint-prior minus anchor is **-.000005359 dB**, percentile95 scene-bootstrap
interval[-.000016622,+.000005397]. Its RGB-MSE difference is
+2.26022e-9, interval[-2.40709e-9,+7.35910e-9]. Joint minus unary-only is
+.000003442 dB, interval[-.000030352,+.000039805]; its MSE interval also crosses
zero. Neither the .05dB anchor requirement nor the .02dB unary-control requirement
is met. The calibrated-versus-constant and prior-versus-pixel interval checks
also fail. The tiny mean advantage over shuffled descriptors alone cannot
support a prior contribution. All modes, scenes and intervals are retained in
[the complete analysis](results/source_internal_analysis.json).

At the final checkpoint, joint-prior has mean absolute source offset
9.06e-5 pixels and amplitude .0168338; constant variance has amplitude0 and
exactly recovers the anchor. This records very little active correction in
this tested setting. It does not prove a particular cause or justify retuning
the frozen model after looking at these results.

## Implementation, execution and verification

[PLAN.md](PLAN.md), analyzer, cache builder, four-scalar model and trainer were
frozen in commit `b08c480bba7d060372d619ac806993b28a6fc31e` before training.
All five use the same feasible candidate pairs, initial tensors, source plan,
400-step Adam budget, observations and 6,626 frozen selector parameters.
The unary-only control removes only pair appearance compatibility. Calibration
uses exactly51 FIT scenes/102 clips/417,792 patch observations. There is no GT
input to the cost/prediction helpers, no free RGB decoder and no new encoder,
flow or NPU inference in this screen.

- 28 distinct real Torch CPU tests passed,0 skipped. The259-file runtime was
  hash verified; the formal31051 environment passed the same focused suite.
- Each mode's initial source-FIT prediction matches the anchor **exactly**.
- All 2,000 training updates have finite records; the five400-step curves and
  parameter projections are preserved. Selector state remained unchanged.
- A separate NumPy count-matrix bootstrap recomputed522 numeric values and all
  gate decisions; largest difference7.11e-15. Root repeated this on downloaded
  records. The unchanged base/anchor replay has136 scalar checks, max error0.
- Root checked all11 owner-frozen files, all10 downloaded initial/final checkpoint
  byte hashes, identical initial-state/plan identities and calibration split.

The first orchestration attempt successfully wrote exact FIT parity, then
encountered a receipt/payload filename collision. No training or held-out
score preceded that failure. [The recorded correction](orchestration_fix_v2.json)
creates separate checks/log directories and a distinct job label; original
release/failed records remain intact. The same tests and FIT parity passed
again, followed by the unchanged experiment. This was an execution-path fix,
not a model, budget, data, metric or gate change.

Actual per-mode optimization time was598–703 seconds, excluding initial input
hash verification; five CPU jobs used four threads each. These are observed
execution times, not a throughput benchmark. All five complete before the
single34-clip evaluation. No target frames were read by this module screen.
The independent external-baseline replay did use previously viewed BVI VAL;
no global target-blind claim is made.

## Preservation and next evidence

The exclusive SFS backup
`/temp/zihaohe/seer_cvpr/method_dev_20261112/joint_observation_search_v1_final_v1`
contains481 verified payloads/843,407,925 bytes, including compact observable
costs, all five heads and runtime sources. It contains no copied original RGB
cache. Backup-manifest SHA256:
`8cbf3743ad69185d1060a49b55a345fb25ffe74de3c9bbbf4d0d45ae8deccfbf`.
The6,287,360-byte download bundle hash and all345 downloaded payload hashes
were verified; only UTF-8 evidence enters Git. See
[local archive checks](checks/local_archive_verification.json) and
[root evidence review](checks/root_evidence_review.json).

The separately registered [HVI FIT diagnostic](../hvi_temporal_fit_feasibility_v1/RESULTS.md)
shows fixed temporal-fusion headroom on FIT data. HVI is a strong BVI spatial
reference but weak on these SDSD source inputs; it must not be called a
universally strong source anchor. Those FIT gains do not validate a module or
cross-domain improvement. A future method must beat the fixed fusion controls,
demonstrate a defensible operator difference and pass its own frozen protocol.
