# Observation / trajectory v2 — completed experiments

All42 declared source-trained heads completed1000 updates. Both base arms
were evaluated on all16 BVI VAL sequences with all31 declared methods each.
Manuscript work remains paused; the objective is a competitive method with
positive evidence and a defensible contribution.

## Positive result to retain

Correct neighboring observation values improve over the matched same-frame
detail control. In the pretrained arm, raw_node gains PSNR-Y **+0.02276 dB**
(scene95%CI[+0.00333,+0.04429]), reduces LPIPS **0.01769**
(difference CI[-0.02577,-0.01068]) and reduces absolute tLP gap **0.00261**
(difference CI[-0.00465,-0.00090]). This directly tests explicit neighboring
values while keeping the base, capacity, source sampling and training budget
fixed. It does not remove neighbor information already present in the video base.

The none arm also improves LPIPS by0.01738 and gap by0.00445 against the
same-frame control, with scene intervals excluding zero. Its PSNR-Y gain
+0.02874 has CI[-0.00062,+0.06884], so that metric's improvement is uncertain.
Against mismatched donors, raw_node PSNR-Y gains are **+0.09370 / +0.24170 dB**
for pretrained/none; both intervals exclude zero, as do both LPIPS improvements.
This supports retaining an aligned observation branch for the next method.

## Quality and temporal reconstruction

Numbers below average the three final head seeds where applicable. Lower LPIPS
and gap are better; higher PSNR is better. Complete31-method, individual-seed,
sequence and scene values are in [analysis_v1.json](results/analysis_v1.json),
with all prespecified contrasts in [analysis_v1.md](results/analysis_v1.md).

| Base arm | Method | PSNR-Y | PSNR-RGB | LPIPS | Absolute tLP gap |
|---|---|---:|---:|---:|---:|
| pretrained | base | 19.28493 | 18.35735 | .24483 | .005237 |
| pretrained | raw_node | 19.36563 | 18.42266 | .21121 | .005427 |
| pretrained | trajectory | 19.35858 | 18.42261 | .22850 | .004028 |
| pretrained | trajectory_fixed_edge | 19.36420 | 18.42642 | .22276 | .003993 |
| pretrained | previous v1 raw/direct | 19.38255 | 18.43303 | .20256 | .006853 |
| pretrained | fixed raw equal thirds | 19.38691 | 18.42675 | .18784 | .013272 |
| none | base | 18.54648 | 17.36560 | .26488 | .005863 |
| none | raw_node | 18.76055 | 17.49872 | .21323 | .006662 |
| none | trajectory | 18.76725 | 17.51056 | .22805 | .004056 |
| none | trajectory_fixed_edge | 18.77400 | 17.51336 | .22299 | .004013 |
| none | previous v1 raw/direct | 18.79900 | 17.52006 | .20482 | .008565 |
| none | fixed raw equal thirds | 18.80986 | 17.50621 | .20436 | .016532 |

The trajectory model improves PSNR-Y over its own base by **+0.07365 /
+0.22078 dB**, with CIs[+0.03121,+0.13341] /[+0.09789,+0.36986]. LPIPS
differences are **-0.01633 /-0.03683**, with both intervals below zero.
Gap differences against base are-0.00121 /-0.00181; only the none arm's
interval excludes zero ([-0.00368,-0.00012]). Thus there is joint positive
cross-dataset evidence in the none arm; temporal improvement over the
pretrained base remains uncertain.

Compared with the preceding frozen v1 raw/direct reference, trajectory reduces
gap by **0.00282 /0.00451**, with both scene intervals below zero. This costs
PSNR-Y0.02397 /0.03175 dB and worsens LPIPS0.02595 /0.02323; the corresponding
intervals exclude zero. V1 has a different crop/frame training recipe, so this
is a frozen reference comparison rather than a single-mechanism isolation.

## What has not earned a method claim

The learned conditioned-edge model does not beat fixed edge precision0.5.
Fixed-edge has better mean values on all four metrics in both arms; its LPIPS
advantage is supported by both scene intervals. Removing other-donor features
does not hurt the conditioned model; scrambling them changes metrics only
slightly. The proposed cross-conditioning mechanism therefore has no convincing
positive support under this recipe and is not retained as a contribution.

The useful next starting point is aligned observation correction plus the
fixed temporal reconstruction control. Learned reliability needs a stronger,
source-validated mechanism that beats these controls. Graph regularization,
frequency separation and convex fusion themselves are established principles.
This completed round is method development, not a negative-results paper.

## Additional video reference

The author-labelled DID release of [TempRetinex](https://github.com/liyinibristol/TempRetinex/tree/03ca79f57f2ad7b4c6120364de3a8c59afa42eed)
scores PSNR-Y19.71782, PSNR-RGB18.89748, LPIPS.16328 and gap.015543 on the
common VAL16/center16/crop256 protocol. It has higher spatial quality and a
larger gap than the trajectory models. Its exact source training split/config
is undocumented, and it is not an SDSD-matched reference. No SOTA conclusion
follows in either direction.

The frozen adapter preserves forward recurrence, full RAFT20 and the official
four-view spatial ensemble. GT histogram matching and target optimization are
excluded. Protocol choices and differences from full-video author evaluation
are fixed in [REFERENCE_PROTOCOL.md](REFERENCE_PROTOCOL.md). Raw metrics,
strict44+179-key loading and unchanged-state evidence are in
[references/tempretinex/](references/tempretinex/).

## Reproducibility and scope

- [PLAN.md](PLAN.md) was frozen before source training and target scoring.
  Seven variants ×three head seeds ×two historical source-only base arms.
  Source is SDSD136/68 scenes; fit/internal split51/17 scenes. Target is BVI
  VAL16/eight scenes/two illumination conditions; TEST images remain unused.
- The two historical base arms each have one training seed0 and unmatched
  original pixel initializations. Head seeds do not replicate base training.
  BVI VAL informed earlier development, so no whole-history target-blind claim.
- Both source hosts passed133 CPU tests without skips. Final per-seed
  initialization and sampling plans match across variants and arms. Raw-family
  normalization is exactly equal to the preceding v1 records in every arm/seed.
- The [local training verification](checks/root_training_verification.json)
  checks all42 final file hashes and42000 consecutive finite curve entries.
  Full text records are in [artifacts/](artifacts/); weights stay outside Git.
  The [backup manifest](checks/checkpoint_backup_manifest.json) identifies
  verified shared copies of all42 heads, initial states and plans.
- All methods share base prediction and RAFT geometry per clip. Extra adjacent
  flow/CPU graph work is additional cost. Main scoring uses the same31051
  environment as v1. All base/RAFT/new-head/reference-head states remain fixed.
- Statistical analysis was frozen before new target scores. It averages head
  seed effects per sequence before bootstrapping eight whole scenes,10000
  draws/RNG0. Intervals are exploratory, unadjusted, conditional on fitted
  models, and do not include base-training or model-selection uncertainty.
- [Independent verification](checks/independent_statistics/verification_complete_v1.json)
  confirms372 raw means exactly, all408 interval endpoints within4.45e-16,
  and41544 scalar comparisons. GT temporal references match exactly between
  both arms and TempRetinex. Replay drift from v1 is recorded, not attributed
  entirely to head randomness.
- [Input content verification](checks/input_audit/) records512 original-file
  and actual crop/32-clip tensor hashes. Both capture endpoints have identical
  content and no changes. Capture overlapped early main setup and followed the
  TempRetinex run; it is not retroactive byte-level proof for earlier inference.
  The post check followed all inference. No TEST images or model scoring were
  used by this independent image/NumPy-only audit.
- All task computation processes are finished on31051,32193 and30794.
  Final process scans and the record of four verified temporary transport tar
  removals are in `checks/`. Permanent weights/source/raw/log/input-audit
  evidence remains under shared `method_dev_20261112/observation_v2_final_v1/`.

Recompute the complete analysis from [pretrained_raw.json](results/pretrained_raw.json)
and [none_raw.json](results/none_raw.json) using `analyze_observation_v2.py`.
Raw result SHA256 values are9a90aafea8abf40fc91e7c64e6546e6966705649e9c52a6c0936569af54f9cb8
and d357b6497fa4538de1ed3d4a7ab98601c99ea91773860ee38128f75af839a7e9.
