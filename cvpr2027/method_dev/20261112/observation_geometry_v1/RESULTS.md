# Observation geometry v1: complete source screen

The bounded coordinate module is implemented and the three final400 heads
have been evaluated. The source-development gate did not pass. Prior geometry
preserves the strong anchor's quality, but adds no demonstrated improvement
over it or shuffled addresses. No target-domain evaluation was run.

| Method | Mean clip RGB PSNR (dB) | Mean RGB MSE |
|---|---:|---:|
| Historical base | 28.564868 | 0.002290923952 |
| Frozen raw_node1000 anchor | 28.715434 | 0.002252018721 |
| Prior geometry | 28.714550 | 0.002252348058 |
| Pixel geometry | 28.691713 | 0.002257957094 |
| Shuffled-prior geometry | 28.715970 | 0.002251583609 |

These are full16-frame/full256 float RGB scores on34 cached source TRAIN
clips/17 scenes, already seen by the historical base. They are not target
PSNR-Y benchmark scores. Two illumination clips remain paired within scenes.

Prior-minus-anchor mean PSNR was **−0.000883dB**, scene-bootstrap95%CI
[−0.011393,+0.009716]; RGB MSE difference was +3.2934e−7 with CI crossing
zero. Prior-minus-pixel was **+0.022838dB** [0.008575,0.041324], with a
favorable MSE interval. However, prior-minus-shuffled was −0.001420dB
[−0.005981,+0.003174]. Thus the pixel contrast alone cannot establish a
useful video prior: the unchanged strong anchor and shuffled control already
match it. Only two of five prespecified checks passed. See the complete
[raw scores](results/source_internal_raw.json),
[analysis](results/source_internal_analysis.json), and frozen [PLAN.md](PLAN.md).
Intervals are unadjusted exploratory development statistics, conditional on
one module seed/one historical base, without training-seed uncertainty.

## What was implemented

The module has8418 trainable coordinate parameters and6626 frozen selector
parameters. Own/source/difference descriptors, original flow and validity
predict shared-direction residual coordinates, bounded by2 pixels per axis.
The refined locations sample actual calibrated low-light observations; the
existing RGB selector and its source-fit normalization remain frozen. There
is no trainable RGB decoder. Full256 context is retained; only the supervised
loss is cropped128 over eight sampled frames.

All three modes have exactly the same initial tensors and400-update
clip/crop/frame plan. Each starts at zero coordinate increment. On the
declared real FIT clip, candidate details,19 selector features and final
predictions were **bitwise identical** to the old anchor before training.
All18 actual CPU tests passed with no skips, including gradients through the
frozen selector and the full-context/loss-crop contract. A separate real-size
synthetic smoke measured approximately1.6–1.7 seconds/update; this is not an
end-to-end video benchmark.

The trained source-holdout mean absolute coordinate increments were0.1764,
0.2922 and0.0931 pixels for prior, pixel and shuffled prior respectively.
The networks therefore did move sampling locations. This does not prove that
their correspondences improved. Refinement also changes validity and the
selector's observable inputs; frozen parameters do not imply fixed mixture
weights. Registration accuracy would require additional mechanism controls.

## Independent verification and artifacts

All three heads completed400 consecutive finite updates, with selector state
unchanged; elapsed training was683.68/640.31/669.93 seconds. The complete
source screen ended at2026-11-12T04:11:51Z. The final process scan at04:15:02Z
found no remaining task processes. No new encoder, RAFT, NPU or target run was
needed.

[Separate statistics verification](checks/separate_verification.json) recomputed
340 scalar statistics with a multiplicity-matrix bootstrap and matched exactly.
[Reference replay verification](checks/reference_repeat_verification.json)
checked136 base/anchor scalars across all34 clips against the preceding memory
screen: maximum difference was exactly zero. This confirms the unchanged
controls reproduce the prior evaluation.

The frozen code/protocol commit is `48c03abc4d29a5e9a9a11ac4c3410c26faabfafc`.
Release package SHA256 is
`a7a381310e16e2571ce56d3dbda1a51da2e99e62184fac1cfda3ed9ef53246c7`;
all253 Python files matched the repository and remote release. Code and gate
were frozen before this screen; neither was edited after scoring.

Full weights and text are stored outside Git in
`seer_geometry_dev_20261112/artifacts/source_geometry_v1/` and the verified
shared backup
`/temp/zihaohe/seer_cvpr/method_dev_20261112/observation_geometry_v1_final_v1`.
All314 payload files/4902471 bytes were independently SHA-checked locally;
the archive also contains its own manifest. Backup manifest SHA256 is
`d602b3e1d06b93d822677107740f3e00edc2b1e341eea4a0b9b60a05a92df725`.
The backup includes the original43941-byte anchor, all three final/initial
states, all1200 curve rows/plans, frozen sources, metadata, tests and logs.
Original RGB/flow caches were not duplicated.

[The repository inventory](evidence/archived_files_sha256.json) identifies
copied text evidence; tensor files were excluded from Git. The prior memory
pilot and every earlier protocol/result remain untouched.

## Method decision

Do not expand this coordinate-regression configuration to target evaluation
or larger sweeps. The stronger next hypothesis is explicit competition among
actual observation correspondences, with source-calibrated effective
uncertainty and donor–donor compatibility. It must be tested against an
otherwise identical independent-correspondence control. This is a proposed
next module, not an implemented or validated contribution. No paper or SOTA
claim follows from the current screen.
