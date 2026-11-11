# Method development — 2026-11-11

User objective: develop an innovative, competitive low-light video enhancement
method with positive experimental evidence. Manuscript work is paused.
The previous protocol audit supplies reliable infrastructure; its analysis-paper
framing is not the research objective. Work branch: `codex/seer-method-dev-20261111`.

## First executable hypothesis

The current pixel branch exchanges frames only at its coarse bottleneck.
Test whether transporting actual neighbouring restoration features at 1/8
resolution recovers detail that global token conditioning does not supply.
The prototype searches 5×5 positions in frames t−2 and t+2. A clip-centered
block-17 video-feature similarity may guide the correspondence. Values always
come from restoration features; the output projection starts at zero.
This is a performance screen, not a validated novelty or SOTA claim.

Matched arms: baseline continuation, local fusion, prior-guided local fusion.
All start from E17 pretrained seed 0, identical base weights, a fixed explicit
sample/augmentation plan, 400 optimizer steps, accumulation 1, base LR 2e-5,
fusion LR 2e-4, crop256/clip16. The loss matches the existing pixel/Retinex
recipe. VAL16 is evaluated at steps 0/200/400 with centre16 frames and crop256,
rounded uint8 RGB/Y PSNR, LPIPS and absolute tLP-to-GT gap. No test frames are
opened. Results are exploratory validation evidence used to select follow-ups.
Continuation of the reference receives the same data and update budget.

Deployment requires CPU tests and a real NPU forward/backward smoke, including
zero-initialization identity, equal local/prior parameter initialization and
finite nonzero fusion gradients. Output checkpoints include optimizer state,
model configuration, inputs/source hashes and data-plan identity. Runs use
fresh local `/cache/zihaohe/seer_method_dev_20261111/` directories.

## Positive learned result and fixed strong control

The frozen E12 BVI restoration model plus restored-RGB RAFT t−2/t+2 donors and a
6,626-parameter selector has now been trained and independently validated.
The selector uses 19 observable patch features; GT enters only training losses
and final scoring. Base and selector state digests are checked unchanged during validation; RAFT uses the same strictly loaded weight file in every run.

Three selector initialization/sampling seeds, each fixed at 1,000 optimizer
steps, give direct-MSE PSNR-Y gains of +0.4503 / +0.4661 / +0.4601 dB on
BVI VAL16 (eight unseen development scenes, both illumination conditions,
center16/crop256). Their mean is +0.4588 dB, with a whole-scene 95% interval
[+0.1364,+0.9482] after averaging each sequence's effect over selectors.
PSNR-RGB improves by +0.4497 dB. LPIPS falls from .15323 to .14494 on average,
but its scene interval crosses zero, as does that of the temporal gap.
These intervals are conditional on the three fitted selectors and one E12
base seed0. They do not include base-training or model-selection uncertainty.

The same-capacity direct-MSE control with a fit-only scalar loss normalization
gives +0.4581 dB. Risk supervision gives +0.4032 dB and loses to direct MSE.
Its simplex objective is algebraically equivalent to CsNet's classical MSE
fusion objective (see BASELINES); it is a control, not a new principle.
All three final models from every seed were evaluated; no best checkpoint or
best seed was selected. Fixed donor controls have tiny repeated-NPU numerical
drift (maximum per-sequence PSNR-Y .0003214 dB); the base and GT metrics match
exactly. Do not attribute all cross-run changes to selector randomness.

Raw validation rows, scene analyses, launch/source/checkpoint hashes and
backup manifests are in `selector_val_v1/`. Training plans, full curves,
fit-only normalization and internal holdout results are in `selector_v1/`.
The nine small selector weights are retained outside Git and on shared SFS;
the full E12 base is also backed up with verified SHA. No TEST images were read.

## Next method experiment

Use the positive direct-MSE selector as the minimum performance control.
The next question is whether truly different source observations can correct
errors shared by all-clip restored donors. A candidate constructed only by
warping another output of the same all-clip model does not isolate that
frame's information. Test proposals that use source low-light observations as
values, with prediction-only alignment and photometric guidance. A cheap
sampling-residual/backprojection proposal is a separate candidate control.
Tone/detail transfer, backprojection, noise weighting and negative kernels
already have strong prior art; combining them does not establish novelty.

Freeze a concrete candidate mechanism before further validation. First audit
its dataflow and measure training-only conditional oracle headroom, donor error
correlation and recovered learned gain. Then compare the same candidate family
under matched direct-MSE, full-Gram risk, and any proposed context-dependent
nonlinear fusion training. A contribution must beat the direct learned outcome
at matched capacity, source information and compute; a larger GT oracle or a
renamed GT confidence target is insufficient. Any context-dependent utility
experiment must intervene on added messages while retaining identical
background messages. The base already sees the whole clip, so this is not
removal of all information from a source frame.

These follow-up mechanisms have not yet been implemented or evaluated.
Manuscript work remains paused while method performance and novelty develop.

## Completed feasibility screens

Pilot v1 local beat matched continuation by 0.1142 dB, with a scene interval
crossing zero. Pilot v2 froze the base exactly and trained only fusion for 600
steps: local improved by 0.00415 dB, prior by 0.00263 dB. Both PSNR intervals
cross zero. Full curves and raw rows remain in `pilot_v1/` and `pilot_v2/`.

The BVI TRAIN48 cache then established a useful RGB correction family.
One restoration pass supplies all RGB frames; frozen RAFT on restored
predictions transports donors t−2 and t+2. No GT enters flow or candidates.
Per 8×8 patch, observable Gram G and training-only signed benefits a define the
classical bounded quadratic risk. Fixed averaging reduces TRAIN MSE by 3.15%;
its GT-assisted oracle reduces it by 21.16%. The latter is a feasibility bound,
not learned performance. These facts motivated the selector experiment above.

The selector's 24 cached training scenes use a fixed 18 fit / 6 internal holdout
split, fit-only normalization, and one final holdout pass after all models finish.
Those six scenes may have been seen by the historical base; they are not final
validation. The E12 checkpoint lacks a historical split hash, so retained
logs/counts are not claimed as hash-level historical split proof. The separate
BVI VAL scenes do not overlap the current frozen training lists.

## Competitive evaluation

See `BASELINES.md` for primary-source methods, code and protocol differences.
The development SDSD39/Y/crop256 results are not a public leaderboard.
Establish separate comparable official-protocol evaluations, RGB PSNR/SSIM,
full resolution, temporal quality and compute, with STCD/AMNet-class video
models as references. The BVI 10/20 conditions denote illumination levels,
not frame rates. New model selection uses validation; benchmark comparisons
need a separately frozen final training/evaluation protocol.
