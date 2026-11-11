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

## Completed screens and current candidate

Pilot v1 local beat matched continuation by 0.1142 dB, with a scene interval
crossing zero. Pilot v2 froze the base exactly and trained only fusion for 600
steps: local improved by 0.00415 dB, prior by 0.00263 dB. Both PSNR intervals
cross zero. Full curves and raw rows are in `pilot_v1/` and `pilot_v2/`.
The extra feature path has not supplied a substantial restoration gain.

The next feasibility experiment uses the stronger BVI-only full restoration
checkpoint and real BVI motion. One restoration pass supplies all RGB frames;
frozen RAFT on restored predictions transports donors t−2 and t+2 to the target.
No GT is used for flow or candidate generation. On each 8×8 output patch,
correction directions d define observable Gram G_ij=mean(d_i*d_j).
Training GT supplies signed benefits a_j=mean((GT−base)*d_j). Constrained weights
minimize w^T G w−2 a^T w, w≥0, sum(w)≤1; zero correction is feasible.
This quadratic identity and minimizer are classical, not a novelty claim.

First cache one fixed crop/clip per TRAIN48 sequence and measure the GT oracle
within this candidate family. Only a sufficiently useful candidate family
warrants learning. The subsequent small selector pilot will split these 24
training scenes into 18 fit / 6 internal holdout scenes, fit input statistics on
fit scenes only, and compare signed-risk supervision with direct MSE gating at
equal input/capacity/update budget. These are internal selector-development
scenes: the historical base model may have seen all of them. Its checkpoint
does not contain a historical split hash. Neither this oracle nor that holdout
is a final learned benchmark result. GT-derived a/weights must never be inputs.

## Competitive evaluation

See `BASELINES.md` for primary-source methods, code and protocol differences.
The development SDSD39/Y/crop256 results are not a public leaderboard.
Establish separate comparable official-protocol evaluations, RGB PSNR/SSIM,
full resolution, temporal quality and compute, with STCD/AMNet-class video
models as references. The BVI 10/20 conditions denote illumination levels,
not frame rates. New model selection uses validation; benchmark comparisons
need a separately frozen final training/evaluation protocol.
