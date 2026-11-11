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

## Next method hypothesis

If the fusion path can improve recovery, supervise evidence selection using
counterfactual restoration utility: compare local GT error when neighbouring
evidence is admitted versus withheld, and train a selector to reject harmful
evidence. Motion/feature confidence alone is not the intended contribution.
Teacher information is confined to training supervision; inference uses only
the observed low-light video. This requires controls against ordinary local
attention, semantic guidance alone and conventional confidence gates.

## Competitive evaluation

See `BASELINES.md` for primary-source methods, code and protocol differences.
The development SDSD39/Y/crop256 results are not a public leaderboard.
Establish separate comparable official-protocol evaluations, RGB PSNR/SSIM,
full resolution, temporal quality and compute, with STCD/AMNet-class video
models as references. The BVI 10/20 conditions denote illumination levels,
not frame rates. New model selection uses validation; benchmark comparisons
need a separately frozen final training/evaluation protocol.
