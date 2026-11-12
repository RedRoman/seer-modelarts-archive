# Source-only observation-memory module screen

The user requested a concrete module contribution under time pressure. This
protocol is specified before new module training or source holdout scores.
Manuscript work remains paused. The module is a hypothesis, not established
novelty; the attention/memory primitives have close prior art in PRIOR_ART.md.

## Mechanism being tested

Frozen V-JEPA block17 builds cross-frame content addresses. Calibrated raw
low-light detail supplies actual RGB patch values. A shared32-slot memory
transports these values, preserving all16x16 within-patch RGB positions. For
each target, cross-frame mode removes that target frame's entire observation
write and recomputes normalization. This excludes target observation values,
not target information from the all-clip V-JEPA addresses. A common small
zero-output-initialized residual decoder restores pixels; the frozen base is
initially reproduced exactly. No flow/graph or GT is used in new module input.

This changes the information path tested by previous local-guidance and
candidate-weighting pilots. It does not make two-stage attention or memory a
new algorithm. The specific question is whether video-pretrained addresses
retrieve more useful actual observations than matched low-level addresses.

## Fixed source scope and screening budget

- Historical pretrained E17 seed0 SDSD-only final20k base, unchanged. One base
  seed; no causal base-training replication claim.
- Reuse all136 existing source TRAIN cache clips,68 scenes,16 frames/crop256,
  data seed20261113. No new images are read to produce the feature export.
- Same source scene split seed20261114:51fit/17internal scenes, two clips each.
  The base saw these scenes; this is module development, not final validation.
- Export only block17 projected addresses from cached low input, never GT.
  Fixed CPU projection seed1729, descriptor64, token_grid8x16x16. Encoder in
  eval/no-grad; fixed official weight SHA and frozen parameter state recorded.
- All five modes train seed0, exactly400 Adam updates/lr0.001/default betas,
  no weight decay, CPUfloat32/four threads, one fit clip/update, crop128,
  top/left multiples16, all16frames as context and eight unique supervised
  frames/update. MSE is divided by mean base MSE from fit scenes only.
- Same exact initial parameter tensors and same per-update clip/crop/frame
  plan across modes. Address descriptors are precomputed from full256 context;
 128 crops slice matching patches. No claim that the encoder itself sees128.
- Five declared modes: prior_cross, pixel_cross, shuffled_cross, prior_same,
  pixel_local. The latter is an ordinary same-coordinate adjacent-frame
  retrieval control; it is not a complete nonlocal-attention benchmark.
- A final checkpoint at400 is the only new trained checkpoint scored. No
  intermediate holdout evaluation, target scoring, adaptive training budget,
  early stopping or best-seed selection.

## One final source gate

After all five modes complete, score all34 internal clips/17 scenes, all16
frames, full cached256 RGB. Primary quantity is per-clip RGB float MSE over
every pixel, then a mean of the two conditions per scene. Also show PSNR from
each clip's MSE, equal-scene means and indoor/outdoor breakdowns. This is not
the uint8 common target benchmark or a public leaderboard.

Include base and the frozen preceding observation-v2 raw_node seed0 head as
strong references. The latter trained1000 steps and has a different module,
capacity and budget; it is a practical reference, not an isolated ablation.

Use paired17-scene bootstrap,10000 draws/RNG0, exploratory unadjusted95%
percentile intervals. Keep all raw sequences, scenes, modes and contrasts.
The prior_cross candidate advances only if it meets all of the following:

1. At least+0.05dB mean clip-PSNR gain over both pixel_cross and base.
2. Its RGB MSE contrast against pixel_cross has95% upper endpoint below zero.
3. It has lower mean scene MSE than the frozen raw_node reference.
4. It beats prior_same and shuffled_cross on mean scene MSE, supporting use of
   correctly associated neighboring observations rather than decoder capacity.

Failing this gate rejects this recipe as the next target experiment; it does
not establish impossibility or that more training cannot help. Passing is only
permission to invest compute: require additional seeds, ordinary nonlocal and
prior-local matched controls, and temporal/perceptual validation before a new
BVI comparison. No BVI/TEST inference is authorized by this source screen.
Earlier BVI development remains acknowledged; no whole-history target-blind
claim. Cross-domain and SOTA are not established by this module screen.

## Reproducibility and runtime

Meaningful tests cover packing/cropping, target-value exclusion, address/value
interventions, exact zero-init identity, finite gradients and mode state
matching. Record code/source/input/encoder/init/plan/final SHA, all400 curve
rows, source scores and unchanged evaluation states. Save tensors on local
/cache; backup verified final small weights and text to shared storage.
Use only fresh process-free NPU for the frozen feature export. No other user's
jobs, shared environments, frozen earlier sources or results are changed.
