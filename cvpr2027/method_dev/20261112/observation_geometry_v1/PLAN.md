# Observation geometry v1: source-only coordinate module screen

Frozen before training or new source-holdout scores, 2026-11-12 UTC.
No manuscript work, target frames, target metric computation, or target selection.

## Question and mechanism

Does a video-prior-conditioned residual coordinate predictor improve real
cross-frame measurements over the existing strong raw-observation selector,
and over an otherwise identical pixel-conditioned coordinate predictor?

The preceding soft-slot memory pilot failed its prespecified source gate.
Its posthoc frozen-head diagnostic found prediction changes of only 0.000073
RMS when removing all memory values; the output residual was predominantly
smooth (LP9-filtered residual energy / residual energy 0.966, not an orthogonal
frequency partition). This motivates removing the trainable RGB path. It
does not prove that phase averaging caused the failure.

This module predicts only bounded residual target-to-source flow coordinates.
Original offsets are [-2,+2] frames. A shared64-to16 descriptor projection,
then own/source/difference descriptors plus original flow and validity, feed
a shared direction CNN with hidden16. It predicts two per-axis increments,
bounded by 2*tanh at patch16 resolution and bilinearly expanded to full256.
The last convolution starts at zero. Original flow samples descriptor addresses
at patch centers; refined coordinates sample calibrated actual low-light RGB.
Candidate construction is the unchanged raw_detail recipe: warp calibrated RGB,
then high-pass the warped observation, with the original LP9 support rule.

The pretrained seed0 raw_node1000 RGB selector and its fit-only normalization
are frozen. Gradients traverse recomputed observable features and the frozen
selector to the coordinates. There is no learned RGB residual or decoder.
Output equals the frozen selector evaluated on refined observations; the
zero-offset implementation must reproduce the original strong anchor.
Masks intersect original validity with refined geometric bounds, then retain
the old LP9 support erosion. Original occlusion estimates are not recomputed;
there is no claim that geometric validity guarantees occlusion correctness.
Flow refinement and deformable sampling are established operations. This is
a testable module hypothesis, not a validated novelty or SOTA claim.

## Fixed source data and training

- Historical SDSD-only E17 small pretrained seed0 final20k base, unchanged.
- Reuse all136 deterministic source TRAIN clips, full16 frames/full256, the
  original RAFT flows/masks, and the already exported frozen E17 descriptors.
  No new encoder/RAFT execution or NPU allocation is needed.
- Bind original source RGB, memory-address, reference-prepared and checkpoint
  SHA256 hashes in a new metadata-only manifest. Preserve all prior files.
- Same51 fit /17 internal development scenes, split seed20261114. These17
  scenes were seen by the historical base; this is module development data.
- Three modes: `prior`, `pixel`, `shuffled_prior`; same parameter shapes,
  exact initial tensors, and exact source/crop/frame plan, seed0.
- Prior and pixel are cached64-dimensional full-clip addresses from the memory
  pilot. Shuffled prior uses a fixed private permutation (seed2718) over the
  full16x16x16 address positions. GT never enters module inputs.
- Exactly400 Adam updates, lr0.001/default betas/eps/no weight decay, CPU
  float32/four threads per mode. Same plan as memory pilot: one fit clip per
  update, eight distinct loss frames and one aligned128 crop. Crucially,
  compute the module on full256 context first and crop only predictions/GT
  for the loss. Normalize by the unchanged fit-only base RGB MSE.
- No early stopping, holdout checkpoint selection, LR search, extra modes,
  or post-score implementation/threshold edits within this version.

## Preflight and evaluation

Before training: meaningful module/gradient/mask tests, trainer coordinate
tests, exact initial-state/plan matching, and zero-offset parity on one FIT
clip against the original frozen raw_node path (prediction/features/candidates
maximum absolute error <=1e-6). Check frozen selector state after training.

After all three final400 checkpoints: one complete evaluation on34 cached
source clips /17 scenes, all16 frames/full256. Include base and unchanged
raw_node1000 anchor. The anchor had a different training budget, but each new
mode starts from exactly that same frozen anchor; only coordinate parameters
are optimized. Record RGB float32 squared residuals with float64 reduction,
PSNR from each clip MSE, then equal scene means. These are not target uint8
PSNR-Y or published benchmark numbers.

Use scene-paired percentile bootstrap, 10,000 draws, RNG0, 95% intervals;
two clips stay together. Intervals are unadjusted exploratory development
statistics, with no training-seed uncertainty.

Advance to source replication only if ALL checks pass:

1. Prior improves mean clip PSNR over the frozen anchor by at least0.05dB.
2. Prior improves mean clip PSNR over pixel geometry by at least0.02dB.
3. Prior-minus-anchor RGB MSE bootstrap upper95% bound is below zero.
4. Prior-minus-pixel RGB MSE bootstrap upper95% bound is below zero.
5. Prior mean RGB MSE is lower than shuffled-prior geometry.

Passing this pilot authorizes only source replication and stronger controls.
It does not authorize target inference or establish cross-domain/SOTA claims.
On failure, preserve all results and diagnose within source data; do not
promote a favorable metric or partial subset as a successful module.
