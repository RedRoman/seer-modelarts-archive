# HVI temporal FIT feasibility v1

Freeze this diagnostic before reading any HVI source quality score. This is a
separate source experiment; it does not change the no-GT HVI export or the
running joint-search screen. Root coordinates execution only after the final
HVI manifest exists. No real-data scoring is performed by the implementation
or its synthetic tests alone.

Use exactly the original136 SDSD TRAIN clips and the existing seed20261114
51/17 scene split. **Only the51 FIT scenes/102 clips may load GT, predictions,
or flow tensors for this diagnostic.** The remaining17 scenes/34 clips and all
target-domain frames are excluded. All136 file hashes and metadata may be
checked for lineage, without interpreting their pixels. Keep the original
16-frame/256-crop tensors and their source-file bindings. HVI float outputs
must come from the completed GT-free `hvi_source_anchor_v1/cache_v1` export,
with weight/state/release/input/output hashes and unchanged model state.

Declare exactly seven result rows, no other lambda, checkpoint, optimization,
calibration or oracle:

1. HVI spatial anchor.
2. Temporal mean mixed with anchor at fixed lambda0.25.
3. Temporal mean mixed with anchor at fixed lambda0.5.
4. Temporal mean mixed with anchor at fixed lambda1.0.
5–7. Same-frame duplicate arithmetic checks at the same three lambdas. They
copy the center HVI value and are **not warped**. They must reproduce the
anchor bit-for-bit. These are identity/parity checks, not strictly matched
causal controls for the temporal operation.

For each target frame, gather HVI donor RGB at saved Seer target-to-source
flow±2 using bilinear interpolation, `align_corners=False`, border sampling.
Intersect the original binary mask with actual image geometry and donor-time
availability. This diagnostic does not use HP9, new flows, new model inference,
an occlusion predictor, V-JEPA addresses or GT-dependent correspondence.
Average the valid donors equally per pixel: two valid donors each have weight
0.5; one valid donor has weight1; no valid donor leaves the anchor unchanged.
Then `prediction=anchor+lambda*(donor_mean-anchor)` on supported pixels.
No padded invalid value contributes to the average. Clamp the final float
prediction to[0,1]; retain ordinary bilinear float arithmetic.

CPU float32, four threads. Report all RGB pixels of all16/full256 for every
FIT clip. Compute squared residuals in float32 and reduce them in float64;
clip PSNR is `-10*log10(clip_RGB_MSE)`. Report the mean clip PSNR, mean RGBMSE,
and each of the51 scenes' two-clip means and differences versus HVI. Equal
scene weighting equals equal clip weighting because every scene has two clips.
Also report valid-neighbor coverage and exact identity checks. Save JSON
records and hashes only; no prediction RGB or GT arrays are written.

The question is whether these actual neighboring HVI values provide a
plausible fixed-fusion starting point on **FIT data**. All three lambda results
remain visible. A favorable FIT result does not select a deployable lambda,
establish generalization, validate a learned module, or prove SOTA. An
unfavorable fixed mean also cannot rule out all possible learned temporal
methods. Any later module design/evaluation needs its own frozen protocol.

CLI, after root's dispatch:
`python run_fit_feasibility.py --source-cache ORIGINAL --hvi-cache COMPLETE_HVI
--geometry-manifest FROZEN_GEOMETRY_MANIFEST --out NEW_DIRECTORY`.
Use `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`.
The original source/HVI cache and all previous evidence remain unchanged.
