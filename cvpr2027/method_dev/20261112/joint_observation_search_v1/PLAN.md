# Joint observation search v1: source-only mechanism screen

Status at registration: implementation and synthetic tests only; no new held-out
source scores or target predictions have been computed. Previous memory and
learned geometry pilots failed their frozen source gates. This is a new module,
not a reinterpretation of those results. Manuscript work remains paused.

## Mechanism and limitations

Use two actual neighboring observations at offsets -2/+2. Around each original
flow location, search the fixed nine (dx,dy) residual offsets
(-2,-2),(0,-2),(2,-2),(-2,0),(0,0),(2,0),(-2,2),(0,2),(2,2).
The explicit 81-pair posterior combines target-to-donor observation costs,
donor-to-donor compatibility, frozen descriptor costs and displacement cost.
Use phase-preserving RGB HP9 residuals of the same calibrated real observations
as the previous anchor; warp calibrated RGB before computing high-pass values.
Only mutually supported pixels contribute. The cost builder receives no GT.

Calibrate a nonnegative two-coefficient effective error proxy from source FIT
labels only: v=a*g^2*LP9(low)+b*g^2, with variance floor 1e-8. Fit exactly 102
clips/51 scenes by two-variable nonnegative least squares against squared
HP9(q)-HP9(FIT GT) residuals. This is an empirical effective error scale, not a
camera noise model; no independence or physical uncertainty claim is made.
The constant-variance control uses the all-FIT mean squared residual instead.

Posterior marginals produce expected residual coordinates; multiply by
one-minus-normalized marginal entropy and a nonnegative tanh amplitude. There
are exactly four learned scalars: amplitude (initial 0), log descriptor weight
(0), log displacement weight (log(.05)), log temperature (0). After every Adam
step project respectively to [0,4], [-4,4], [-6,2], [-4,4]. Residual coordinates
are bounded by two pixels. Both donors but no valid pair means abstention;
one available donor uses the same single-donor fallback in every mode. An
unchanged 6,626-parameter RGB selector combines the resulting real-observation
corrections. There is no learned RGB decoder.

Frozen VJEPA/pixel descriptors and original flows/observations are reused from
the immutable source caches. The old trained base, reference head, selector
normalization, source split and source sampling plan are unchanged. Joint
matching, uncertainty weighting and frozen-prior video enhancement have prior
art; this screen cannot by itself establish algorithmic novelty or SOTA.

## Five matched modes

1. `joint_prior`: all terms, calibrated error scale, frozen VJEPA descriptors.
2. `independent_prior`: identical state, costs and joint feasible-pair mask,
   with only donor-to-donor appearance compatibility removed. It is a unary-only
   posterior on the same joint feasible set. Marginals factorize only when that
   feasible set is Cartesian; no broader independence claim is made.
3. `constant_prior`: joint search with constant all-FIT effective variance.
4. `joint_pixel`: joint search with fixed pixel descriptors.
5. `joint_shuffled`: joint search with clip-local shuffled VJEPA descriptors,
   using a private CPU generator seeded 2718, with no global RNG mutation.

All modes have identical initial tensors, four trainable scalars, loss plans,
support rules, fixed selector, full input context and compute budget. A zero
amplitude must exactly recover the frozen raw-observation anchor before any
training. Synthetic tests cover arbitrary valid-pair masks, pair-term isolation,
fallbacks, axis/batch conventions, finite gradients and frozen-state invariance.
Full256 synthetic speed measurements do not count as efficacy results.

## Fixed training and evaluation

One historical SDSD P base/seed0; source TRAIN136 from 68 scenes. The pre-existing
scene split has 51 FIT scenes/102 clips and 17 internal-development scenes/34
clips. The historical base saw all 68 scenes. This is module development, not an
unseen-scene base generalization test. FIT labels alone calibrate error scales
and normalization and train the four scalars. No internal-development label is
used in costs, training, normalization or checkpoint selection.

Train all five modes for exactly 400 updates, Adam lr=.01 with default other
settings and no weight decay. Use exactly the previous seed0 source-only clip,
128x128 crop and eight distinct loss-frame plan. Forward uses full16/full256
context; loss alone uses the fixed crop/frames. CPU float32, four threads/mode,
no scheduler, grid, early stop, validation selection or target-based tuning.
Only final400 checkpoints are eligible. All five must finish before one complete
34-clip evaluation. Preserve failures and training curves; do not substitute an
earlier or partial checkpoint. No new target frames enter this module screen.

Compute full16/full256 RGB squared residual in float32 and its mean in float64.
Clip PSNR = -10 log10(clip RGB MSE); average the two illumination conditions
within each scene, then equally weight all 17 scenes. These source RGB metrics
must not be compared to target uint8 Y-channel metrics. Report every mode and
both indoor/outdoor groups, including unfavorable results.

## Prespecified decision

Primary mechanism gate: `joint_prior` must satisfy **all four**:

- mean scene PSNR gain over `frozen_raw_node_1000` >= .05 dB;
- mean scene PSNR gain over `independent_prior` >= .02 dB;
- paired scene RGB-MSE difference versus anchor has percentile95 upper bound <0;
- paired scene RGB-MSE difference versus unary-only control has upper bound <0.

Use 10,000 paired whole-scene bootstrap draws with NumPy default_rng(0),
resampling both illumination clips together. These are unadjusted exploratory
intervals conditional on one base and one training seed, not confirmatory
training-uncertainty or multiple-comparison corrected evidence.

Secondary component evidence is reported separately and cannot rescue a failed
primary gate: calibrated versus constant RGB-MSE interval upper <0; prior versus
pixel RGB-MSE interval upper <0; prior versus shuffled mean RGB-MSE <0. Passing
the primary gate only prioritizes source replication. Component claims require
their own supporting controls. This screen never authorizes target inference,
cross-domain claims, novelty claims or SOTA claims by itself.

## Execution and provenance

Freeze this plan, analyzer, model, cache builder, trainer and their tests before
source training. Validate immutable input hashes, exactly 102 FIT calibration
clips, all five initialization hashes and zero-offset anchor parity on one FIT
clip before training. Cache preparation can use four spawn CPU workers, four
threads each, with fixed manifest ordering; no GPU/NPU inference is required.
Keep original frozen releases intact. All new outputs live in a fresh
`seer_joint_search_20261112` cache root. Record every code/config/checkpoint hash,
exact plans, all 400-step curves, complete raw scores, independent statistical
verification and unchanged-anchor replay. Archive text only in Git; verify
binary checkpoint/cost-cache backups separately on shared storage.

External HVI-CIDNet, DarkIR and same-SDSD Retinexformer replay is an independent,
previously fixed comparison-provenance task. Its target results do not select
this module or alter any source gate or hyperparameter.
