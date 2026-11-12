# Source FIT predictive-credit mechanism probe v1

This is an experiment protocol, not a paper draft. The question is whether actual masked V-JEPA2 prediction changes identify **incremental usefulness of observed neighbouring frames**, beyond five fixed controls. No learned restoration module, parameter search, internal evaluation, or target evaluation is authorized by this probe release.

## Frozen inputs and actions

- `subsets.json` SHA256 `35292bbec0bc056481129857436950644448ca669d2261b72de350f841a2a92e` fixes 12 source FIT scenes: lexicographic first six indoor and first six outdoor scenes, first pass only. Selection did not use scores.
- The central target is tubelet 4 (frames 8,9). Donors are tubelets 3,5 (frames 6,7 and 10,11). A patch is one spatial 16x16 cell across the two target frames; each scene contributes 256 cells.
- Source cache LQ is native BGR. Use the pinned selective LQ reader and verified BGR-to-RGB helper. RGB HVI values come from the already frozen, recomputed canonical RGB cache. Old flow/validity are retained exactly; they are not RGB HVI recomputed flow.
- Critic inputs are canonical RGB LQ only. Four actual encoder masks retain C, C+donor1, C+donor2, C+both, with original token positions and masks before attention. C excludes target and both donors. All four losses use the same target-local EMA reference. Each loss is squared distance between unit-normalized 1024-channel vectors.
- Retain the existing spatial 256 crop, without the hub's resize292/center256 operation, to preserve the transport grid. Use ImageNet RGB mean/std. This is a documented spatial adaptation. Target-local reference also differs from the official full-context teacher distribution; report its discrepancy, without claiming calibrated restoration risk.
- Fixed RGB actions use gamma .5, valid-support donor residuals with per-patch/frame/channel DC removed, pair valid-count normalization, and box/fixed-sum Euclidean projection. Preserve the anchor exactly outside valid support. The utility labels describe this exact action recipe, not every possible fusion action.

## Separation of observables and labels

The value producer first exports a GT-free sidecar: mean valid flow, valid fraction, negative warped LQ photometric MSE and negative warped HVI consistency MSE. The critic release binds its hash before source forward. Flow aligns the full-context static EMA cosine control. The sidecar does not contain GT or utility fields.

Only the separate source FIT value-scoring stage opens GT to produce four scalar patch MSEs. The prediction helper has no GT argument. Neither the critic nor the statistical analysis opens GT tensor entries. Whole-file hashes may read opaque source archive bytes for provenance. No raw images or GT arrays are written to Git or critic results.

Freeze code, protocol, subset, dependencies and unit-test evidence before the corresponding real-data stage. Record release timestamps and hashes. Synthetic CPU/NPU contract tests may precede the final sidecar-bound data release; they use no dataset clips or GT.

## Primary, predeclared decision

For coalition losses L0,L1,L2,L12, candidate scores for donors 1,2 are respectively L2-L12 and L1-L12. Labels are the signs of MSE2-MSE12 and MSE1-MSE12 under the fixed RGB actions. Require both donor valid fractions >=.5. Exclude utility magnitudes <=1e-10 and report excluded counts.

Combine both donor observations within each scene. Compute AUROC with midranks; single-class scenes are undefined. Use scene means, never pool patches across scenes. Compare against all five controls, with higher scores always meaning more useful:

1. The same conditional losses from an identically constructed random predictor (seed 20261122), without trained predictor weights.
2. The corresponding donor's flow-aligned full-context EMA cosine.
3. Negative warped LQ photometric MSE.
4. Negative warped HVI consistency MSE.
5. Conditional losses from the trained predictor after a fixed wrong-time content permutation. Non-target tubelets [0,1,2,3,5,6,7] rotate by three positions; target4 and token coordinates stay fixed. This uses new masked encoder calls and the real fixed target-local reference.

The **common scene intersection across candidate and all five controls** must contain >=8 scenes. On that common set candidate macro AUROC must be >=.55, and each of five candidate-minus-control mean differences must have lower percentile 95% CI >0 under 20,000 paired complete-scene bootstrap samples, seed 20261123. All comparisons must pass. No best-baseline selection, sign flipping, threshold tuning, or replacement of undefined scenes is allowed. Candidate macro across all its defined scenes is descriptive only when the common set differs.

## Secondary diagnostics and stopping

Report per-scene rank correlations, label balance, normalized loss ranges, local/full reference discrepancy and cost. Predictive interaction versus RGB action-recipe interaction is descriptive, not physical interaction ground truth.

At tau=1, phi1=(v1+v12-v2)/2, phi2=(v2+v12-v1)/2; B=clip(v12,0,1), c=clip(phi,0,1). Report the actual two-variable capped allocation with total B, zero budget, total/individual clipping, active caps, and the algebraically unique small-credit region. Positive phi sum is **not** the budget. If both phi are nonnegative and v12<=1, weights equal phi and the allocation has no extra freedom. This diagnostic is not a learned module or tau selection.

Passing this small source FIT screen only permits proposing a separately frozen train/control experiment; it does not establish novelty, SOTA, or cross-domain performance. Failure stops this particular predictive-credit module before expensive training. Retain existing fixed temporal fusion gains as separate facts; do not recast them as new module evidence.
