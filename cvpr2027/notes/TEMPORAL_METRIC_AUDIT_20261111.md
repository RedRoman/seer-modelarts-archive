# H17 temporal metric correction (2026-11-11)

The archived evaluation JSONs are unchanged. This correction changes the
interpretation and aggregation of their existing tLP and GT tLP fields.
It does not rerun a model or change the underlying predictions.

## Error and definition

Both copies of tools/make_macros_cvpr.py previously reported the change in
the **signed** gap tLP(pred) − tLP(GT). The released
code_full/tools/aggregate.py --temporal instead measures an **absolute**
distance to the target's temporal activity. A signed gap can decrease when an
output moves farther below the target, so it is not the same score.

The corrected macro calculation follows aggregate.py exactly:

1. For each sequence, average output tLP and GT tLP over the seeds in each arm.
2. Compute abs(mean_seed(tLP) − mean_seed(GT_tLP)) for each arm.
3. Subtract the reference arm's absolute gap from the intervention arm's.
4. Average these paired sequence differences and bootstrap sequences.

The bootstrap is NumPy default_rng(0), 10,000 resamples of the 39 sequences,
with the 2.5th and 97.5th percentiles. Both arms have seeds 0 and 1 here.
The reference files are indomain/e17_small_pretrained_s{0,1}.json; the
interventions are h17/e22_tdelta_s{0,1}.json and
h17/e24_tdcos_s{0,1}.json.

Taking an absolute value after seed averaging is not generally identical to
averaging per-seed absolute values. One E24 sequence changes signed-gap sign
between seeds, making the pooled E24 effect slightly different from the mean
of its two per-seed absolute-gap effects. This order is explicit in the paper
and retained to match the existing aggregation protocol.

## Corrected results

Positive differences mean farther from the target temporal activity.

| Intervention | Old signed difference [95% CI] | Corrected absolute-gap difference [95% CI] |
|---|---|---|
| E22 relative loss | +0.3587 [+0.3425, +0.3757] | **+0.3555 [+0.3391, +0.3724]** |
| E24 cosine loss | +0.0085 [+0.0066, +0.0103] | **+0.0066 [+0.0044, +0.0088]** |

Full-precision corrected differences:

- E22: 0.35547707854993005,
  CI [0.3390883968668417, 0.37240168775599963].
- E24: 0.006621651291353708,
  CI [0.004377631380304328, 0.008786382734483011].

Per-seed absolute-gap effects are:

- E22: +0.3687095636657129, +0.3422445934341472.
- E24: +0.008141651149027242, +0.005206214308213345.

Both seed signs remain positive and both corrected intervals exclude zero.
The registered H17 temporal-improvement criterion still fails.
The LPIPS, PSNR, latent cosine and latent magnitude macros are unchanged.
The tLP result concerns an unwarped temporal-activity distance and does not
establish a change in motion-compensated reconstruction error.

## Verification and changed files

Run from the repository root:

    python3 cvpr2027/paper/tools/make_macros_cvpr.py
    python3 cvpr2027/paper/tools/check_temporal_metric.py

The independent check uses Python statistics.mean for the arm-level
calculation, NumPy only for the bootstrap, verifies all individual seed signs,
compares the raw JSON archive with the paper's mirrored JSONs, and asserts the
formatted macro values and intervals. It performs no inference and writes
nothing.

Changed:

- paper/tools/make_macros_cvpr.py and its identical copy in code_full/tools/.
- Four h17@{tdrel,tdcos}@tlp{,@ci} entries in the generated CVPR macro file.
- Temporal-gap definitions in paper/sections/experiments.tex and appendix.tex.
- Added paper/tools/check_temporal_metric.py.

Original per-sequence files and legacy ICLR macros were preserved.
