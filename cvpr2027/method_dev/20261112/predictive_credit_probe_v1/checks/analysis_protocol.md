# Frozen scalar analysis protocol

This document supplements the phase `PLAN.md`. It is finalized before reading any actual critic losses or RGB utility scalars. This is a source FIT mechanism screen, with no training, tau/lambda selection, internal-source scoring, target scoring, performance claim, or novelty claim.

## Input boundary

Only the twelve exact `scene`/`name` entries in `subsets.json` are allowed, with one clip per scene. Every entry must belong to the frozen source FIT split and not the internal split. Every clip must supply all 256 row-major 16 by 16 patches. The analyzer consumes scalar JSON and scalar NPZ (`allow_pickle=False`), never Torch archives or GT image/tensor pixels. Duplicate, missing, unknown-scene, wrong-role, or nonfinite records fail closed.

The critic manifest hashes every per-clip JSON and binds its RGB LQ tensor digest to the frozen subset entry. Coalition order is exactly empty, donor1, donor2, both. All three critic arms use a fixed target-local EMA reference. The trained, random-predictor, and wrong-time arms differ as prescribed in `PLAN.md`; the analyzer never changes those interventions. Static flow-aligned full-EMA cosine is a separate frozen reference score for each donor. Value-side features are already negative valid-support LQ photometric MSE and negative valid-support HVI consistency MSE, so the analyzer does not negate them again. Larger values mean more useful donor evidence in every arm.

The utility exporter independently supplies the base, donor1-only, donor2-only, and averaged-pair action MSEs. The analyzer recomputes all five saved utilities by scalar subtraction and requires agreement within `1e-12`; the sameframe zero-action MSE must exactly equal base MSE. These checks use no image pixels. The joint action includes the fixed support demeaning, gamma, valid-donor normalization, and box projection declared by the value producer. Its RGB interaction contrast is therefore an **action-recipe contrast**, not physical interaction ground truth.

## Primary statistic and gate

Only patches whose two valid-support fractions are each at least 0.5 are eligible. For each eligible patch, the two donors contribute separate labels and scores to their scene. The utility of donor1 given donor2 is `MSE(donor2) - MSE(pair)`; donor2 given donor1 is `MSE(donor1) - MSE(pair)`. Donors with absolute utility at most `1e-10` are counted and excluded from binary labels. Positive utility is the positive class.

Candidate conditional scores are `L(donor2)-L(both)` and `L(donor1)-L(both)`. The five fixed comparators are random-predictor conditional scores, wrong-time trained-predictor conditional scores, static full-EMA flow-aligned cosine, negative LQ photometric MSE, and negative HVI consistency MSE. No coalition-level score is duplicated across donors.

AUROC uses one-based midranks and gives ties half credit. Scenes containing one class or no eligible nonzero labels have missing AUROC, not 0.5. A complete-scene intersection is formed where the candidate **and all five comparators** are defined; all comparisons use this same intersection. At least eight of the twelve scenes must remain. Candidate macro AUROC is the unweighted mean on this intersection and must be at least 0.55.

On that same scene set, form each candidate-minus-comparator scene AUROC difference. Resample complete scenes with replacement 20,000 times using Python `random.Random(20261123)`; each resample contains the same number of scenes as the intersection. Each comparator restarts the same seed, producing the same resampled scene-index sequence. The interval is the 2.5th and 97.5th percentiles of mean scene differences, with linear interpolation at `(n-1)*q`. The lower bound must exceed zero against **every** comparator. There is no patch bootstrap, patch pooling, best-baseline selection, or replacement of missing scenes on a comparator-specific basis.

If fewer than eight common scenes remain, the gate is **not evaluable**. If evaluable but any threshold fails, it fails. Passing is only eligibility for a separately frozen small source module pilot. This FIT screen is not publication evidence or a SOTA/generalization result.

## Secondary diagnostics

For each scene, report Spearman correlation of scores with utility standardized within that scene, using midranks for both variables; constant variables have missing correlation. Do not pool scores or utilities across scenes. Report trained-predictor interaction `v12-v1-v2` against RGB action-recipe interaction `u12-u1-u2` using eligible patches and per-scene Spearman only.

For tau fixed to one, compute `phi1=((L0-L1)+(L2-L12))/2`, `phi2=((L0-L2)+(L1-L12))/2`, `B=min(1,max(v12,0))`, caps `ci=min(1,max(phii,0))`, and `r=softmax(phi)`. Exact two-variable water filling is `w0=clip(B*r0,max(0,B-c1),min(c0,B))`, `w1=B-w0`; anchor weight is `1-B`. The budget is **not** the sum of positive Shapley values. Record zero-budget, `v12>=1`, each `phi>=1`, positive-phi, single-negative-phi, and nonnegative-uncapped degeneracy fractions. When both phi values are nonnegative and `v12<=1`, cap sums force the weights to equal phi. These fractions and weights are diagnostic only; no alpha/tau is applied to produce another quality comparison.

## Preread hash binding

The analyzer's `freeze` command binds its own source, tests, this protocol, root `PLAN.md`, `subsets.json`, and explicitly supplied producer/runner dependencies before any actual scalar analysis. The `analyze` command verifies all bound bytes before opening either scalar result input. Input hashes and per-clip hash receipts are preserved in the analysis output. New result output is exclusive-create; no frozen or prior output is overwritten. The producer release remains the authority for checkpoint, raw-source, model-interface, GT-firewall, and before-encoder masking hashes.
