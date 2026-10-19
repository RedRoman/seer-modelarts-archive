# Seer → CVPR 2027: status report (2026-10-19)

Deadline 2026-11-16 AoE (supplementary 11-23). This directory holds the CVPR-round code,
the frozen data splits, and the results measured so far. Background on why the AAAI/ICLR
numbers cannot be reused is in the repository root `README.md`.

## 1. What changed in the method, and why

The ICLR-round ablations (750 iterations, 2 seeds, identical Stage-3 start) measured each
component of the original architecture. Three of the four claimed contributions do not help:

| component | L0 | L1 | L2 | reading |
|---|---|---|---|---|
| M1 forced on (g0=0.5) | −0.24* | −0.38* | −0.19 | enabling the counterfactual predictor **hurts** |
| M3 soft init (c=1) | +0.29* | +0.24 | −0.05 | weakening the learnable router **helps** |
| w/o IGAB | −0.02 | +0.67* | +0.39 | removing it **helps** |
| w/o LIC | +0.33* | −0.58* | −0.24* | keep (L1/L2) |
| w/o temporal mixer + CFIM | +0.15 | −0.26 | −0.22* | keep (L2, tLP) |
| w/o Retinex aux losses | +0.16 | +0.18 | −0.33 | ≈neutral, kept (training-only) |

PSNR change in dB vs. the full model; * = 95% bootstrap CI over sequences excludes 0.
Independently: M1's gate converges to ~1e-2 and M2's loss does not decrease during training,
so both are inert by construction rather than merely weak.

**Seer-v2** (`code/pixel_branch_v2.py`) keeps only what the ablations support — V-JEPA tokens
through zero-init residual cross-attention, LIC, the bottleneck temporal mixer + CFIM, and the
training-only Retinex heads — and drops M1, M2, the learnable router (replaced by the fixed
layer-per-scale map it converged to anyway) and IGAB. Trainable parameters: 370.4M → 361.1M.

## 2. New component: Exposure-Decoupled Conditioning (EDC)

**Diagnosis.** The datasets disagree on how much brightening the target needs (median GT/input
luma ratio): SDSD-indoor 4.5×, SDSD-outdoor 3.5×, BVI-RLV 2.8×, LOL-v1 7.2×, LOLv2-Real 6.8×.
Normalising every output to the ground-truth mean luma ("GT-mean") gains **every** method 4–7 dB
on the unseen datasets (HVI-CIDNet on L2: 18.92 → 25.75). So most of the cross-domain error is
global exposure, not structure.

This is caused by the property V-JEPA was chosen for. Illumination-invariant features carry no
information about how bright the target should be, so a model trained on SDSD silently applies
the SDSD exposure prior to LOL and comes out too dark. The invariance that motivates the prior
is also what breaks its transfer.

**Method** (`code/exposure_head.py`). Split the problem: the pixel branch predicts structure at
whatever exposure it likes, and a 0.43M-parameter head predicts the scalar log-gain separately
from (a) input statistics that do carry exposure (luma percentiles, dark-region noise) and
(b) the pooled V-JEPA token as a *scene prior* — the feature cannot report current brightness,
but scene identity constrains plausible target brightness.

**Probe result** (`code/edc_probe.py`, head trained alone, 2 seeds, MAE of log-gain):

| input | SDSD-val | BVI-train (cross-domain) |
|---|---|---|
| baseline (predict train mean) | 0.405 | 0.560 |
| statistics only | 0.451 | **0.534** |
| V-JEPA only | 0.275 | 0.717 |
| statistics + V-JEPA | **0.220** | 0.612 |

In-domain the V-JEPA prior cuts the error by 46% and beats the statistics that directly observe
brightness — an invariant feature predicting the quantity it is invariant to, through semantics.
Across datasets the learned scene→exposure mapping misfits and is worse than the baseline, while
plain statistics transfer slightly better. Read together: **the exposure prior is a property of
the target dataset and is not inferable from source data alone.** This also explains why the
original CP needed its α tuned on test data to appear effective.

**In progress** (`code/edc_calib.py`): how cheaply the residual is correctable — fit one scalar
bias on k paired target clips (k = 0,1,2,3,5), leave-k-out, and measure MAE on the rest. If k≈2
suffices, the claim becomes "cross-domain failure is localised to exposure and is correctable
with a handful of paired clips, without retraining."

## 3. Data and splits (frozen)

Historical experiments did **not** use official SDSD; they used Retinexformer's `static_np`
release (1080p static-scene subset, `pairX`/`MVI_xxxx` naming). Official `indoor_np` is
512×960 dynamic video — different content, not interchangeable. Both are now on the server;
`static_np` is the one used.

`results/splits_cvpr.json` (built by `code/make_splits.py` + `code/deleak.py`): test lists are
identical to the ICLR round (so baseline numbers remain comparable); val is held out from train
by scene (every 10th for SDSD, every 4th for BVI). Scene-level leakage present in the historical
split was removed — `pair3`/`pair3_2`, `pair34`, `MVI_1007` had the two takes of one scene split
across train and test — by dropping the train-side sequences. Final: SDSD train 136 / val 16;
BVI 48/16/16 clips.

Consistency with the ICLR round: SDSD-outdoor test 25/25 and BVI test 16/16 match exactly;
SDSD-indoor matches 9/15. The missing six are `pair50/52/58/60/68/70` ("clean-6", the sequences
no SDSD baseline trained on). They are absent from the Retinexformer release and were originally
converted from official `indoor_png` (1080p); the 512×960 `indoor_np` cannot substitute, since
evaluation centre-crops without rescaling. That download is pending.

## 4. Experiments

| run | status |
|---|---|
| E1 v1 (original architecture), 3 arms × seed 0 | capped at 2000 it as the original-architecture reference; `none` done, other two finishing |
| E1 v2 (Seer-v2), 3 arms × 3 seeds, 4000 it | queued on NPU 4/5/6, waiting for free HBM |
| EDC probe | done (table above) |
| EDC few-shot calibration | running |

Hardware: Ascend 910B on host 31051 (the only one whose CANN 910B operator package is complete).
Measured 30 s/optimiser step, 30.3 GB peak. The cards are shared with other tenants, so the
runner waits for ≥33 GB free before starting a job.

## 5. Open items

- `clean-6`: needs the six sequences from official `indoor_png`, else the clean test subset has
  to be redefined from the 9 available sequences (self-consistent, but not comparable to history).
- TLC (temporal latent consistency, using V-JEPA's predictor to constrain output dynamics) is
  designed but not implemented; it depends on how EDC resolves and on GPU availability.
- Whether "Cross-Domain Prior" survives in the title depends on the pre-registered rule in
  `notes/CVPR_PREREG.md` (H5): V-JEPA must win on ≥2 of 3 held-out domains in the
  leave-one-domain-out experiment. The ICLR-round evidence says it will not.
