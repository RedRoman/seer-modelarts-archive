# Seer → CVPR 2027: pre-registration (written 2026-09-27, before any CVPR-round run)

Deadline: 2026-11-16 AoE (supplementary 11-23). Nothing in this file may be changed after the
corresponding experiment has produced test numbers; changes before that are logged at the bottom.

## Ground rules
1. **Test sets are evaluated once**, on the final checkpoint of a pre-fixed schedule. Every design
   choice (iterations, loss weights, layers, CP calibration, ...) is made on validation sets only.
2. **One hardware type per experiment family.** All arms of E1, E2, ... run on the same device type
   (NPU or GPU) and the same code version; the paper states which.
3. **Seeds**: 3 per arm for E1/E4, 2 for the rest unless noted. Seeds are 0, 1, 2.
4. **Statistics**: for each split, arm differences are paired over sequences on seed-averaged
   outputs with a 10k-sample bootstrap 95% CI. An effect counts as "significant" only if the CI
   excludes 0 **and** all seeds show the same sign.
5. Every number in the paper is generated from per-sequence JSON by `tools/make_tables.py`.

## Splits
- SDSD train/test: unchanged (sorted-name 80/20; L0 = 15 indoor, L1 = 25 outdoor test sequences).
- **SDSD-val**: from the SDSD *train* split, hold out every 10th scene in sorted order of base
  name (`pairX` and `pairX_2`, `MVI_Y` and `MVI_Y_2` always together). Removed from training in all
  CVPR runs.
- **BVI-val**: from the 32 BVI-RLV *train* scenes, every 4th in sorted order (8 scenes, both fps).
- BVI test (L2) = the 8 BVI-RLV test scenes; LOL test sets (L3-L5) unchanged.
- LOL-val (for E4 only): 10% of each LOL training set, fixed by sorted name.

## Hypotheses and decision rules
| id | hypothesis | experiment | pass rule |
|---|---|---|---|
| H1 | the rewritten V-JEPA path differs from the original | P0-a | pass if a configuration reproduces the original Stage-4 zero-iter audit 27.08 ± 0.10 dB and the jepa loss 0.31 ± 0.02 on `v35_stage3/final.pt`; if nothing does within 2 working days, continue with the rewrite and state it |
| H2 | warm-starting from a V-JEPA-trained Stage 3 biases the controls | E1 (from scratch: pretrained / none / random-init V-JEPA) | report the in-domain delta; the E1 result supersedes the ICLR-round numbers |
| H3 | shallow taps carry sensor-specific signal that hurts transfer | E2 (all stages read block 23; and {17, 23}) | supported if cross-domain PSNR or LPIPS improves significantly on BVI-val without a significant SDSD-val drop > 0.3 dB |
| H4 | a 370 M pixel branch memorises SDSD through the tokens | E3 (≈20 M pixel branch × with/without V-JEPA) | supported if the with-minus-without delta on L2 is significantly larger for the small branch |
| H5 | single-dataset training cannot show transfer; multi-domain training can | E4 (leave-one-domain-out over {SDSD, BVI, LOL} × with/without V-JEPA) | **"Cross-Domain Prior" claim allowed only if** with-V-JEPA is significantly better on PSNR or LPIPS on ≥ 2 of 3 held-out domains, with no significant degradation of the other metric on those domains |
| H6 | V-JEPA helps more as a judge (feature loss on Ŷ) than as conditioning | E6 (true M2 on Ŷ, gradient through the frozen encoder) × with/without conditioning | M1/M2 kept in the paper only if the loss on Ŷ is significant on SDSD-val or BVI-val; otherwise removed and reported as negative results |

## Decision point 2026-10-18
The main line of the paper follows the H5 rule above. If it fails, the paper is written as
"in-domain fidelity + temporal consistency from a frozen video world model, and why it does not
transfer", and the title is changed (decision to be confirmed with the authors).

## Change log
- (none yet)

---

## Amendment 1 (2026-10-19, before any v2 test number existed)

The ICLR-round ablations (750 iterations, 2 seeds, same Stage-3 start) showed that three of
the four components the original paper claims as contributions do not help:

| component | L0 | L1 | L2 | reading |
|---|---|---|---|---|
| M1 forced on (g0=0.5) | -0.24* | -0.38* | -0.19 | enabling the counterfactual predictor hurts |
| M3 soft init (c=1) | +0.29* | +0.24 | -0.05 | weakening the router helps |
| w/o IGAB | -0.02 | +0.67* | +0.39 | removing it helps |
| w/o LIC | +0.33* | -0.58* | -0.24* | keep (L1/L2) |
| w/o temporal mixer+CFIM | +0.15 | -0.26 | -0.22* | keep (L2, tLP) |
| w/o Retinex aux losses | +0.16 | +0.18 | -0.33 | ~neutral, kept (training-only) |

(* = 95% bootstrap CI over sequences excludes 0.) In addition M1's gate converges to ~1e-2 and
M2's loss does not decrease during training, i.e. both are inert by construction.

**Architecture decision.** The CVPR method (`models/pixel_branch_v2.py`, "Seer-v2") keeps only
what the ablations support: V-JEPA multi-layer tokens injected through zero-init residual
cross-attention, LIC, the bottleneck temporal mixer + CFIM, and the training-only Retinex heads.
It removes M1, M2, the learnable router M3 (replaced by the fixed layer-per-scale map the router
converged to anyway) and IGAB. Trainable parameters drop from 370.4M to 361.1M.

**Effect on E1.** The three v1 runs started on 10-19 are capped at 2000 iterations and kept as the
"original architecture" reference; E1 proper is re-run with `--arch v2`. The v1-vs-v2 comparison
is reported as measured, whichever direction it goes. No test set has been touched for either.
