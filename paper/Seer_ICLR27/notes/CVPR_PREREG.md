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
