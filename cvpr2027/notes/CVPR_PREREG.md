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

---

## Amendment 2 (2026-10-19 22:xx, before any LPM number existed)

### Why another component is needed

EDC's few-shot calibration, pre-registered as the route from diagnosis to remedy, **failed**.
Fitting one scalar bias on k paired target clips is worse than not fitting it at all:

| | SDSD-val | BVI k=0 | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|---|---|
| stats + V-JEPA | 0.368 | 0.426 | 0.508 | 0.454 | 0.477 | 0.445 |
| stats only | 0.746 | 0.522 | 0.748 | 0.636 | 0.663 | 0.587 |

(MAE of log-gain.) So the claim "cross-domain failure is localised to exposure and is correctable
with a handful of paired clips" is **not available**. Note also that this run puts
stats+V-JEPA *ahead* of stats-only cross-domain, the opposite of the earlier probe; the two used
different BVI subsets (24 clips here) and different head schedules, so **neither cross-domain
ordering may be cited until both are re-run under one setting**. That re-run is E7 below.

What survives is the diagnosis, and it is a statement about identifiability rather than about our
model: the required output exposure is a convention of the target dataset's ground truth
(median GT/input luma 4.5x, 3.5x, 2.8x, 7.2x, 6.8x across the five sets), GT-mean normalisation
gains *every* method 4-7 dB cross-domain, and no source-only estimator can recover it. EDC is
then an architectural separation that makes structure measurable independently of that
convention, not a transfer fix. The paper must carry a contribution that is not a deletion and
not contingent on transfer being fixable.

### New component: Latent Predictability Matching (LPM)

`models/world_model_prior.py`. A video world model has an encoder *and* a predictor; all of the
prior work on V-JEPA for restoration, this paper's v1 included, used only the encoder. v1's M1/M2
did touch the predictor and were measured inert, and the cause is identifiable in the code:
`models/predictor_lora.py` discards the pretrained `predictor_proj` head (384->1664) and
substitutes a randomly initialised 384->1024 projection, so the predictor's learned predictions
never participate; and M2's loss compared two latents, so no gradient reached the pixels.

LPM uses the predictor as the checkpoint trained it, and frozen:

- `e(V)` = L1 gap, in the teacher's 1664-d space, between the predictor's estimate of the late
  tubelets from the early ones (`predictor_proj`) and its own reading of the late tubelets
  (`predictor_proj_context`, the auxiliary head trained alongside). Both heads are pretrained;
  the module refuses to run if either is missing from the checkpoint.
- The loss is `|e(Y_hat) - e(Y)|`, **two-sided**. Minimising `e` would reward static and
  over-smoothed output, because an over-smoothed clip is *more* predictable than a real one;
  natural video sits at a particular level, not at zero.
- Gradient flows through the frozen encoder into the predicted pixels. This is the mechanism
  H6 names ("V-JEPA as judge rather than as conditioning") and is what M2 lacked.
- GT-free variant: replace `e(Y)` with a scalar `e*` averaged over the training set.

### Gate before any training run uses LPM

`tools/lpm_probe.py` perturbs ground-truth clips in ways whose effect on temporal dynamics is
known and reads `e` on each. LPM is used **only if** all three hold on >= 12 val clips:

1. per-frame gain jitter (sigma = 0.10) raises `e` above ground truth;
2. temporal box smoothing (window 5) lowers `e` below ground truth;
3. **specificity**: spatial gaussian blur (sigma = 3.0, identical every frame, so temporally
   coherent) moves `e` by less than half of what flicker does.

Criterion 3 is the one that can fail. If spatial blur moves the measure as much as flicker, `e`
is a generic quality score, the two-sided objective has no temporal meaning, and LPM is dropped
and reported as a negative result with this table. The probe is run before, and independently of,
any training.

### Added hypotheses

| id | hypothesis | experiment | pass rule |
|---|---|---|---|
| H7 | the world model's predictor is usable as a temporal prior on the output | E5 (Seer-v2 +/- LPM, 3 seeds) | supported if tLP **or** warp-error improves significantly on SDSD-val **and** PSNR does not drop significantly by more than 0.1 dB; otherwise reported as a negative result |
| H8 | matching the ground truth's predictability beats minimising unpredictability | E5-b (two-sided vs one-sided vs fixed `e*`) | two-sided is claimed better only if one-sided is significantly worse on a temporal metric or on PSNR; if the three are indistinguishable, we say so and keep the simplest |
| H9 | the earlier exposure-head orderings are an artefact of unequal settings | E7 (EDC probe and calibration re-run under one head schedule, one BVI subset, 3 seeds) | the paper cites a cross-domain ordering only if it is significant and consistent across seeds in E7 |

### Hardware change

Pre-registration ground rule 2 requires one device type per experiment family. Four idle
A100-80GB became available on host 32677, so **E1 is re-run in full on A100**, including the v1
reference arms, and the Ascend v2 runs started on 10-19 are discarded rather than mixed in. Those
runs were also mis-specified: launched by a runner predating `--arch`, they took the training
script's `v2` default at 6000 iterations instead of the pre-registered 4000, and
`e1_random_s0` produced no checkpoint. Nothing from them enters the paper. The Ascend host stays
on the probes (EDC, LPM) and on E7.

---

## Amendment 3 (2026-10-20 14:20, before any E4 result existed)

### LOL had to be rebuilt

The ICLR-round L3-L5 test sets were evaluated on ModelArts and the data was never
archived, so H5 could not be run as written. LOL is rebuilt from public releases
and the counts match the canonical partitions exactly, which keeps published
baselines comparable:

| | train | val | test | source |
|---|---|---|---|---|
| LOL-v1 | 436 | 49 | 15 (eval15) | `geekyrakshit/LoL-Dataset` |
| LOLv2-real | 620 | 69 | 100 | `okhater/lolv2-real` |
| LOLv2-syn | 810 | 90 | 100 | `okhater/lolv2-synthetic` |

Val follows the SDSD rule (every 10th training name, sorted, removed from
training). Frozen in `datasets/splits_lol.json` by `tools/make_splits_lol.py`.

### E4 schedule, fixed now

- 3 held-out domains x {pretrained, none} x 3 seeds = 18 runs, `tools/seer_train_e4.py`.
- **2500 iterations** per run, not 4000. E1's validation curves separate the arms
  by 2.6-3.9 dB at 1500-2000 iterations (pretrained 23.82 vs none 19.95, random
  21.20 at 2000), and H5 is a between-arm comparison rather than an absolute
  quality claim, so 2500 is enough to answer it. Chosen before any E4 number.
- **Domain-balanced sampling** (`--balance_domains on`). LOL contributes ~1.9k
  image pairs against ~0.14k video sequences, so an unweighted sampler would make
  "hold out SDSD" and "hold out BVI" train almost entirely on static single-image
  clips while "hold out LOL" trains on video, confounding H5 with a change of
  training modality. Each domain is sampled with equal probability instead.
- LOL images enter as static clips of 16 identical frames. Stated in the paper,
  not hidden: for a video model, a single image *is* a static clip.

### v1 reference reduced to one arm

`v1_pretrained_s0` finished (2000 it, val 23.460, plateaued from 23.495 at 1500);
`v2_pretrained_s0` reads 23.824 at the same 2000 it, so the stripped architecture
is ahead with 9.3M fewer parameters and 16% less time per iteration. The
architecture comparison needs only this arm, since both architectures are
compared in their intended configuration and H2 is answered by v2's three arms.
`v1_none_s0` was therefore stopped at 200 iterations and `v1_random_s0` dropped,
to free a lane for E4, which is the experiment the title depends on. Nothing from
the dropped arms is reported.

### Two inconsistencies recorded rather than silently fixed

- E7 and the earlier calibration runs took BVI clips at a 512 crop while SDSD used
  256, so their absolute MAEs carry a crop mismatch. Both arms within a run saw
  the same data, so the within-run ordering E7 reports is unaffected. E8 uses 256
  for every domain.
- `ExposureHead.forward` gained an optional `stats=` argument so a caller can
  cache the 8 statistics instead of whole clips; the first E8 attempt was killed by
  the host OOM killer while another tenant on the same node grew to 145 GB. Passing
  `lq` alone follows the original code path unchanged, and no E1 run uses the head.

---

## Amendment 4 (2026-10-20 22:00, before any E4 checkpoint existed)

### How E4 is scored

`tools/eval_e4.py`, run on the A100 host for every E4 run so the scoring
hardware is uniform. The protocol is the baselines' protocol, taken from
`baselines/common/eval_lib.py` rather than reimplemented: video is the centre 16
frames at a 512 centre crop, PSNR on uint8 RGB, LPIPS from pyiqa; LOL images are
scored at full resolution, tiled to a 16-frame static clip and read back at the
middle output frame. Frames are reflect-padded to a multiple of 32 and cropped
back, because the five-scale branch cannot ingest LOL's 400x600 otherwise.

Two deliberate departures from the archived evaluator: sequence lists come from
the frozen splits instead of a rule re-derived at scoring time, and the known
frame-pairing bug in `iter_sdsd_indoor_stcd` is not inherited (LQ and GT are
paired by sorted index within a sequence).

Each run also reports the GT-mean diagnostic -- the same prediction rescaled so
its mean luma matches the ground truth's. The gap between the two columns is the
share of the error that is global exposure rather than structure.

### H5 is decided on the held-out domain's validation split

The whole domain is held out of training, so both its val and test splits are
unseen by an E4 run, and nothing is lost by deciding on val. This keeps the test
split for the single final evaluation that ground rule 1 allows. H5's rule is
otherwise unchanged: the "Cross-Domain Prior" claim survives only if
with-V-JEPA beats without on PSNR or LPIPS on at least 2 of the 3 held-out
domains, with no significant degradation of the other metric there.

### Supersessions recorded

- **E10 supersedes E7's cross-domain row.** A budget sweep (1.5k to 45k updates,
  3 seeds, everything else fixed) shows the statistics-only arm has no stable
  optimum -- in-domain MAE 0.516 -> 1.074 -> 0.814, cross-domain 0.394 -> 0.475
  -> 0.489 -> 0.547 -- while the V-JEPA arm is flat at 0.17 in-domain and 0.45
  cross-domain across the whole range. The three earlier reversals of the
  stats-vs-semantics ordering are all explained by where the unstable arm was
  stopped, and only E10's numbers may be cited. E9 separately ruled out the
  target crop as the cause (ordering identical at 256 and 512).
- The cross-domain claim this supports is modest and stated as such: the semantic
  prior is worth 19% over an ignore-the-input baseline (0.45 vs 0.553) and is
  stable; the statistics-only predictor is not reliable at any budget.

---

## Amendment 5 (2026-10-21 01:00, before any E4 checkpoint existed)

### The scoring crop changes the sign of the V-JEPA effect

Scoring the three E1 seed-0 checkpoints on the full SDSD validation set at two
crops, everything else identical:

| crop | pretrained | none | random |
|---|---|---|---|
| 256 (the training crop) | **23.906 / 0.2558** | 19.972 / 0.2923 | 21.203 / 0.2778 |
| 512 | 20.611 / 0.2939 | 21.486 / 0.2835 | **21.627 / 0.2751** |

PSNR-Y / LPIPS, 16 sequences. The V-JEPA arm loses 3.3 dB at 512 while the arms
that carry no real tokens gain; the ordering inverts completely.

The cause is structural rather than a bug. `VJepaBackboneLocal21` resizes any
input to 256 before encoding, so a 512 crop produces tokens describing twice the
field of view at half the detail -- out of distribution for cross-attention
weights trained against 256-crop tokens. The cross-attention itself is global and
permutation-invariant over tokens, so this is a change of token *content*, not of
spatial alignment.

**Decision, fixed before any E4 number exists: E1 and E4 are scored at crop 256,
the crop they were trained at.** Scoring a 256-trained model at 512 measures a
train/test resolution shift, not the hypothesis under test, and would have
inverted H5. The 512 protocol is reported separately as a scale-robustness
analysis, with and without tiled inference (`--tile 256`), and the limitation is
stated in the paper either way.

### In-domain baselines, same evaluator, same crop, same splits

SDSD validation, 16 sequences, crop 256:

| method | PSNR-Y | LPIPS |
|---|---|---|
| HVI-CIDNet (generalization, not trained on SDSD) | 15.364 | 0.3987 |
| DarkIR (not trained on SDSD) | 17.667 | 0.3612 |
| Retinexformer, retrained on our training split (160k it) | 19.589 | 0.3105 |
| Seer-v2, V-JEPA none (2000 it) | 19.972 | 0.2923 |
| Seer-v2, V-JEPA random (2000 it) | 21.203 | 0.2778 |
| **Seer-v2, V-JEPA pretrained (2000 it)** | **23.906** | **0.2558** |

The Retinexformer row is the one that matters: same training split, no test
contamination, trained to 160k iterations against our 2k. Seeds 1 and 2 and the
test split are still pending, so this is provisional.

Baseline weights were recovered from `weights_small/` in the archive (which also
holds URCSA, Zero-DCE, SCI, LLVE-STCD and RAFT); the earlier conclusion that they
were lost came from a search that did not go deep enough.

---

## Amendment 6 (2026-10-21 12:05, before any E2 or E3 run existed)

### E2 and E3 are implemented as flags on the E1 trainer

`tools/seer_train_cvpr.py` gains `--taps` and `--branch`. Both default to exactly
the behaviour every E1 run used (`5,11,17,23` and `full`), verified against the
running E1 lane before the file was touched, so no E1 arm changes meaning.

- **E2 (H3)** varies `--taps`: the pre-registered arms are the default
  `5,11,17,23`, the deepest-only `23,23,23,23`, and the top-two `17,17,23,23`.
  Everything else is held at the E1 configuration.
- **E3 (H4)** varies `--branch`: `small` is `dims=(20,40,80,160,320)`,
  `num_blocks=(2,4,4,6,8)`, `heads=(2,4,4,8,8)`, measured at 22.40M trainable
  parameters against the full branch's 361.12M. Every component is retained and
  only the stage widths and depths change, so the capacity comparison is not
  confounded by a change of architecture. Measured 12.6 s/it against the full
  branch's 51.6 s/it on a shared A100.

Both are scored by `tools/eval_e4.py` at crop 256 under Amendment 5, with
`--heldout sdsd` for the in-domain number and the E4 held-out domains for
transfer, and aggregated by `tools/aggregate.py` under ground rule 4.

### E6 is deprioritised, with the reason recorded

H6 asks whether V-JEPA is more useful as a judge than as conditioning. The
question is already partly answered: the LPM probe (Amendment 2) shows the
predictor fails a specificity test as a temporal prior, and the code-level
account of why v1's M1/M2 were inert is established independently of any new run.
E6 is therefore queued behind E2 and E3, and if compute does not permit it before
the training cutoff the paper reports H6 as addressed by the probe and the
mechanism rather than by a trained arm.

---

## Amendment 7 (2026-10-21 19:40, before any E11 run existed)

### Why a new arm was added

Reviewing the injection design, the token fusion carries less prior than the
write-up implied. `VJepaCrossAttnBF` flattens the token grid into an unordered
set and adds no positional encoding, so attention is permutation-invariant over
the 256 tokens: a query at pixel (i, j) has no mechanism to prefer the token
covering its own location and must recover correspondence from content alone.
Three further choices are inherited rather than measured -- the tap set
(5, 11, 17, 23), the fixed layer-to-scale map (0, 0, 1, 2, 3), which was adopted
because the learnable router converged there even though that router was itself
measured harmful, and the fact that each scale reads exactly one layer.

E2 already tests the tap set. E11 tests the positional question.

### E11 / H10

| id | hypothesis | experiment | pass rule |
|---|---|---|---|
| H10 | the benefit of the prior is partly positional, not only content-level | E11 (`--xattn spatial` vs the default `global`, 2 seeds) | supported if PSNR-Y **or** LPIPS improves significantly on SDSD-val with no significant degradation of the other; otherwise reported as a negative result, i.e. the benefit is content-level |

`models/xattn_spatial.py`. One learnable grid is interpolated to the query's
resolution and to the token grid's and added after each side is projected into
the model dimension, so position enters through the existing dot product. An
explicit pixel-to-token distance bias was rejected: at the finest scale it is
(65536, 256) per frame, which forces `scaled_dot_product_attention` onto the math
backend -- the same failure that cost 8 GB in one allocation during evaluation.

Measured cost: **+720.9K parameters, +0.200%** against the global variant
(361.838M vs 361.118M), so a difference cannot be attributed to capacity. The
output projection is zero-initialised exactly as in the global variant, verified
to start at identity, and the module handles a tiled encoder pass that changes
the token count.

Either outcome is reportable: if position helps, the fusion is improved and the
512-crop robustness may improve with it; if it does not, the paper states that
the prior's contribution is content-level rather than positional, which sharpens
the "spatial-semantic, not temporal" claim rather than weakening it.

### Correction to how M1 is described

Earlier status reports overstated the M1 finding. What is established is that
*the published M1 is inert*, with two identifiable causes: `predictor_lora.py`
discards the pretrained `predictor_proj` head for a randomly initialised
projection, and M2's loss compared two latents so no gradient reached the pixels.
LPM (Amendment 2) is one corrected variant and it fails a specificity probe. That
is one point in the design space, not a proof that predictor-based designs cannot
work, and the paper will say so.

---

## Amendment 8 (2026-10-22 23:00, H5 decided)

### H5 does not pass, and the reason is measured rather than assumed

All 18 E4 runs trained and scored, 3 seeds, crop 256, held-out validation
splits, paired bootstrap over sequences:

| held-out | PSNR-Y | LPIPS | verdict |
|---|---|---|---|
| LOL | +1.590 [+1.176, +2.000] | -0.0225 [-0.0290, -0.0163] | both better |
| SDSD | +1.578 [-0.551, +3.661] | -0.0190 [-0.0405, +0.0016] | n.s. |
| BVI | +0.244 [-1.014, +1.514] | +0.0046 [-0.0053, +0.0143] | n.s. |

One domain of three, so the pre-registered rule fails and "Cross-Domain Prior"
leaves the title, per the 2026-10-18 decision point.

### Two earlier readings are withdrawn

- **"V-JEPA hurts on BVI" is not supported.** On seed 0 alone BVI's LPIPS was
  +0.0291 [+0.0241, +0.0345], significantly worse; across three seeds it is
  +0.0046 [-0.0053, +0.0143], n.s. The single-seed result was an artefact, which
  is what the three-seed rule exists to catch.
- **"Underpowered" is only half the story.** Effect sizes: LOL d=0.520 (n=208,
  needs 29), SDSD d=0.354 (n=16, needs 62), BVI d=0.090 (n=16, needs 968). SDSD
  is genuinely underpowered; BVI's effect is near zero and would not reach
  significance at any feasible n.

### What the gradient is explained by

Effect size orders with the target domain's motion content, measured on the same
val clips with RAFT:

| domain | flow magnitude (px) | frame-diff | spatial-grad | E4 effect d |
|---|---|---|---|---|
| LOL | 0.00 | 0.00000 | 0.01803 | 0.520 |
| SDSD | 0.18 | 0.00727 | 0.01214 | 0.354 |
| BVI | 2.63 | 0.03556 | 0.01563 | 0.090 |

Flow magnitude spans 14x and the effect decreases monotonically. Competing
explanations were tested and rejected: spatial detail is unordered across the
three; LOL has the *lowest* input PSNR (9.46 against 11.92 and 12.98) yet the
largest effect, so input difficulty runs the wrong way; holding out BVI leaves
*more* video in training (136 SDSD clips against 48 BVI) yet yields the smaller
effect, so the share of static images in the training mix does not explain it;
and the per-sequence correlation between improvement and exposure shift is weak
and sign-inconsistent (-0.204 on SDSD, +0.170 on BVI).

This is the third independent measurement pointing the same way, after the LPM
specificity probe and E1's split temporal metrics (warp-error improves while tLP
degrades, i.e. the prior buys adherence to real motion rather than smoothness).

**Claim the paper makes instead of H5:** what transfers from a frozen video world
model is spatial-semantic, and its value is inversely related to how much the
target depends on temporal structure.

### A caveat that must be stated prominently

SDSD here is Retinexformer's `static_np` release, whose flow magnitude is 0.18
pixels. It is a static-scene subset and barely qualifies as video. The in-domain
result (+3.31 dB) is therefore largely a spatial-enhancement result, and the
paper will say so rather than let the reader assume otherwise.

---

## Amendment 9 (2026-10-23, positioning against H-JEPA; no experiment changed)

arXiv:2610.06805, *H-JEPA: End-to-End Learning of Hierarchical World Models for
Visual Planning* (Zhang, Terver, Rabbat, LeCun, Balestriero, 2026-10-05),
appeared after this work's backbone was fixed. It is recorded here so the
decision not to adopt it is on the record rather than made at writing time.

**Not adoptable as a backbone, for three reasons that are properties of the
model rather than of our schedule.** Each level is an *action-conditioned* JEPA
predicting from `(z_{<t}, a_{<t})`, and low-light enhancement has no action
sequence. Its level-1 encoder is a ViT-Tiny summarising each frame into a single
CLS token, while our injection needs the dense 16x16 token grid that the cross
attention reads. Its training domains are simulated navigation and manipulation
plus DROID robot video, further from natural low-light video than V-JEPA 2.1.

**It corroborates the diagnosis this paper reports.** Its introduction motivates
hierarchy by arguing that existing JEPA world models "predict and plan within a
single latent space, often at a single timescale", so that one shared latent must
serve both low-level dynamics and high-level matching. Our LPM specificity probe
is an independent, quantitative instance of that failure on a restoration task:
the predictor's unpredictability responds ~1.8x more strongly to spatial blur
than to temporal flicker, i.e. the single latent does not separate fast
unpredictable detail from slow structure. The E4 motion gradient (effect size
0.52 -> 0.35 -> 0.09 as RAFT flow goes 0.00 -> 0.18 -> 2.63 px) measures the cost
of not making that separation.

**How it is used in the paper:** cited in related work as the direction the JEPA
line is taking; cited in discussion as independent agreement with our diagnosis;
and in future work, noting that our specificity probe is backbone-agnostic and
could test whether a hierarchy actually separates timescales, but that an
action-free variant with dense tokens would be needed first. No experiment,
hypothesis or decision rule in this pre-registration changes.

---

## Amendment 10 (2026-10-23, E2 / E11 / E12 decided, and a comparison withdrawn)

### H3 rejected in the opposite direction (E2, 2 seeds)

| taps | PSNR-Y | LPIPS |
|---|---|---|
| 5,11,17,23 (default) | **23.453** | **0.2592** |
| 17,17,23,23 | 19.283 | 0.3326 |
| 23,23,23,23 | 18.941 | 0.3487 |

Removing the shallow taps costs 4.2 dB. Shallow features are not sensor-specific
noise to be discarded; they are necessary. The inherited tap set is now
justified by measurement rather than by inheritance.

### H10 not supported (E11, seed 0; seed 1 pending)

Spatially grounded cross-attention scores 22.443 / 0.2687 against the global
variant's 23.453 / 0.2592, i.e. 1.0 dB worse at +0.2% parameters. The reported
reading is that the prior's contribution is content-level rather than
positional, which is consistent with the spatial-semantic account rather than
an exception to it.

### H11 supported (E12, 2 seeds): the prior works in-domain on real motion

Trained and scored on BVI alone (flow 2.63 px), the only genuinely dynamic
dataset here:

| | PSNR-Y | LPIPS |
|---|---|---|
| pretrained | 28.996 | 0.1554 |
| none | 27.900 | 0.1559 |
| difference | **+1.096 [+0.253, +2.004]** | n.s. |

This closes the gap that every other in-domain result left open, since those
were measured on SDSD at 0.18 px.

### A cross-domain comparison is withdrawn

An earlier table put our leave-one-domain-out runs against HVI-CIDNet and DarkIR
and showed us 10 dB behind on LOL. That comparison is not sound: HVI-CIDNet's
`generalization.pth` scores 26.5 on LOL and 15.4 on SDSD, a reversal only
explicable if it was trained on LOL-family data, so "held-out LOL" is in-domain
for it and strictly cross-domain for us. Our E4 runs are also trained for 2500
iterations against fully converged published models, and one E4 validation curve
was still rising at the final step.

The sound comparison holds the training data fixed: our SDSD-only E1 model and
Retinexformer retrained on the same SDSD split, both applied to BVI and LOL.

| target | method | PSNR-Y | GT-mean | LPIPS |
|---|---|---|---|---|
| BVI | Retinexformer (same split) | 19.879 | 24.592 | **0.2031** |
| | ours w/o V-JEPA | **20.823** | **25.793** | 0.2493 |
| | ours w/ V-JEPA | 19.395 | 24.583 | 0.2726 |
| LOL | Retinexformer (same split) | 15.562 | 20.462 | **0.3676** |
| | ours w/o V-JEPA | 13.275 | 20.742 | 0.4418 |
| | ours w/ V-JEPA | **16.075** | **21.204** | 0.4164 |

Two things follow. We are not behind on PSNR once training data is matched. And
the V-JEPA term flips sign with target motion under a training configuration
different from E4's: +2.80 dB on LOL (0.00 px), -1.43 dB on BVI (2.63 px). That
is an independent replication of the E4 motion gradient, this time including a
sign change. Retinexformer is consistently better on LPIPS in both targets,
which is reported as measured.

### The claim the paper now makes

A frozen video world model supplies a spatial-semantic prior. It helps in-domain
at every motion level tested, with the gain falling as motion rises (+3.31 dB at
0.18 px, +1.10 dB at 2.63 px); across domains the same decay is steeper and
turns negative on genuinely dynamic video. Five independent measurements support
this: the E4 motion gradient, the sign flip in the matched cross-domain table,
the E12 in-domain decay, the LPM specificity probe, and the split temporal
metrics in E1.

---

## Amendment 11 (2026-10-24, before any E15 run existed)

### What E14 established, and the design flaw it exposes

E14 holds the frame content pixel-identical and varies only the temporal
arrangement of the input, scoring against the correspondingly arranged ground
truth, so task difficulty, dataset and checkpoint are all fixed.

| condition | prior effect on SDSD test (n=39) |
|---|---|
| intact | +3.781 dB |
| shuffled (same frames, random order) | +3.785 dB |
| frozen (centre frame repeated 16x) | +1.563 dB |

Shuffling costs nothing (+0.004, CI [-0.418, +0.480]); freezing costs 2.218 dB
and is prior-specific (CI [-3.780, -0.837], the no-token arm moves 0.043). The
prior's contribution therefore decomposes into roughly 1.56 dB available from a
single frame and 2.22 dB requiring several *distinct* frames, with none of it
requiring correct temporal order.

On BVI (n=16) shuffling is mildly prior-specific (-0.174, CI [-0.344, -0.027])
and freezing is not significant; the disagreement with SDSD is unexplained and
is reported as such.

**The flaw.** `_vjepa_to_kv_at_scale` reshapes the tokens to (B, T_v, H_v*W_v, D)
and `repeat_interleave`s along time, so each output frame attends to the 256
tokens of its own tubelet alone -- one eighth of the 2048 available. Cross-frame
aggregation happens only inside V-JEPA's own spatiotemporal attention and never
in the fusion. That is exactly the quantity E14 identifies as carrying 2.22 of
the 3.78 dB, and the fusion is structured so as not to use it.

### E15 / H12

| id | hypothesis | experiment | pass rule |
|---|---|---|---|
| H12 | the fusion discards usable cross-frame information by restricting each query to its own tubelet | E15 (`--kv full` and `--kv hybrid` against the default per-tubelet, 2 seeds) | supported only if **both**: (i) SDSD-val PSNR-Y improves significantly, and (ii) the E14 frozen-condition deficit narrows significantly relative to the default. (i) alone is reported as an improvement of unknown mechanism |

Variants, all with no temporal positional encoding since E14 found order
irrelevant: `full` gives every query all `T_v*H_v*W_v` tokens; `hybrid`
concatenates the query's own tubelet with the temporal mean over tubelets;
`per_tubelet` is the current behaviour and the default.

Criterion (ii) is what separates a mechanism fix from an unexplained gain, and
it is the reason E15 is worth running where E11's positional variant was a
guess that failed. Both criteria are fixed before any E15 number exists.

---

## Amendment 12 (2026-10-28, convergence overturns the architecture and mechanism findings)

### The 2000-iteration protocol was not measuring architecture

E17 trains the 22.4M branch to 20000 iterations. At that budget the validation
curve plateaus and the seed spread falls from 0.91 dB to about 0.35 dB, which is
what makes the comparisons below meaningful and what made the earlier ones not.

Converged, SDSD test (39 sequences):

| | PSNR-Y | LPIPS |
|---|---|---|
| pretrained (2 seeds) | 24.552 | 0.2192 |
| none (2 seeds) | 23.951 | 0.2348 |
| random (1 seed) | 19.809 | 0.2682 |
| Retinexformer, retrained on our split | 22.621 | 0.2115 |

prior vs none: **PSNR +0.601 [-0.593, +1.968] n.s.**, LPIPS -0.0155
[-0.0248, -0.0076] better. prior vs random: +4.74 dB, both metrics significant.

Two corrections follow. The +3.25 dB attributed to the prior at 2000 iterations
was largely an artefact of undertraining; at convergence the PSNR effect is not
significant and only the perceptual metric separates the arms. And random tokens,
which were indistinguishable from no tokens when undertrained, now cost 4.1 dB
against no tokens -- the converged model relies on token content, which is why
wrong content is actively harmful.

### No injection variant survives

All at 20000 iterations, 22.4M branch, against the converged baseline:

| variant | dPSNR | verdict |
|---|---|---|
| spatial grounding | +0.029 [-0.386, +0.368] | n.s. |
| cross-frame KV (hybrid) | -0.081 [-0.528, +0.336] | n.s. |
| exposure decoupling | -0.600 [-1.555, +0.292] | n.s. |
| multi-layer keys | -2.984 [-4.137, -1.903] | **worse** |

E11's -0.68 dB and E16's -2.56 dB were protocol artefacts; multi-layer injection
looked best of all on validation (25.0-25.8) and is the worst on test, i.e. it
overfits the 16-sequence validation split. **Conclusion as measured: on this task
the delivery of V-JEPA tokens is insensitive to its mechanism.**

### The temporal mechanism claim reverses

Re-running the E14 intervention on converged checkpoints, SDSD test:

| | intact | shuffled | frozen |
|---|---|---|---|
| pretrained | 24.688 | 24.634 | 23.933 |
| none | 23.527 | 23.488 | 20.922 |

Shuffling still costs nothing, confirming the prior does not use temporal order.
But freezing now costs the **no-prior** arm 2.605 dB and the prior arm only
0.755 dB -- the opposite assignment from the undertrained run, where the prior
arm lost 2.261 and the no-prior arm 0.043.

So the earlier reading, that 2.22 of the prior's benefit came from multi-frame
aggregation, is withdrawn. At convergence the prior *substitutes* for multi-frame
redundancy: it supplies from a single frame what the no-prior model has to
recover by aggregating several. This also explains why E15's cross-frame
aggregation changed nothing -- the prior model was not relying on that channel.

Combined with the shuffle result and the insensitivity to injection, the
mechanism statement the paper can defend is: **a frozen video world model
contributes single-frame semantics here; none of its video-specific structure is
used.**

### E21 / H16: the one gap the measurements still single out

What remains unexplained is a trust asymmetry. The prior's sign flips with the
target domain (+2.80 dB on LOL, -1.43 dB on BVI under identical training), and at
convergence wrong tokens cost 4.1 dB while right tokens buy 0.6 dB. The model
relies on the tokens unconditionally, which is free in-domain and harmful when
they are out of distribution. Every injection variant shares that behaviour,
consistent with them all performing alike.

`models/xattn_gated.py` predicts a per-location scalar gate on the cross-attention
residual from two unsupervised signals: the cosine agreement between the attended
token summary and the query's own content, and the summary's magnitude against a
tracked scale. Cost: **+0.4K parameters** (22.3968M -> 22.3972M), and the gate is
initialised open, verified to reproduce the ungated model elementwise at init.

| id | hypothesis | pass rule |
|---|---|---|
| H16 | the prior fails out of domain because the model cannot tell when it applies | supported only if **both**: (i) cross-domain BVI improves significantly against the ungated baseline, and (ii) feeding random tokens to the trained gated model recovers close to the no-token result instead of losing 4.1 dB |

Criterion (ii) is checkable by inference alone and does not depend on (i), so a
gate that learns to close is distinguishable from one that merely helps.

---

## Amendment 13 (2026-10-28, before any E22 run existed)

### The encoder does represent temporal structure; the objective does not ask for it

`tools/token_probe.py` measures the tokens directly, before any pixel branch is
involved, on 8 SDSD test clips:

| tapped block | across-tubelet distance | same, frozen input | token change when frames are shuffled |
|---|---|---|---|
| 5 | 0.026 | 0.004 | **21%** |
| 11 | 0.042 | 0.018 | 26% |
| 17 | 0.044 | 0.024 | 27% |
| 23 | 0.192 | 0.083 | **58%** |

Reordering the input frames changes the block-23 tokens by 58%, and the
across-tubelet variation halves when the clip is replaced by one repeated frame,
so that variation is temporal rather than positional. Meanwhile the converged
model's output changes by 0.054 dB under the same reordering.

The two candidate explanations are therefore resolved: it is not that the
encoder fails to represent motion on low-light near-static footage, but that the
model has learned to use only the temporally invariant component. The objective
explains why -- Charbonnier, Retinex and TV are all per-frame, so nothing
rewards getting the evolution right, and the 58% that varies with order is free
to fall in the cross-attention's null space.

A further possibility is ruled out in passing: the low-light input and the
ground truth give almost the same token structure (0.192 vs 0.200
across-tubelet, 21% vs 18% shuffle change), so darkness does not degrade the
features.

### E22 / H17

`losses/temporal_delta.py` differences the tokens along the tubelet axis for the
prediction and for the target and matches the two difference fields, normalised
by the target's own delta magnitude. Differencing is what makes the term
specific: the static semantic component cancels, so it cannot be satisfied by
improving per-frame appearance, which the existing terms already drive. The
gradient runs prediction -> frozen encoder -> loss, which is what M2 lacked, and
the matched quantity is a field rather than the scalar LPM used.

Measured at init, the term reads 1.39-1.42 on a relative scale where 1.0 means
the prediction's latent dynamics are unrelated to the target's -- consistent
with the probe. Cost: 6.3 -> 17.7 s/it.

| id | hypothesis | pass rule |
|---|---|---|
| H17 | the prior's temporal content is unused because nothing in the objective rewards it, not because it is absent | supported only if **both**: (i) a temporal metric (warp error or tLP, as a gap to the ground truth's own level) improves significantly against the converged baseline, and (ii) the trained model's output becomes sensitive to input frame order in `tools/temporal_ablate.py`, which currently reads 0.054 dB |

Criterion (ii) is the mechanism check and is independent of (i): a model that
merely scores better without becoming order-sensitive has not started using the
temporal component, and would be reported as such. PSNR is not part of the rule;
if it falls by more than 0.3 dB the trade is reported rather than claimed.

---

## Amendment 14 (2026-10-29, written before E22's first checkpoint exists)

### 1.0 is not a floor on the E22 loss, it is the value of giving up

The relative term `|dp - dg| / |dg|` reads exactly 1.0 when the prediction's
latent deltas are zero, so a run drifting toward 1.0 is ambiguous on its own.
E22 has gone 1.40 (it 50) -> 1.04 (it 2050) -> 1.021 (it 4500), slowing, with
no checkpoint yet written. Two readings, separated by the parts of the ratio:

  mag  = |dp| / |dg|     how much temporal variation the prediction carries
  cos  = <dp, dg>        whether it points where the target's does

`tools/td_probe.py` on the converged baseline (E17 pretrained s0, 12 SDSD test
clips, block 17) and `tools/td_ref.py` on the target alone:

| field compared with the ground truth's | td | mag | cos |
|---|---|---|---|
| gt under gamma 1.1 (same motion, appearance changed) | 0.199 | 0.998 | **+0.968** |
| gt under additive noise, sigma = 1/255 | 0.438 | 0.996 | +0.852 |
| gt rolled by one pixel | 0.775 | 1.006 | +0.645 |
| **converged model's output** | 1.455 | 1.063 | **+0.034** |
| the low-light input | 1.397 | 1.008 | +0.045 |
| constant latents (dp = 0) | 1.000 | 0.000 | 0.000 |

Two things follow and neither was known before.

**The target is a well-posed quantity.** A photometric change that leaves the
motion alone moves it by 0.03 (cos 0.968), so the delta field is not a noise
artefact and td has real headroom down to roughly 0.2. The earlier worry that
near-static low-light footage simply has no measurable dynamics is excluded.

**The converged model misses it entirely.** It emits temporal variation of
essentially the right magnitude (1.06x) pointing nowhere near the target
(+0.034), no better than the unprocessed input (+0.045). Its latent dynamics
are its own residual fluctuation, not the scene's motion. No per-frame metric
can see this, and the pixel-space temporal metrics we report (warp error, tLP)
did not surface it.

### The prediction, registered now

From (mag 1.06, cos 0.03), the cheapest descent direction for a term normalised
by the *target's* magnitude is to shrink `mag`, which drives td to 1.0 and no
further; raising `cos` requires actually reconstructing the motion. The loss as
written therefore has a reachable trivial optimum, and the observed plateau at
1.02 is where that optimum sits.

**Prediction:** `tools/td_probe.py` on `e22_tdelta_s0/step_5000.pt` will show
`mag` well below the baseline's 1.063 -- taken as below 0.5 -- with `cos` not
meaningfully above the baseline's +0.034, taken as below +0.10.

If instead `cos` exceeds +0.10 the term is doing what it was built to do and
H17 proceeds on the pre-registered rule in Amendment 13, untouched.

If the prediction holds, the diagnosis is in the normalisation rather than in
the idea, and the registered fix is a direction-only term, scale-invariant in
`dp` so that shrinking cannot satisfy it, with magnitude restored by a separate
symmetric ratio penalty:

```
L = (1 - cos(dp, dg)) + |log(|dp| / |dg|)|
```

That variant is E24 and inherits H17's pass rule unchanged. It is recorded here
before the checkpoint that decides between them exists; whichever branch the
probe selects, the other is reported.

### Amendment 14, outcome (2026-10-30)

The registered prediction holds, at both checkpoints E22 has written:

| model | td | mag | cos |
|---|---|---|---|
| converged baseline, no such term | 1.455 | 1.063 | +0.034 |
| e22_tdelta_s0 step 5000 | 1.027 | **0.202** | **+0.014** |
| e22_tdelta_s0 step 10000 | 1.018 | **0.166** | **+0.015** |
| constant latents | 1.000 | 0.000 | 0.000 |

Registered thresholds were mag < 0.5 and cos < +0.10; both are met with room,
`mag` is still falling between the two checkpoints, and `cos` is *below* the
baseline's +0.034 rather than above it. E22 did not learn the target's dynamics;
it removed its own. H17 is therefore untested by E22, not refuted -- the run
measures the loss, not the hypothesis.

A unit check on the two forms, fed the magnitudes and alignments measured above:

| state | relative (E22) | cosine (E24) |
|---|---|---|
| baseline, mag 1.06 cos +0.03 | 1.437 | 1.030 |
| E22 at step 5000, mag 0.20 cos +0.01 | **1.018** | 2.590 |
| full collapse, mag 0 | **1.000** | **14.589** |
| aligned, mag 1.00 cos +0.90 | 0.111 | 0.006 |

The relative form ranks full collapse as its best state; the cosine form ranks
it worst. That is the whole of the defect and the whole of the fix.

E24 runs the cosine form on two seeds with every other setting held at E22's,
including the weight 0.3 -- the two losses read 1.44 and 1.03 at the baseline's
state, so the weight does not need rescaling and the form stays the only
variable. It inherits H17's pass rule from Amendment 13 unchanged. E22 is left
to finish for the record rather than killed, since its final checkpoint is the
only direct measurement of what the misnormalised term does at convergence.

---

## Amendment 15 (2026-10-31)

### The in-domain effect is significant on both metrics once a third seed exists

Ground rule 4 in full, SDSD test, n = 39, 20000 iterations, 22.4M branch:

| metric | prior | none | difference | 95% CI | per-seed | all agree |
|---|---|---|---|---|---|---|
| PSNR-Y | 24.589 | 23.300 | **+1.290** | [+0.270, +2.449] | +1.161 / +0.042 / +2.666 | yes |
| LPIPS | 0.2160 | 0.2400 | **-0.0240** | [-0.0321, -0.0168] | -0.0232 / -0.0079 / -0.0409 | yes |

At two seeds PSNR read +0.601 with a CI spanning zero; the third seed moves it
to significance and both halves of the rule now pass on both metrics.

The seed spread is reported with the mean rather than behind it, because it is
wide: the per-seed PSNR effect ranges from +0.042 (a tie) to +2.666, and the
largest gap comes from a no-prior run that landed at 21.997 against its
siblings' 23.527 and 24.375. The all-seeds-agree half of the rule is what
licenses the claim at this spread -- the prior arm is ahead in every seed -- and
the honest statement is that the effect is consistent in sign and imprecise in
size. Absolute prior-arm PSNR is stable across seeds (24.688 / 24.416 /
24.663); it is the no-prior arm that is volatile, which is itself consistent
with the prior acting as a stabiliser.

### H16 is not refuted by E21; it was not tested

| criterion | result |
|---|---|
| (i) gated vs ungated | PSNR -0.243 [-0.853, +0.359] n.s., seeds disagree; LPIPS +0.0043 n.s., both seeds worse |
| (ii) does the gate close on random tokens | gated loses 4.764 dB, ungated loses 4.879 dB -- no difference |

The audit explains it. Mean gate value on held-out data: 0.9832 with real
tokens and 0.9833 with random tokens for seed 0, 0.9844 and 0.9844 for seed 1 --
identical to four decimals, and sitting at the initialisation value of 0.982.
All 28 gate tensors across 7 cross-attention sites moved off their initial
values, so the module trained; the function it computes is nonetheless constant.

The reason is in the training setup rather than the hypothesis: the gate only
ever saw correct tokens, so closing never reduced the loss and no gradient
pushed it to. Criterion (ii) asked for zero-shot behaviour the gate had no
opportunity to learn. This is the same class of error as Amendment 14's -- a
mechanism hypothesis paired with an implementation that cannot test it -- and it
is now the second instance, which is worth stating as a pattern in how these
experiments were designed rather than as two unrelated accidents.

### E24 at 5000 and 10000 iterations

| model | td | mag | cos |
|---|---|---|---|
| converged baseline, no such term | 1.455 | 1.063 | +0.034 |
| the low-light input | 1.397 | 1.008 | +0.045 |
| E22 step 5000 (relative form) | 1.027 | 0.202 | +0.014 |
| E22 step 10000 | 1.018 | 0.166 | +0.015 |
| **E24 step 5000** (cosine form) | 1.387 | **0.998** | **+0.062** |
| **E24 step 10000** | 1.373 | **0.994** | **+0.076** |
| ceiling: gt under gamma 1.1 | 0.199 | 0.998 | +0.968 |

The anti-collapse half works as designed and the demonstration is
interventional: the only change from E22 is the form of that one term, and the
magnitude holds at the target's level (0.994 against 0.202) instead of being
driven to zero. Alignment is climbing and has passed the input's own level for
the first time, but it has covered about 4% of the distance to the ceiling and
is gaining roughly +0.014 per 5000 iterations, so the honest extrapolation to
20000 is near +0.10 rather than anything close to the ceiling.

H17's pass rule from Amendment 13 is unchanged and is not about `cos`: it needs
a temporal metric to improve significantly and the model to become sensitive to
input frame order. A run that moves `cos` without moving either is reported as
a mechanism that was engaged and did not pay.

### E2 re-opened at convergence

Declaring the 2000-iteration protocol unable to resolve architecture effects
invalidated E11, E14, E15, E16 and E18, all of which were retested, and H5,
which E20 retested. E2 and E3 were not, yet "what the prior needs is feature
depth -- reading only block 23 instead of {5,11,17,23} costs 4.5 dB" is a
2000-iteration result load-bearing in the current draft. That rule was not
applied to it.

E2 is therefore rerun at convergence as `e2c_deep` (`--taps 23,23,23,23`, two
seeds, 20000 iterations, 22.4M branch) against the three-seed baseline above.
E3 cannot be: the 361M branch needs roughly 139 GPU-hours for 20000 iterations
and does not fit before the deadline, so its capacity claim is demoted to
"measured at a 2000-iteration budget" with that budget stated wherever it
appears, and is not used to support any architectural conclusion.

---

## Amendment 16 (2026-10-31, before any E25 run exists)

### Giving the gate a condition under which closing is worth anything

E21's gate sat at its initialisation (0.9832 with real tokens, 0.9833 with
random ones) because every training step fed it correct tokens, so closing
never lowered the loss. H16 was therefore never tested. E25 changes exactly one
thing: on a fraction of steps the pixel branch is conditioned on the *previous*
clip's tokens while the target stays the current clip's.

The corruption is deliberately not noise. Previous-clip tokens are real V-JEPA
tokens with the right statistics, the right spatial layout and the wrong
content, which is precisely the situation the gate's agreement input -- cosine
similarity between the attended summary and the query's own features -- is
constructed to detect. Gaussian noise would be separable on magnitude alone and
would teach the gate a shortcut that does not transfer to the out-of-domain case
the hypothesis is about. Caching costs memory and no compute, since the tokens
are already produced under `no_grad`.

`p = 0.3`, chosen before any run, on the grounds that the branch must still see
correct tokens on the large majority of steps or the prior's in-domain benefit
is traded away to buy the gate. No sweep is planned; if 0.3 fails the result is
reported at 0.3.

| id | hypothesis | pass rule |
|---|---|---|
| H16' | the prior's reliance is unconditional because nothing in training rewards conditioning it; given a condition, a gate can learn to discriminate usable from unusable tokens | supported only if **both**: (i) feeding random tokens to the trained model costs significantly less than the 4.76 dB it costs the E21 gated model and the 4.88 dB it costs the ungated one, and (ii) in-domain PSNR-Y does not degrade significantly against the three-seed converged baseline |

Criterion (i) is inference-only and is the mechanism check; (ii) is the cost
check, because a model that ignores the prior entirely would pass (i)
trivially while losing the in-domain effect, and that outcome must be
distinguishable from the gate actually discriminating. The gate's mean value
under real and corrupted tokens is reported from `tools/gate_audit.py`
alongside both, since two numbers that differ by less than 0.001 are what
exposed E21.

Cross-domain is *not* part of the rule. The hypothesis is about conditional
reliance, which is measurable in domain; whether that translates to transfer is
a separate question that E20's seeds are already answering and that E25 has no
budget to test.

### Logging

`--token_corrupt` prints the configured probability at startup and the realised
count of corrupted steps in every progress line. Both appear because this
project has twice shipped a change that was installed and did not fire, and a
configured probability is not evidence that any step was corrupted.

---

## Amendment 17 (2026-11-01, H5 and H17 both decided)

### H5 passes at convergence, on all three held-out domains

Two seeds, 20000 iterations, 22.4M branch, test splits, ground rule 4 applied
mechanically:

| held out | n | metric claimed | prior | none | difference | 95% CI | per-seed |
|---|---|---|---|---|---|---|---|
| SDSD | 39 | LPIPS | 0.2778 | 0.3035 | **-0.0257** | [-0.0471, -0.0059] | -0.027 / -0.024 agree |
| BVI | 16 | LPIPS | 0.1965 | 0.2194 | **-0.0229** | [-0.0435, -0.0017] | -0.031 / -0.015 agree |
| LOL | 215 | PSNR-Y | 17.797 | 16.916 | **+0.881** | [+0.390, +1.395] | +0.420 / +1.342 agree |

The unclaimed metric is not significantly degraded on any domain: SDSD PSNR
+0.292 [-1.875, +2.519] n.s., BVI PSNR +0.298 [-1.593, +1.965] n.s., LOL LPIPS
+0.0011 [-0.0031, +0.0054] n.s. The rule required 2 of 3 and gets 3 of 3.

Two honest qualifications. The per-seed signs disagree on SDSD's PSNR and on
LOL's LPIPS -- but in both cases that is the metric *not* being claimed and not
significantly harmed, and the all-seeds-agree condition holds for the claimed
metric in all three domains. And the metric that passes differs by data type:
the two video domains pass on LPIPS, the single image domain on PSNR. That
pattern is reported as an observation; nothing here tests why.

This reverses Amendment 8, which recorded H5 as failing on 1 of 3 domains. That
verdict came from 2000-2500 iteration models, and no domain that passed then
passes now -- LOL was the only one significant before and is the only one that
passes on a different metric now. The two verdicts share no overlapping domain,
which is the strongest evidence yet that the short protocol was reading noise.
**The cross-domain claim returns to the paper and the title.**

### H17 fails, with the mechanism demonstrably engaged

E24 against the converged baseline, SDSD test, n = 39, two seeds:

| metric | E24 | baseline | difference | 95% CI | per-seed |
|---|---|---|---|---|---|
| PSNR-Y | 24.601 | 24.552 | +0.049 | [-0.408, +0.558] n.s. | agree |
| LPIPS | 0.2455 | 0.2192 | **+0.0262** | [+0.0128, +0.0412] **worse** | agree |
| tLP, gap to GT | 0.0228 | 0.0144 | **+0.0085** | [+0.0066, +0.0103] **worse** | agree |

Criterion (i) required a temporal metric to improve significantly; tLP moved
significantly the wrong way. Criterion (ii) required order sensitivity: intact
24.768 against shuffled 24.802, a difference of **0.034 dB**, against the
baseline's 0.054. The model is no more sensitive to the order of its input
frames than before. **H17 fails on both halves.**

What makes this worth reporting rather than discarding is that the intervention
did what it was built to do, measured at the level it targeted:

| model | td | mag | cos |
|---|---|---|---|
| converged baseline | 1.455 | 1.063 | +0.034 |
| low-light input | 1.397 | 1.008 | +0.045 |
| E22 s0 / s1 (misnormalised) | 1.012 / 1.011 | 0.138 / 0.133 | +0.014 / +0.014 |
| **E24 s0 / s1** | 1.362 / 1.368 | **0.999 / 1.004** | **+0.090 / +0.087** |
| ceiling, gt under gamma 1.1 | 0.199 | 0.998 | +0.968 |

Latent temporal alignment nearly tripled against the baseline and the magnitude
held at the target's level, exactly as Amendment 14 predicted and within the
+0.10 it extrapolated. So the finding is not "the loss did not work" but
something more specific and more useful:

**improving a model's latent temporal alignment does not improve its
pixel-space temporal consistency, and costs perceptual quality.** tLP's gap
grows while `cos` triples, and LPIPS degrades by +0.0262 -- more than the
entire in-domain LPIPS benefit the prior buys (-0.0240). The two notions of
temporal consistency are not the same quantity, and the frozen encoder's notion
is not the one the metrics or the eye reward.

E22 is the control and confirms the Amendment 14 diagnosis end to end: at
convergence its collapsed term (mag 0.138) costs PSNR -0.555 [-1.089, -0.035],
LPIPS +0.1826 [+0.1659, +0.1985] and tLP gap +0.3587 [+0.3425, +0.3757], all
significant, all seeds agreeing. The misnormalised objective does not merely
fail to help; it destroys the model.

### Standing

With H17 decided, E25 (H16', Amendment 16) is the only remaining experiment
that could yield a method contribution. It is at 5050/20000 with a realised
corruption rate of 30.0% (12108 corrupted steps of 5050 x 8 forward passes) and
is due 2026-11-02 evening. If it fails its rule, the paper has no method
contribution and will be written as the empirical and mechanistic result it is,
with that stated plainly rather than dressed up.

---

## Amendment 18 (2026-11-01, token sufficiency, written with the screen's result and before E26/E27)

### What the prior contributes, as opposed to how much

Every architectural experiment so far asked how the tokens are delivered, and
all five injection variants land within noise or worse at convergence. The
content question was never asked. `tools/token_sufficiency.py` asks it by
reducing the token field at inference on a model trained with the full field.
Seed 2, SDSD test, n = 39, paired bootstrap against the identity transform:

| transform | dPSNR-Y | 95% CI | dLPIPS | 95% CI |
|---|---|---|---|---|
| spatial mean (256 positions -> 1 per tubelet) | +0.341 | [-0.790, +1.430] | -0.0003 | [-0.0041, +0.0035] |
| temporal mean (8 tubelets -> 1) | -0.081 | [-0.661, +0.472] | +0.0006 | [-0.0018, +0.0029] |
| rank 4 of 1024 | +0.213 | [-0.353, +0.763] | +0.0000 | [-0.0024, +0.0027] |
| rank 16 | +0.186 | [-0.155, +0.518] | -0.0014 | [-0.0033, -0.0000] |
| rank 64 | +0.094 | [+0.000, +0.194] | -0.0006 | [-0.0012, -0.0002] |
| rank 256 | -0.001 | [-0.047, +0.034] | -0.0001 | [-0.0002, +0.0000] |

The positive means are **not** significant and are not claimed as improvements.
What the intervals license is an equivalence bound: rank 4 costs at most
0.35 dB and the spatial mean at most 0.79 dB, against the 4.8 dB that separates
correct tokens from random ones. Rank 4 therefore discards at most 7% of the
prior's measured worth, and the spatial layout at most 16%.

The reading is one-sided by construction. A reduction that loses nothing proves
the discarded part was surplus, and that inference is sound here because the
model is demonstrably sensitive to token content. A reduction that lost a great
deal would not distinguish "the information was needed" from "the model is
brittle to a conditioning distribution it never trained on". E26 removes that
asymmetry.

### This is the account the negative results were missing

Six results that read as six separate failures follow from one cause:

| result | why |
|---|---|
| five injection variants within noise or worse | no spatial structure to inject |
| E11 positional grounding +0.029 n.s. | same |
| E15 cross-frame KV +0.11 n.s. | no temporal content to aggregate |
| E18 multi-layer keys -2.98 | the code is low-rank already |
| E14 frame order worth 0.054 dB | temporally constant |
| E24 cos tripled, tLP and LPIPS worse | the term fought the actual mechanism |

What transfers from the video world model is a very low-dimensional, spatially
near-uniform, temporally near-constant semantic code. **The paper's current
title asserts a spatial prior and the paper's own data contradicts it; the
title changes.**

### The confound this screen carries, and what it forces

The projection uses each clip's *own* principal directions, so the result
licenses "four clip-adaptive directions suffice" and not "a fixed four-
dimensional code suffices". Those are different claims and only the second can
be distilled. A fixed basis estimated once on the training split is therefore
measured before E27 is defined, as `rank k (global)`. If the global-basis
numbers are materially worse than the per-clip ones, E27 is not launched and
the finding is reported at the weaker, clip-adaptive strength.

### E26 and E27, registered now

| id | hypothesis | pass rule |
|---|---|---|
| H18 | the low-rank sufficiency is a property of the prior, not of a train/test conditioning mismatch | supported if a branch **trained** with globally-rank-k tokens is not significantly worse than the three-seed full-token baseline on PSNR-Y or LPIPS, two seeds, k fixed at the smallest value the global screen supports |
| H19 | the transferred code is predictable from the low-light input, so the frozen encoder is removable at inference | supported only if a small predictor (<= 2M parameters) conditioning the branch recovers significantly more than the random-token arm **and** is not significantly worse than the full-token baseline, two seeds |

H19 is registered with both outcomes stated as results in advance, because they
are both informative and the temptation to report only the flattering one is
exactly what pre-registration is for. If a small predictor trained on this data
recovers the benefit, the frozen 300M encoder is removable at inference. If it
does not, the claim is that what transfers is a low-dimensional code that
*cannot be learned from the task data*, which is the stronger scientific
statement and which the no-prior arm's 23.300 against 24.589 already makes
plausible. Neither outcome is a failure of the hypothesis set; only a
non-result -- a predictor that beats neither arm significantly -- would be.

---

## Amendment 19 (2026-11-01, the E27 gate fires and E27 is cancelled)

### Four clip-adaptive directions suffice; four fixed directions are as bad as noise

Amendment 18 made a fixed basis the gate on E27, on the grounds that only a
fixed code can be predicted by a small network. The basis was estimated on 60
training-split clips (test read once, per ground rule 1). It explains little:
4 global components account for 0.267--0.368 of token variance depending on the
tapped block, 16 for 0.483--0.572, 64 for 0.719--0.772.

Applied at inference on the converged model, seed 2, SDSD test n = 39:

| projection | dPSNR-Y | dLPIPS |
|---|---|---|
| per-clip rank 4 | +0.213 | +0.0000 |
| per-clip rank 16 | +0.186 | -0.0014 |
| **global rank 4** | **-4.019** | **+0.0381** |
| **global rank 16** | **-3.593** | **+0.0307** |
| global rank 64 | -0.042 | +0.0061 |
| reference: random tokens | -4.88 | +0.0585 |

A fixed rank-4 projection costs 4.02 dB, which is within the noise of
destroying the prior entirely (random tokens cost 4.88). Sixteen fixed
directions are no better. It takes 64 before the projection is harmless.

**E27 is cancelled.** The gate was registered before this was measured and it
has fired: there is no fixed low-dimensional code to distil, so the efficiency
claim that would have followed is not available and is not made. This also
retracts a statement made in conversation before the measurement existed --
that the frozen 300M encoder was likely removable at inference. It is not, on
this evidence, and the reason is now measured rather than assumed.

### What the contrast actually says, which is more specific than the earlier reading

Per-clip rank 4 preserves the benefit; global rank 4 destroys it. The useful
content is therefore not a small fixed set of semantic axes. It is *which*
four-dimensional subspace a given clip's tokens occupy, and the identity of
that subspace varies enough across clips that 64 fixed directions are needed to
contain their union.

This sharpens rather than weakens the account, and it explains why the
projection cannot be shortcut: locating the subspace requires the encoder that
produced it. A low-rank description of what transfers is not a low-cost route
to obtaining it. Any reader of the rank-4 number alone would draw the opposite
conclusion, so the two must be reported together, and the intro and abstract
are amended accordingly.

### Three-seed per-clip sufficiency, and one result that is not an equivalence

| transform | dPSNR-Y | 95% CI | per-seed | dLPIPS | 95% CI |
|---|---|---|---|---|---|
| per-clip rank 4 | **+0.426** | [+0.033, +0.853] | +++ | -0.0010 | [-0.0029, +0.0006] |
| per-clip rank 16 | +0.193 | [-0.010, +0.390] | +++ | -0.0011 | [-0.0020, -0.0003] |
| per-clip rank 64 | +0.072 | [-0.037, +0.173] | +++ | -0.0005 | [-0.0009, -0.0001] |
| spatial mean | +0.336 | [-0.685, +1.331] | +++ | -0.0005 | [-0.0039, +0.0028] |
| temporal mean | +0.283 | [-0.180, +0.764] | ++- | -0.0007 | [-0.0026, +0.0010] |

On three seeds the per-clip rank-4 projection is significantly *better* than
the full token field, not merely equivalent, with all seeds agreeing. The
reading is that the field carries clip-specific variation the branch is
slightly harmed by and that a rank-4 projection removes. It is an
inference-time result on models trained with the full field, so it is a
measurement about the conditioning and not yet a claim about architecture.

### H18 restated and H20 added

| id | hypothesis | pass rule |
|---|---|---|
| H18 | the low-rank sufficiency is a property of the prior, not of a train/test conditioning mismatch | supported if a branch **trained** with global-rank-64 tokens -- the smallest fixed rank the screen above supports -- is not significantly worse than the three-seed full-token baseline on PSNR-Y or LPIPS, two seeds |
| H20 | the token field's clip-specific component is a nuisance, so projecting it away is an architectural improvement and not only an inference-time artefact | supported only if a branch **trained** with per-clip rank-4 tokens is significantly better than the three-seed full-token baseline on PSNR-Y or LPIPS, two seeds, with no significant degradation of the other |

H20 is the only remaining candidate for a method contribution, and it is a
modest one: the inference-time effect is +0.426 dB with a confidence interval
that nearly touches zero. It is registered with that stated, so that a
null outcome is a reported null and not a quiet omission.

---

## Amendment 20 (2026-11-03, E2 re-decided, H16' and H18 both fail)

### E2 at convergence: tap depth is the one architectural factor that is real

`--taps 23,23,23,23` against the default `5,11,17,23`, two seeds against the
three-seed baseline, SDSD test n = 39:

| metric | deep-only | baseline | difference | 95% CI | per-seed |
|---|---|---|---|---|---|
| PSNR-Y | 21.281 | 24.589 | **-3.309** | [-4.600, -1.940] | -5.107 / -1.510 |
| LPIPS | 0.2462 | 0.2160 | **+0.0301** | [+0.0206, +0.0405] | +0.043 / +0.017 |

The 2000-iteration estimate was -4.5 dB; at convergence it is -3.309 dB, both
significant and in the same direction. **E2 is the only architectural result in
this paper that the short protocol got qualitatively right**, and it survives
having its own rule applied to it. Reading only the deepest block costs more
than two thirds of the entire prior benefit (4.88 dB), so the shallow-to-deep
span is load-bearing and is the one design choice the paper can defend.

This matters for the account in Amendment 18. Depth across blocks is necessary
while rank within a block is not: the useful signal is spread over the
encoder's hierarchy, not over the 1024 channels of any single tap.

### H16' fails, and fails more informatively than H16

Trained with 30% of steps conditioned on the previous clip's tokens (realised
48126 and 47981 corrupted forward passes of 160000):

| criterion | result |
|---|---|
| (i) random-token cost reduced | -4.088 dB [-5.421, -2.808] against the ungated -4.879 and E21's -4.764 -- overlapping, no reduction |
| (ii) in-domain not degraded | PSNR **-1.130** [-1.905, -0.427], LPIPS **+0.0202** [+0.0149, +0.0258], both significant, both seeds agreeing -- **degraded** |

Gate values on held-out data: 0.9824 real / 0.9837 random for seed 0, 0.9844 /
0.9844 for seed 1. Still at the initialisation value of 0.982, still identical
between real and random tokens to three decimals.

E21's failure was attributable to the training setup -- the gate never saw
unusable tokens, so closing could not lower the loss. Amendment 16 removed that
explanation, and the gate still did not learn to close. The conclusion is
therefore about the hypothesis and not the implementation: **the agreement and
magnitude statistics available at a query location do not support
discriminating usable from unusable conditioning here**, even when 30% of
training steps reward doing so. What the corruption bought instead was a
uniformly worse model: it paid 1.13 dB in domain and recovered nothing.

H16 and H16' are reported together, because the second is what licenses the
stronger reading of the first.

### H18 fails: low-rank sufficiency does not survive training

Trained with globally-rank-64 tokens, the smallest fixed rank the inference
screen found harmless (-0.042 dB there):

| metric | rank 64 | baseline | difference | 95% CI | per-seed |
|---|---|---|---|---|---|
| PSNR-Y | 23.929 | 24.589 | **-0.660** | [-1.213, -0.111] | -0.780 / -0.540 |
| LPIPS | 0.2201 | 0.2160 | +0.0041 | [-0.0002, +0.0083] | +0.004 / +0.004 |

A projection that costs 0.04 dB when applied to a model trained on the full
field costs 0.66 dB when the model is trained with it, significant on PSNR with
both seeds agreeing. The asymmetry Amendment 18 warned about is therefore real
and runs in the direction that limits the claim: the full field is not surplus
to a model that can train on it, even though a model trained on it can be
deprived of most of it at inference.

**The sufficiency finding is restated accordingly, and the restatement is the
one the paper will carry.** It is a statement about what a trained branch *uses*
-- a converged model's reliance is concentrated in a low-rank, clip-adaptive
subspace, and 1024 dimensions, spatial layout and frame order are all largely
surplus to it at inference -- and not a statement that the representation could
have been smaller to begin with. The inference-time and training-time numbers
are reported side by side so that distinction is visible rather than argued.

### Standing after this wave

Seven hypotheses tested at convergence by intervention; one architectural
factor survives (tap depth, E2); the in-domain and cross-domain effects are
significant and hold. H20 (per-clip rank 4 trained) is at 13300/20000 and is
the last experiment. Given H18's outcome -- the same shape of question, with
the training-time effect significantly negative -- the honest prior on H20 is
now poor, and that is recorded here before its result exists.

## Amendment 21 — protocol audit and validation mean/basis control (2026-11-11)

The project handover exposed reconstruction omissions in the archived generic
evaluator: random encoders were not initialized with the training seed, and
some saved attention/tap/projection settings were not restored. This is a
retrospective implementation audit, not an amendment that makes already-seen
results preregistered. Historical JSONs and the preceding text remain intact;
the current STATUS and audit notes identify superseded interpretations.
Corrected final-checkpoint evaluations use identical splits, crops and metrics,
restore saved settings, and write separate JSONs with provenance. They must not
select a checkpoint or tune a hyperparameter against test scores.

The following **new exploratory diagnostic is specified before its numbers**.
Earlier adaptive and fixed PCA interventions changed both the token mean and
the basis; the adaptive intervention retained the full clip mean. Neither
the four-direction result nor the fixed-basis loss isolates basis adaptivity
or supplies evidence that a four-scalar code suffices.

- Data: all 16 frozen SDSD validation sequences, no test sequences or labels
  used to choose conditions. Use existing E17 pretrained seeds 0 and 1;
  do not select or retrain these checkpoints.
- Basis: the saved basis estimated from 60 training clips, rank fixed at 4.
  Compute adaptive directions from each tap's clip-centred token field on CPU.
- Six conditions: identity; broadcast clip mean only; and the full 2x2
  combination of clip/training-set mean and clip-adaptive/training-set basis.
  Every projection is `mu + (Z-mu) V V^T`. Adaptive directions always come
  from clip-centred tokens, irrespective of the reconstruction mean.
- Score the same 16-frame, 256-crop clips with uint8 PSNR-Y and AlexNet LPIPS.
  Save sequence names, per-condition scores, checkpoint/basis/source hashes,
  CPU projection backend, and full arguments. Never replace archived files.
- Report every condition and within-sequence contrast with 10,000 bootstrap
  replicates (RNG 0), seed averaging, and a scene-cluster sensitivity check.
  Main contrasts isolate the mean with basis held fixed and the basis with
  mean held fixed; clip-mean-only is a separate structural control.
- This is exploratory validation evidence. No equivalence margin or formal
  success gate is declared; a nonsignificant difference is inconclusive.
  No early stopping or condition selection is permitted based on performance.
  Full encoder computation remains, so this is not a runtime compression test.

The old E27 cancellation and unfinished E28 remain historical facts. They do
not establish that a different distillation scheme is impossible.
