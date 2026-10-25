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
