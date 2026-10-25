# Seer → CVPR 2027: status (2026-10-25)

Deadline 2026-11-16 AoE. All numbers below are produced by one evaluator from
the frozen splits in `results/splits_cvpr.json` and `results/splits_lol.json`;
`notes/CVPR_PREREG.md` carries eleven amendments, each written before the
corresponding result existed.

## 1. What is measured and holds

### The prior helps in-domain, and the benefit is attributable

SDSD test, 39 sequences, crop 256 (the training crop), seed-averaged:

| method | PSNR-Y | LPIPS | params |
|---|---|---|---|
| **Seer-v2 (V-JEPA)** | **23.481** | 0.2437 | 361M |
| Retinexformer, retrained on our split (160k it) | 22.621 | **0.2115** | 1.6M |
| Seer-v2, random tokens | 20.729 | 0.2830 | 361M |
| Seer-v2, no tokens | 20.231 | 0.2852 | 361M |
| DarkIR | 20.154 | 0.2911 | 13M |
| HVI-CIDNet (generalization) | 17.728 | 0.3574 | 8M |

Random tokens score essentially the same as no tokens (20.729 vs 20.231), so the
+3.25 dB comes from the tokens' content and not from having an extra
conditioning path.

The margin over the training-matched baseline is thinner than the mean suggests:
**median -0.27 dB, 18/39 sequences won**. The advantage is concentrated where the
input is already bright (correlation between input PSNR and our margin +0.684):
on the brighter half we are +5.14 dB and win 14/20; on the darker half -1.41 dB
and win 4/19. Retinexformer fails catastrophically on bright inputs — on
`MVI_0921_2` it turns a 34.3 dB input into 11.0 dB output — and that failure mode
drives our mean.

### What the prior actually uses

Interventions that hold frame content pixel-identical and vary only its
arrangement (`code/temporal_ablate.py`), SDSD test:

| condition | prior effect |
|---|---|
| intact | +3.781 dB |
| shuffled (same frames, random order) | +3.785 dB |
| frozen (centre frame x16) | +1.563 dB |

Shuffling costs nothing (gap +0.004, CI [-0.418, +0.480]); freezing costs
2.218 dB and is prior-specific (CI [-3.780, -0.837]). **The prior does not use
temporal order.** Two further ablations agree that it needs neither spatial
correspondence (E11, -0.68 dB when positional grounding is added) nor the pixel
branch's capacity (E3, +2.02 dB at 22.4M against +3.31 dB at 361M).

What it does need is feature depth: reading only block 23 instead of
{5,11,17,23} costs 4.5 dB, more than the entire prior benefit.

### Exposure is a per-scene quantity

Variance decomposition over 271 sequences in three domains: between-domain
sd 0.200, within-domain sd 0.469, **ICC 0.154**. A per-domain offset estimated
from k clips has standard error 0.469/sqrt(k), which exceeds the offset being
corrected for every k <= 5 — confirmed by sweeping k = 0..30 (E7). GT-mean
normalisation is worth 4-7 dB cross-domain to *every* method tested, so this is
a property of the task rather than of any model.

## 2. What was tried and did not work

Four negative results, each with the instrument that produced it:

| attempt | result |
|---|---|
| LPM: world-model predictor as a temporal prior | fails a pre-registered specificity probe — spatial blur moves the measure 1.8x more than flicker |
| E11: spatially grounded cross-attention | -0.68 dB at +0.2% parameters |
| E15: cross-frame KV aggregation | +0.11 dB, n.s.; removes the frozen-condition deficit without a net gain |
| E16: exposure decoupling switched on | **-2.56 dB cross-domain** [-3.29, -1.81]; the residual exposure error *grows* (headroom 4.54 -> 6.5) |

H5 (cross-domain prior) fails: significant on 1 of 3 held-out domains (LOL
+1.59 dB, SDSD and BVI n.s. across 3 seeds), so the claim leaves the title.

## 3. The finding that changes how the rest is read

Every architecture comparison above was made at 2000 iterations, and that
protocol cannot resolve the effects being judged:

```
one configuration, three seeds, step 2000:  23.82 / 23.44 / 22.91   (spread 0.91 dB)
effects being judged:                       +0.11 (E15), -0.68 (E11)
```

The validation curves also cross: E15 hybrid leads the baseline by 1.0 dB at
steps 500 and 1000 and is overtaken by step 2000; several runs *decline* between
1500 and 2000. None has reached a plateau. The protocol additionally carries a
systematic bias against architectural additions, since anything with new
parameters to learn looks worse inside an undertrained window — which is exactly
the shape of all three failures.

For scale: Retinexformer reaches 22.6 dB with 1.6M parameters and 160k
iterations; our 361M branch without the prior reaches 20.2 dB in 2000. A 225x
parameter ratio losing by 2.4 dB is a training-budget result, not an
architectural one.

**No architectural conclusion in section 2 should be treated as established.**

## 4. Running now

E17: convergence runs, 20000 iterations on the 22.4M branch (6.3 s/it on 910B,
four times faster than the 361M branch, the only configuration that can reach
convergence before the deadline). Arms: pretrained x2, none x2, random x1, with
validation every 1000 steps. Due 2026-10-27.

It answers three things the current data cannot: the prior's benefit at
convergence, where we stand against Retinexformer at a comparable training
budget, and whether the seed spread falls far enough to make architecture
comparison possible at all. The third decides whether any method contribution
is reachable in the remaining time.

## 5. Open

- Paper skeleton moved to CVPR format and compiling; title changed to
  "What Transfers from a Video World Model? A Spatial--Semantic Prior for
  Low-Light Video Enhancement" per the pre-registered decision point.
- `clean-6` still undecided; the contamination it was designed to avoid is
  already handled by retraining Retinexformer on our split, so dropping it with
  a stated reason is the current plan.
- LPIPS trails Retinexformer in-domain (0.2437 vs 0.2115) and the two metrics
  disagree only in aggregate: per sequence they correlate at -0.78, so the
  sequences we lose on lose on both.
