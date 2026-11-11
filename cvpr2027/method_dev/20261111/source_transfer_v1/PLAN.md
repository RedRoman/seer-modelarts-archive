# Source-trained temporal correction: SDSD to BVI

This is method development, with manuscript work paused. Freeze this protocol
before source caching/training or target inference. It is an exploratory
cross-dataset experiment: BVI validation has already informed project research,
so this is not a claim of an entirely target-blind development history.

## Inputs and isolation

- Base: E17 SDSD-only pretrained seed0, small branch, final20k. Its checkpoint
  SHA is `5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740`.
  Archived run identity/logs specify SDSD136; the historical checkpoint does
  not contain a training split hash. Record the present split hash without
  presenting it as historical hash proof.
- Source: all 136 SDSD training sequences, one deterministic crop256/clip16
  per sequence, data seed20261113. Preserve indoor/outdoor prefixes and group
  terminal `_2` passes with their scene. Source frames pair by sorted index:
  65 `_2` sequences have different original LQ/GT numbering. Require equal
  counts and record complete file-pair mapping, rather than force false
  basename equality.
- Source selector fit/internal holdout: 75/25 scene split with seed20261114.
  The internal holdout was available to historical base training. All feature
  normalization and scalar loss scaling use selector fit scenes only.
- Target: frozen BVI VAL16, eight scenes/two illumination conditions,
  center16/crop256. Require identical complete LQ/GT basenames. GT is used
  only for final scoring, with no oracle, target fitting or checkpoint choice.
  Do not read TEST images in this experiment.

## Matched candidate controls

One frozen base pass provides restored frames; prediction-only RAFT-small
provides target-to-source flow for t−2/t+2. Cache low/base/source-GT/flow/masks
once. GT is a source loss label and never an input to base, flow or candidates.
Use the same CPU float32 candidate implementation in source preparation and
target evaluation. Record reconstruction differences from cached NPU warps.

1. **restored:** existing full-RGB restored donors.
2. **restored_detail:** target base low frequency plus aligned restored donor
   high frequency; controls for the frequency split alone.
3. **raw_detail:** use source low-light high frequency calibrated with the
   observable source lowpass(base)/lowpass(low) gain, then align and retain
   target base low frequency. This changes the value source without another
   base or RAFT call. It does not imply statistically independent evidence.

Fix reflect box lowpass9, gain clamp[.25,16], denominator epsilon1e-4 and
RGB clipping before outcomes. Both detail controls share lowpass-support
validity erosion. Frequency separation, guided detail transfer and burst
observation fusion are established principles; these controls do not alone
constitute a novel method. Test whether the new values improve actual learned
restoration before developing a contribution around them.

## Fixed learning and evaluation

All three families use the same 19-feature/6,626-parameter CNN, and each uses
both direct-MSE and fit-scalar-normalized direct-MSE. Selector seeds0/1/2 vary
initialization and a precomputed frame plan. Match both exactly across families
and losses for each seed. Each head receives Adam1e-3, batch8, 1,000 updates.
All zero heads start at equal thirds over base/two donors. No intermediate
holdout or target checkpoint ranking; evaluate all final heads.

Use one shared target base/RAFT calculation per clip for all 18 final heads,
plus fixed averages/equal-thirds controls. Report rounded uint8 PSNR-RGB,
PSNR-Y, LPIPS and the per-sequence absolute tLP-to-GT gap. Compare every learned
arm to the same source-trained base, and compare each detail family directly
to restored at matched loss/seed. First average effects across selector seeds,
then bootstrap complete target scenes. These are three selector fits over one
base seed, not three independently trained restorers.

Success means a learned improvement over the restored-donor direct control,
supported across seeds/scenes and checked against fidelity/perceptual/temporal
tradeoffs. A larger GT oracle or an isolated favorable target row is
insufficient. No SOTA conclusion follows from this pilot. An official
SDSD-indoor STCD checkpoint may provide a labeled zero-finetuning reference,
but has different source training data and is not a matched-training verdict.

## Follow-ups gated by evidence

If candidate gains survive, develop the new fusion/evidence mechanism and
repeat over independently trained bases and a video domain not used during
development. Add matched no-prior/DINO controls when attributing a benefit to
the video prior. Public SOTA comparison requires common source data, extra
pretraining disclosure, final selection rules and native-resolution metrics.

## Pre-target extension: no-prior restoration control

Added while source caching is running, before any selector training or target
result in this experiment: repeat the same source136 cache/three families/two
losses/three selector seeds over E17 `none_s0` final20k, SHA
`fa3c0c9e720a28b56f01069f7a2cd407bc5b4d752a2a1d4482f1bd63557b5516`.
Use the same source crop/augmentation plan and training budgets, with fresh
candidate values/flow from that base. This can run on a separate free host;
target scoring for both arms uses the same 31051 environment. All 36 final
heads are reported, without choosing an arm from target results.

The historical pretrained/no-prior restorers did not share exactly identical
pixel-branch initial states despite the same seed name. This is a comparison
of the two available trained bases and added corrections, not an isolated
causal test of semantic pretraining. Each base still has only one training
seed. A matched new retraining remains necessary for that stronger attribution.
