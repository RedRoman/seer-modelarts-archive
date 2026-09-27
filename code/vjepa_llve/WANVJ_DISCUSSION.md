# WANVJ Discussion Thread — Wan2.1+V-JEPA Phase 1 → Phase 2

**Read first:** [`WANVJ_BRIEFING.md`](./WANVJ_BRIEFING.md)  
**Watch:** 5 representative videos from `/home/ma-user/workspace/zihaohe/visualizations/wanvj_25k/` (pair31 best, pair70 typical-good, pair60 chroma-drift, pair52 worst-typical, pair68 disaster). Compare against `/home/ma-user/workspace/zihaohe/visualizations/nafnet_pp_30k/` for the same seq IDs.

---

## EA. Zihao opens — current state and 4 open questions

**Date:** 2026-05-11

### EA1. State

Phase 1 done. Numbers and viz in `WANVJ_BRIEFING.md`.

Headline: **wanvj 18.42 dB ≈ -2.3 dB vs NAFNet++ 20.71**, but the videos look noticeably better in three ways (temporal stability, object consistency, HF detail). Main visible defect is chroma drift.

### EA2. Why I think this is promising despite the PSNR loss

The pattern matches what happens when you fine-tune a pretrained video diffusion model on a small downstream dataset:
- Generative prior → temporal coherence + HF synthesis (these are first-class objectives in Wan2.1's pretraining).
- Pixel-level accuracy → not optimized in our Phase 1 loss → drifts.

The fix should be loss-recipe, not architecture. Hence Phase 2 is "add pixel + REPA + LPIPS losses, keep everything else."

### EA3. Question 1 — chroma fix

Phase 2 proposal: `L_flow + 1.0·L_pixel_L1 + 0.5·L_pixel_Lab(chroma-weighted) + 0.3·L_REPA + optional·LPIPS`.

**Q to R:**
- (a) Will heavy pixel loss collapse the Wan-induced perceptual quality back to NAFNet-style smooth-but-flat output?
- (b) Is Lab chroma-weighted loss the right way to decouple "luminance freedom" from "chroma anchoring"? Or is there a cleaner signal (e.g., YCbCr, OKLab)?
- (c) Should we add an inference-time AdaIN color-match (free, no retrain) before resorting to retraining?

### EA4. Question 2 — domain shift between BVI (train) and SDSD (test)

Train data BVI-RLV is studio-controlled (uniform lighting, fixed cameras). Test SDSD is outdoor + indoor mixed. Color distribution and noise statistics differ.

**Q to R:**
- (a) Is the 2 dB PSNR gap mostly attributable to this train/test shift, or to the missing pixel loss? How to attribute?
- (b) Should we mix in SDSD train fold during Phase 2 training? (We have a 122-seq SDSD train split; it's tiny vs BVI's 44 scenes × 2 fps = 80 paired sequences ~ 50K frames.)
- (c) Or train BVI-only Phase 2 first, then add a brief SDSD-only finetune at the end?

### EA5. Question 3 — V-JEPA replacing T5 text

Wan2.1's cross-attention was originally tuned to attend to T5 text embeddings (4096-dim semantic tokens). We replaced this with V-JEPA layer 3 features (1024-dim → projected to 4096). The distribution is completely different from T5 outputs.

Smoke training shows L_flow converges fine, but I worry the cross-attention isn't really "using" V-JEPA — it might be just averaging it as a constant bias.

**Q to R:**
- (a) How would you sanity-check whether the cross-attn is actually using V-JEPA features? (E.g., shuffle V-JEPA across batch and see if PSNR drops?)
- (b) Better alternatives: CLIP image features (closer to text-style semantics), or a tiny prompt prefix ("low light video to enhance, restore color and details")?
- (c) Should the V-JEPA projector be deeper than a single Linear (e.g., MLP with non-linearity)?

### EA6. Question 4 — 1-NFE vs multi-step

1-NFE is required for real-time, but it's a brutal constraint for what is essentially a generative model. Wan was pretrained with multi-step flow matching.

**Q to R:**
- (a) Would 4-step Euler ODE inference give us a meaningful PSNR/quality boost? (Cost: 4× slower, ~1.4 sec/clip — not real-time but still acceptable as "near real-time".)
- (b) Is the right play: train Phase 2 with 4-step inference target, then distill to 1-NFE in Phase 3 (MeanFlow-style)?
- (c) Or train directly with MeanFlow loss (Kaiming et al. 2025) which is designed for 1-NFE?

### EA7. What I'd like from this discussion

1. Validate or invalidate my "perceptual win is real" claim. Watch the 5 videos and tell me whether wanvj is genuinely better on temporal/HF dimensions, or if I'm projecting.
2. Pick the 2-3 most promising Phase 2 changes from the questions above.
3. Sanity check the loss-recipe proposal (Lab chroma weight, lambda values).

---

## EB. R's response

**Date:** 2026-05-11  
**Responder:** Codex/R

Short version: I would not treat Phase 1 as a failure. The important signal is that a frozen Wan video prior + small LoRA is alive under 1-NFE and cross-dataset evaluation. But I also would not move straight to a large Phase 2 recipe with pixel + Lab + REPA + LPIPS all at once. First separate three confounders: color drift, V-JEPA usefulness, and 1-NFE under-integration.

### EB1. Visual assessment

I buy the hypothesis that the perceptual win can be real, but it should be framed carefully. The current evidence is qualitative: temporal stability, object consistency, and high-frequency detail look better, while PSNR is hurt by chroma/exposure mismatch. That pattern is plausible for a pretrained video diffusion prior.

However, before making the paper claim, we need one small quantitative table that supports the visual claim:

| Claim | Minimal metric |
|---|---|
| Less flicker | temporal warping error or frame-difference variance after flow alignment |
| Better object consistency | detector/tracker feature drift on the 12 SDSD clips |
| Better HF detail | gradient magnitude / Laplacian energy near GT edges, plus LPIPS/DISTS |
| Color drift is the main PSNR issue | PSNR-Y vs PSNR-RGB, ΔE/ab error, per-channel mean/std error |

If wanvj is much closer on PSNR-Y than PSNR-RGB, then Phase 1 is exactly what it looks like: structure/luminance prior is useful, chroma calibration is wrong. If PSNR-Y is also weak, then the problem is not just color.

### EB2. Phase 2 priorities

I would do Phase 2 in three tightly scoped stages, not one big recipe.

**Priority 1: no-retrain color calibration baseline.**

Run inference-time color matching first:

```python
pred_ycbcr = rgb_to_ycbcr(pred)
lq_ycbcr = rgb_to_ycbcr(lq_trim)

# keep predicted Y, match Cb/Cr mean/std to LQ or GT-train statistics
pred_ycbcr[:, 1:] = adain(pred_ycbcr[:, 1:], lq_ycbcr[:, 1:])
pred_calib = ycbcr_to_rgb(pred_ycbcr)
```

Do not match full RGB mean/std blindly as the first option, because it can undo legitimate exposure correction. Match chroma channels only first. Also test a conservative variant: blend `CbCr = 0.7 * pred + 0.3 * lq`.

This tells us whether Phase 1 is already strong but miscalibrated. It is cheap and should be done before retraining.

**Priority 2: add RGB supervision, but keep it weak and late.**

I would not start with `1.0 * L1 + 0.5 * Lab + 0.3 * REPA + LPIPS`. That risks making the first Phase 2 result hard to interpret. Use:

```python
L = L_flow
  + 0.25 * L1(pred_rgb, gt_rgb)
  + 0.25 * L_chroma(pred_rgb, gt_rgb)
```

Use YCbCr or OKLab. My preference here is YCbCr for the first experiment because it is easy to debug and directly answers the observed issue:

```python
L_chroma = L1(CbCr(pred), CbCr(gt))
L_luma   = L1(Y(pred), Y(gt))
L_rgb    = L1(pred, gt)

L = L_flow + 0.15 * L_rgb + 0.35 * L_chroma + 0.10 * L_luma
```

The weights should be ramped in after a warmup, not active from step 0:

```text
0-2k:      L_flow only
2k-10k:    ramp pixel/chroma to target weights
10k-end:   hold
```

Reason: preserve the Wan prior first, then calibrate the decoder output.

**Priority 3: attribution experiments before REPA/LPIPS.**

Before adding REPA or LPIPS, run these four ablations:

| Exp | Change | Purpose |
|---|---|---|
| A | Phase1 + chroma AdaIN only | tests if retraining is needed |
| B | Phase1 + weak YCbCr/RGB loss | tests if color supervision fixes PSNR |
| C | BVI-only vs BVI+SDSD mix | separates missing pixel loss from domain shift |
| D | V-JEPA shuffle/drop eval | checks if V-JEPA is causal |

Only add REPA/LPIPS after B shows the color anchor does not destroy the visual advantage.

### EB3. Methodology

**On chroma loss.**

Lab is reasonable, but I would start with YCbCr or OKLab. Lab conversion can be numerically annoying in torch and may introduce scale surprises. YCbCr has a cleaner interpretation for this specific failure: keep/generate luminance, anchor chroma. OKLab is a good second choice if you want perceptual uniformity.

The core idea should be:

```text
Wan prior is allowed to decide structure and temporal dynamics.
GT supervision anchors color and coarse exposure.
Do not let pixel loss dominate high-frequency synthesis.
```

So the first Phase 2 loss should be deliberately weak. If it helps PSNR-RGB by 1-2 dB without hurting LPIPS/DISTS/flicker, then increase weights.

**On domain shift.**

The current BVI-train/SDSD-test setup is useful because it is harsh, but it confounds the diagnosis. I would run:

```text
Phase2-BVI:       train BVI only, eval SDSD
Phase2-BVI+SDSD:  sample 50/50 BVI and SDSD-train, eval SDSD
Phase2-SDSD-ft:   start from BVI model, finetune 2k-5k iters on SDSD-train
```

If BVI-only + pixel/chroma already recovers most of the gap, the issue was loss. If SDSD-ft is required, the issue is domain calibration. If neither fixes pair68, inspect pair68 separately; it may be a VAE/temporal trimming/alignment or severe exposure outlier.

**On V-JEPA replacing T5.**

Do not change V-JEPA to CLIP or add text yet. First measure whether V-JEPA is used:

1. Eval with correct V-JEPA.
2. Eval with V-JEPA features shuffled across sequences.
3. Eval with V-JEPA features time-reversed.
4. Eval with zero V-JEPA features.
5. Eval with per-token mean feature repeated everywhere.

Report PSNR, LPIPS/DISTS, and flicker. If all results are similar, cross-attn is using V-JEPA as a bias or not using it. If shuffled hurts structure or temporal stability, V-JEPA is load-bearing.

I would also log cross-attn entropy and output norm changes per block. The useful signal is not just "attention has nonzero weights"; it is whether changing V-JEPA changes the DiT residual in a spatial/temporal-specific way.

For the projector, a single Linear is probably too weak distribution matching from V-JEPA tokens to T5-like Wan condition space. But do not change it until the shuffle/drop test says V-JEPA matters. If it matters, next projector should be:

```python
LayerNorm(1024)
Linear(1024, 4096)
GELU
Linear(4096, 4096)
learned scale initialized small
```

Initialize so Phase 1 behavior is not destroyed.

**On 1-NFE vs 4-step.**

Run 4-step Euler as a diagnostic, not as the Phase 2 target yet. If 4-step gives a large jump, then 1-NFE is the bottleneck and MeanFlow/distillation becomes central. If 4-step barely helps, the bottleneck is supervision/domain/color.

The decision rule I would use:

```text
4-step improves >= 0.8 dB or clearly improves LPIPS/flicker:
    pursue 4-step teacher -> 1-step distillation / MeanFlow
else:
    keep 1-NFE and fix loss/data
```

**My recommended minimum Phase 2.**

Do this in order:

1. `eval_color_calib`: Phase1 checkpoint + chroma-only YCbCr AdaIN/blend, no retrain.
2. `eval_vjepa_causal`: correct vs shuffled vs zero vs mean V-JEPA.
3. `train_phase2_chroma`: continue from wanvj_25k with weak RGB + YCbCr chroma loss, BVI only, 5k-10k iters.
4. `train_phase2_sdsd_ft`: short SDSD-train finetune only if BVI-only still has large chroma/domain gap.
5. Only after the above: add REPA/LPIPS or MeanFlow.

**Paper-worthiness judgment.**

This is paper-worthy only if we can show one of the following clean claims:

```text
1. Wan prior gives better temporal/object consistency than from-scratch LLVE under equal or lower NFE.
2. V-JEPA conditioning is causally useful as non-text conditioning for video restoration.
3. 1-NFE Wan-LoRA can be made competitive after simple color/domain calibration.
```

The strongest story is probably not "best PSNR." It is:

```text
Pretrained video diffusion priors can be compressed into a real-time 1-NFE LLVE adapter, but restoration needs explicit color/data-consistency anchoring.
```

That is a clean and defensible Phase 2 target.

---

## EC. Zihao response to R (EC1)

**Date:** 2026-05-11

### EC1.1 Where I fully agree with R

1. **Diagnostic-first, recipe-second.** My EA3 was indeed too eager — bundling pixel + Lab + REPA + LPIPS makes attribution impossible. R's three-stage breakdown (no-retrain calibration → weak ramped loss → ablations before REPA/LPIPS) is the right discipline.
2. **YCbCr > Lab for first chroma loss.** Conceded. Lab conversion in torch is fragile (sRGB linearization, D65 whitepoint), debugging surprises are not worth it for the first try. YCbCr ITU-R BT.601/709 is one matrix multiply.
3. **PSNR-Y vs PSNR-RGB is the right cheap diagnostic.** I should have computed this from the start on the existing wanvj_25k eval. Doing it now (Step A below).
4. **5-condition V-JEPA causality test.** Yes — without this we cannot defend the V-JEPA-replaces-T5 claim in any narrative.
5. **4-step Euler as diagnostic, not target.** The 0.8 dB rule is reasonable. Will run.
6. **Narrative target.** Accept R's framing: *"Pretrained video diffusion priors can be compressed into a real-time 1-NFE LLVE adapter, but restoration needs explicit color/data-consistency anchoring."* This is honest, defensible, and doesn't depend on PSNR SOTA.

### EC1.2 One pushback / clarification

**On Priority 2's warmup schedule.** R writes "0-2k: L_flow only, 2k-10k ramp, 10k-end hold." This implies retraining from scratch. Cheaper alternative: **continue from the wanvj_25k checkpoint** (already converged, L_flow=0.005), and ramp the new pixel/chroma losses in over the first 1k iter of the resumed run, then hold for 5-10k iter total. Same effect, 2.5× cheaper.

Concern with R's "from-scratch ramp": with 0-2k iter of L_flow only, the model is essentially re-learning what wanvj_25k already learned (L_flow drops fast in the first 2k as we saw). Wasted compute. Resuming + short ramp avoids this.

If R disagrees, the alternative I'd accept: train from scratch but **with a higher initial LR for the new losses** (e.g., 2e-4 ramped), so they catch up faster.

### EC1.3 Concern with "0.7 pred + 0.3 lq" chroma blend

For pair60 / pair52 / pair68 — sequences where the GT is in **a totally different color space than LQ** (severe low-light has near-monochrome chroma) — blending toward LQ will pull predictions back toward bad chroma. The blend ratio probably needs to be **per-seq adaptive** based on LQ chroma variance, OR use GT-train statistics (R's other suggestion) instead of LQ. I'll test both in Step C below.

### EC1.4 My execution plan (in order)

Per R's ordering, but front-loading the no-retrain diagnostics into one batched session before any retraining commits GPU.

| Step | Action | Cost | Deliverable |
|---|---|---|---|
| **A** | Re-eval wanvj_25k with extended metrics: PSNR-Y, PSNR-Cb, PSNR-Cr, ΔE76, per-channel mean/std error vs GT | ~10 min, no retrain | EC2 table — confirms or refutes chroma-only hypothesis |
| **B** | V-JEPA causality test: 5 conditions (correct, shuffle-batch, time-reverse, zero, mean), report PSNR/LPIPS/temporal-warping-error per condition | ~40 min, no retrain | EC3 table — V-JEPA load-bearing or not |
| **C** | Inference-time chroma calibration baselines: (a) YCbCr AdaIN match-to-LQ, (b) match-to-GT-train-stats, (c) 0.7/0.3 blend, (d) per-seq adaptive | ~20 min, no retrain | EC4 table — recovery without retrain |
| **D** | 4-step Euler ODE eval | ~10 min, no retrain | EC5 — single number, decides MeanFlow path |

After A-D, **then** decide: continue from ckpt with weak loss (Priority 2), or jump to MeanFlow distillation, or invest in V-JEPA projector upgrade.

Estimated total: ~1.5 hr of GPU. Currently nafnet_pp_clip_30k and supernafnet_pp_30k are using GPU but with 70+ GB free. wanvj eval is ~10-15 GB peak. Compatible.

### EC1.5 Open question back to R

The temporal-warping-error metric in EB1 — would you compute it as:
- (a) frame-difference variance after RAFT/optical-flow alignment, or
- (b) mean absolute Laplacian of (frame_t - flow-warped(frame_t-1))?

I have RAFT available. Want to make sure the metric matches what you mean before I commit.

---

## ED. R response to Zihao (ED1)

**Date:** 2026-05-11  
**Responder:** Codex/R

### ED1.1 Resume from `wanvj_25k`, do not retrain from scratch

Agree with the pushback. My warmup schedule was meant conceptually, not as a demand to restart from random LoRA. Given Phase 1 already converged and the goal is to isolate the effect of color/pixel anchoring, the cleaner first experiment is:

```text
start: wanvj_25k checkpoint
0-1k: ramp new RGB/YCbCr losses from 0 to target
1k-5k/10k: hold target weights
```

One caveat: the current `ckpt.pt` saves LoRA/adapters but not optimizer/scheduler state. That is fine, but I would restart AdamW conservatively:

```text
lr = 3e-5 or 5e-5, not 1e-4
weight_decay = 1e-4
new-loss ramp = 1000 iter
clip_grad_norm = 1.0
```

If you keep `lr=1e-4`, I would at least use a 500-1000 iter LR warmup. The failure mode is not catastrophic forgetting of Wan itself because base DiT is frozen; it is LoRA/adapters moving too fast and losing the Phase 1 temporal/structure behavior.

Recommended first resumed loss:

```python
L = L_flow
  + w_rgb(t)    * 0.15 * L1(pred_rgb, gt_rgb)
  + w_chroma(t) * 0.35 * L1(CbCr(pred_rgb), CbCr(gt_rgb))
  + w_y(t)      * 0.05 * L1(Y(pred_rgb), Y(gt_rgb))
```

Keep luma weak initially. The current visible complaint is chroma drift, not lack of luminance freedom.

### ED1.2 On chroma blend: agree, LQ chroma can be a bad anchor

Your concern is correct. In extreme low light, LQ chroma may be nearly collapsed or sensor/ISP-biased, so fixed `0.7 pred + 0.3 lq` can drag good predictions toward bad chroma. I would treat LQ chroma as a reliability-gated signal, not a universal target.

For no-retrain calibration, test these in this order:

| Variant | Formula | Expected use |
|---|---|---|
| C1 | chroma AdaIN to LQ | works when LQ chroma is reliable |
| C2 | chroma AdaIN to SDSD-train GT stats | works when LQ chroma is collapsed |
| C3 | gated LQ/GT-stat blend | best diagnostic candidate |
| C4 | oracle GT chroma stats | upper bound only, not deployable |

The adaptive gate can be simple:

```python
sat = mean(abs(Cb_lq - 0.5) + abs(Cr_lq - 0.5))
var = std(Cb_lq) + std(Cr_lq)
r_lq = clamp((sat + var - tau_low) / (tau_high - tau_low), 0, 1)

target_mu  = r_lq * mu_lq  + (1 - r_lq) * mu_train
target_std = r_lq * std_lq + (1 - r_lq) * std_train
```

Use this only for evaluation-time diagnosis. I would not put this exact heuristic into the final method unless it clearly survives across datasets.

### ED1.3 Temporal metric: use RAFT warp error first; Laplacian residual as secondary

Use (a) as the primary metric:

```text
E_warp = mean_t | pred_t - warp(pred_{t-1}, flow_{t->t-1}) |
```

But compute it with two safeguards:

1. Use RAFT flow estimated on GT or LQ, not on each model's prediction, otherwise a cleaner prediction may get an easier flow estimate.
2. Mask likely occlusions/disocclusions with forward-backward consistency if available; if not, report the unmasked metric and mark it as approximate.

I would report three variants:

```text
warp_L1_RGB       = mean absolute RGB residual after RAFT warp
warp_L1_Y         = same on luma only
warp_Lap_Y        = Laplacian of luma residual, secondary HF flicker metric
```

The Laplacian residual is useful, but not as the main metric. It overweights high-frequency texture differences and can punish genuinely sharper outputs. It is best used to support the claim "wanvj has less high-frequency shimmer" after the plain warp metric is already favorable.

For the Phase 1 claim, the table I want is:

| Method | PSNR-RGB | PSNR-Y | LPIPS | DISTS | warp_L1_Y | warp_Lap_Y |
|---|---:|---:|---:|---:|---:|---:|
| NAFNet++ | | | | | | |
| wanvj_25k | | | | | | |
| wanvj_25k + chroma calib | | | | | | |

If wanvj loses RGB PSNR but wins or ties `warp_L1_Y / warp_Lap_Y`, the perceptual-temporal story becomes much more credible.

### ED1.4 V-JEPA causality test details

For Step B, I would add one more condition:

```text
correct V-JEPA, but tokens spatially permuted within the same clip
```

So the full set is:

1. correct
2. batch-shuffled
3. time-reversed
4. zero
5. per-token global mean
6. spatially permuted within clip

Why: batch shuffle tests semantic mismatch; time reverse tests temporal order; spatial permutation tests whether cross-attn is using spatial layout or just global statistics. If spatial permutation barely changes output, the current V-JEPA pathway is probably acting as a global bias.

Also save a small difference visualization:

```text
abs(pred_correct - pred_shuffled)
abs(pred_correct - pred_zero)
```

If the difference map is spatially structured around objects/edges, good. If it is almost uniform color/exposure shift, V-JEPA is not providing the kind of conditioning we want.

### ED1.5 4-step Euler diagnostic

One caution for Step D: the current training samples `t ~ U(0,1)` but eval uses `t=0` one-shot. A naive 4-step Euler needs a convention for repeatedly evaluating the velocity at intermediate `t`. The simple diagnostic is:

```python
z = z_lq
for t in [0.0, 0.25, 0.5, 0.75]:
    v = model(z + lq_cond(z_lq), t, vjepa_cond)
    z = z + 0.25 * v
```

This is not guaranteed to improve because the model was trained to predict the straight velocity between `z_lq` and `z_hq`, not necessarily a curved denoising trajectory. If 4-step does not help, do not overinterpret it as "1-NFE is enough"; it may mean the current rectified-flow target is already a one-step straight-path target.

Decision rule adjusted:

```text
4-step helps a lot:
    multi-step teacher / distillation is worth exploring
4-step flat or worse:
    focus on color/data/V-JEPA first; no conclusion about MeanFlow yet
```

### ED1.6 Final recommended immediate plan

I agree with A-D exactly, with two small edits:

1. In Step B, add spatial permutation.
2. In Step C, include oracle GT-chroma-stat calibration as an upper bound.

Do not start REPA, LPIPS, CLIP substitution, text prefix, or projector MLP until A-D are in the file. The next real decision should be data-driven:

```text
If PSNR-Y is okay but PSNR-RGB bad:
    color calibration/loss is first.
If V-JEPA shuffle barely matters:
    fix conditioning path before more losses.
If chroma calibration recovers most of the gap:
    Phase 2 can be a cheap calibration + weak loss paper story.
If pair68 remains catastrophic in every setting:
    isolate pair68 as a failure/outlier case before optimizing around it.
```

So yes: proceed with A-D. I would not spend GPU on training until those four diagnostic tables exist.

---

## EE. Decisions log

| Date | Decision | Made by | Rationale |
|---|---|---|---|
| 2026-05-11 | Open this discussion thread | Zihao | Friend R needs structured briefing before Phase 2 |
| | | | |

---

## EF. Background — why we're here

Path so far on this project:

1. **From-scratch CNN/Transformer/DDPM** (NAFNet, vjlf, vjpf, MMDiT, JiT, MeanFlow, etc.) — all hit a ceiling around PSNR 20.5-20.7 on SDSD. Best is NAFNet++ 20.71.
2. **Tried multiple SSL teachers** (V-JEPA video, CLIP image, DINOv2 image) — all in the same band, no breakthrough.
3. **Realized we're missing the modern paradigm**: pretrained video diffusion + lightweight adapter. This is what 2024-2026 SOTA (MGLD-VSR, LLVE-STCD, Relit-LiVE) all use. Decided to try Wan2.1.
4. **Phase 1 wanvj** (this discussion) — minimal "is it alive?" sanity check. PSNR low but viz suggests genuine perceptual improvement. Need to decide Phase 2.

The question this discussion is supposed to settle: **is wanvj a paper-worthy direction, and what's the minimum Phase 2 to confirm that?**
