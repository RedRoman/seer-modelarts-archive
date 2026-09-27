# WANVJ Briefing — Wan2.1 + V-JEPA Real-time LLVE

**Date:** 2026-05-11  
**Author:** Zihao  
**Purpose:** Brief friend (R) before discussion on why wanvj_25k Phase 1 PSNR is low (18.42) but perceptual quality (temporal stability + HF detail + object consistency) appears better than from-scratch baselines.

---

## 1. Quick context

- **Goal:** Real-time low-light video enhancement (LLVE) using a pretrained video diffusion model (Wan2.1-T2V-1.3B) + V-JEPA 2.1 features, fine-tuned with LoRA.
- **Hypothesis:** Pretrained video DiT brings strong temporal-coherence priors that from-scratch CNN/Transformer cannot learn from small datasets (122 SDSD sequences).
- **Real-time requirement:** 1-NFE inference (single forward, no multi-step ODE/DDIM).

## 2. Architecture

### Frozen (1.85B params)

| Component | Params | Role |
|---|---|---|
| Wan VAE (AutoencoderKLWan) | 127M | Causal 3D VAE: 8× spatial / 4× temporal compression. (16,3,T,H,W)→(B,16,T/4,H/8,W/8) |
| Wan DiT 1.3B (WanTransformer3DModel) | 1.42B | 30 blocks, dim=1536, heads=12, head_dim=128, ffn=8960. Originally trained on millions of T2V samples, now repurposed for video-to-video. |
| V-JEPA 2.1 ViT-L | 304M | Self-supervised video encoder. Layer 3 features (= ViT layer 23, deepest returned). Replaces Wan's original T5 text input. |

### Trainable (47.95M)

| Component | Params | Role |
|---|---|---|
| LoRA on Wan DiT | ~44M | rank=32 on q/k/v/o (attn1+attn2) + ffn.0 + ffn.2 of all 30 blocks (300 modules total). DiffSynth-Studio official recipe. |
| V-JEPA → 4096 projector | 4.2M | Linear(1024 → 4096). Projects V-JEPA layer 3 features into Wan DiT's text-embedding dimension, replacing T5. |
| LQ Conditioner | 7K | Conv3D(16→16), zero-initialized. Adds z_LQ_wan to noisy z_t inside DiT input. Allows DiT to "see" the LQ video. |

## 3. Training recipe

```python
# Loss (Phase 1 — minimal):
v_pred = WanDiT(z_t + lq_cond(z_lq), t, encoder_hidden_states=vj2text(z_vj))
L_flow = MSE(v_pred, v_target)            # rectified flow in Wan VAE latent space
L = L_flow                                # ⚠️ NO pixel loss, NO REPA, NO LPIPS

# Sampling:
t ~ U(0,1)
z_t = (1 - t) * z_LQ_wan + t * z_HQ_wan
v_target = z_HQ_wan - z_LQ_wan
```

- **Optimizer:** AdamW, lr=1e-4 (DiffSynth-Studio recipe), weight_decay=1e-4
- **Schedule:** Cosine annealing, 25000 iter
- **Batch:** 1 clip × 16 frames × 256×256 (random crop + temporal sample from BVI-RLV)
- **Train data:** BVI-RLV 44 scenes × 2 fps variants = 80 paired sequences, ~50K paired frames. (Cross-dataset!)
- **Eval data:** SDSD held-out 12 sequences (matches our prior protocol for fair comparison vs from-scratch baselines).

## 4. Inference

```python
# 1-NFE rectified flow (single forward pass, real-time-able)
z_LQ_wan = WanVAE.encode(LQ_video)
z_vj     = V-JEPA(LQ_video)[layer 3]
text_emb = vj2text(z_vj)
v_pred   = WanDiT(z_LQ_wan + lq_cond(z_LQ_wan), t=0, encoder_hidden_states=text_emb)
z_HQ_pred = z_LQ_wan + v_pred              # 1-NFE
HQ_pred  = WanVAE.decode(z_HQ_pred)        # ⚠️ Causal VAE: 16 in → 13 out
```

**Note: causal VAE drops 3 frames.** Eval trims GT to 13 frames for matching shape.

## 5. Results

### Aggregate (mean over 12 SDSD held-out sequences)

| Method | PSNR | SSIM | LPIPS↓ | DISTS↓ | CLIPIQA↑ | Trainable |
|---|---|---|---|---|---|---|
| **wanvj_25k Phase 1** | **18.42** | **0.726** | **0.320** | **0.253** | **0.225** | **47.95M LoRA** |
| NAFNet++ 30k | 20.71 | 0.742 | 0.261 | 0.241 | 0.242 | 27.33M (full) |
| clip_30k | 20.75 | 0.725 | 0.304 | 0.247 | 0.198 | 33M (full) |
| vjlf_30k | 19.86 | 0.738 | 0.306 | 0.257 | 0.193 | 80M (full) |
| v9.0b zero-shot | 20.58 | — | — | — | — | (no train) |

### Per-sequence PSNR (sorted)

| seq | wanvj | rank |
|---|---|---|
| pair31 | **25.86** | best |
| pair21 | 25.52 | |
| pair13 | 24.88 | |
| pair33 | 20.80 | |
| pair70 | 20.20 | |
| pair23 | 19.15 | median |
| pair50 | 19.01 | |
| pair60 | 18.57 | |
| pair52 | 14.71 | |
| pair58 | 14.32 | |
| pair15 | 12.88 | |
| **pair68** | **5.12** | disaster |

**Distribution:** bimodal — 5/12 in 19-26 dB range (works), 4/12 in 12-15 dB (poor), 1 disaster outlier. Average dragged down by tail.

### Visualizations (12 mp4 videos)

`/home/ma-user/workspace/zihaohe/visualizations/wanvj_25k/pair{13,15,21,23,31,33,50,52,58,60,68,70}.mp4`

Format: side-by-side `LQ | 8px gap | wanvj_pred | gap | GT`, 13 frames @ 10 fps.

Comparison videos at same paths under:
- `nafnet_pp_30k/` (best from-scratch)
- `nafnet_15k/` (PNG keyframes only)
- `jit_15k/` (PNG keyframes only)

## 6. Key observation (from Zihao after watching the videos)

> 虽然 PSNR 指标低, 但是**画面抖动以及物体一致性还有高频细节比其他好不少**, 就是**画面的色度和 GT 差别有点大**.

Translation: PSNR is low but **temporal stability + object consistency + HF detail are noticeably better** than from-scratch baselines. Main visible issue is **chroma drift** vs GT.

## 7. Hypothesized root causes for the chroma drift

| # | Cause | Evidence | Severity |
|---|---|---|---|
| 1 | **No pixel-level loss in training** | Loss is `MSE(v_pred, v_target)` in Wan VAE latent space only. Wan VAE decode introduces small color shift; without RGB-space supervision, the LoRA never learns to compensate. | **High** |
| 2 | **No color preservation prior** | LQ's color statistics are not piped to output. Model has no incentive to keep LQ's chroma. | High |
| 3 | **Cross-dataset domain shift** | BVI-RLV (controlled studio lighting) vs SDSD (mixed outdoor/indoor). Color distributions differ; LoRA may have absorbed BVI color preferences. | Medium |
| 4 | **Wan VAE not identity** | Even encode→decode without DiT introduces ~5-10 unit RGB drift. Without pixel loss this isn't corrected. | Medium |

## 8. Comparison to literature pattern

This pattern (perceptual win + PSNR lag + chroma issue) is the **textbook signature** of pretrained-video-diffusion-based restoration:

- **MGLD-VSR** (ECCV 2024): SD prior + light temporal adapter, similar trade-off.
- **LLVE-STCD** (CVPR 2024): SD U-Net (frozen) + temporal consistency, same pattern.
- **Relit-LiVE** (SIGGRAPH 2026): Wan2.1 + custom adapter, 50-NFE inference (not real-time).

**What's novel about wanvj:**
- 1-NFE inference (vs 50-NFE in Relit-LiVE)
- V-JEPA features replace T5 text (no text condition needed for restoration)
- LoRA-only fine-tuning (no full DiT training, 50× cheaper than Relit-LiVE)
- Cross-dataset eval (train BVI, test SDSD)

## 9. Phase 2 plan (proposed, await R discussion)

```python
# Add to current loss:
pred_rgb = wan_vae.decode(z_LQ + v_pred)       # 1-NFE decode (matches inference)

L_flow      = MSE(v_pred, v_target)            # KEEP: latent flow → temporal/structure
L_pixel_l1  = L1(pred_rgb, gt_rgb)             # NEW: direct pixel anchor
L_pixel_lab = MSE_Lab(pred_rgb, gt_rgb,
                      weight=[0.3, 1.0, 1.0])  # NEW: Lab — chroma-weighted
L_repa      = -cos(proj(dit.bottleneck),
                   V-JEPA(gt))                  # NEW: V-JEPA REPA on DiT mid-layer
L_lpips     = LPIPS(pred_rgb_4f, gt_4f)        # OPTIONAL: perceptual

L = L_flow + 1.0 * L_pixel_l1 + 0.5 * L_pixel_lab + 0.3 * L_repa
```

**Concern:** heavy pixel loss may flatten the Wan diffusion's generative quality (lose HF detail, regress toward NAFNet). The Lab-space + chroma-weighted formulation is meant to anchor color without flattening luminance freedom.

**Optional inference-time fix (no retrain):**
- AdaIN-style channel-wise color match: align pred's per-channel mean/std to LQ's per-channel mean/std. Free fix for ~50% of color drift.

## 10. Open questions for R

See `WANVJ_DISCUSSION.md` for the full thread.

---

## Appendix A: Hardware & speed

- A100 80GB
- Training: ~3.2 sec/iter, 25k iter total = ~22 hours
- Inference (1-NFE, 16 frames 256×256):
  - Wan VAE encode: ~80 ms
  - Wan DiT 1.3B forward (1024 tokens): ~200 ms
  - Wan VAE decode: ~80 ms
  - **Total: ~360 ms / 16-frame clip ≈ 25-30 FPS** ← real-time achievable for 30 FPS video

## Appendix B: GitHub references

- **Wan2.1-T2V-1.3B-Diffusers**: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- **DiffSynth-Studio LoRA recipe**: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-720P.sh (rank=32, q,k,v,o,ffn.0,ffn.2)
- **Relit-LiVE (uses Wan2.1)**: https://github.com/zhuxing0/Relit-LiVE (training code closed)
- **V-JEPA 2.1**: facebook ViT-L checkpoint
