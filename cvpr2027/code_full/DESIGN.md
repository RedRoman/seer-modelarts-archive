# Design rationale

## Problem statement

Low-light video enhancement (LLVE) on SDSD-indoor. Target: PSNR ≥ 27 dB (competitive with TempRetinex 30.21, VLLVE++ ~30.5, ReDDiT 29.95). Constraint: 80 GB A100, 155 train sequences (small dataset).

## What didn't work (history)

| Iteration | Architecture | SDSD-indoor PSNR | Why we moved on |
|-----------|--------------|------------------|-----------------|
| v0 | U-Net + FiLM + V-JEPA 2 | ~22 (LOLv2 image pretrain) | Image pretrain wastes video architecture; never reached 27 |
| V-JUN | + iterative refinement (recurrent K loops) | 23.6 | Mis-implementation of "depth-unfolding"; recurrent same-resolution loops don't help |
| V-DUN v3 | Depth-unfolding (VLU-Net style), 8 DUN blocks × 4 scales, 69M | **25.8 (best feed-forward)** | Saturated capacity; small-dataset regression-to-mean limits PSNR |
| v4.0 DiT | Naive DiT on pixel patches + bilinear upsample, 150M | 24.4 (regression) | DiT for restoration needs VAE pipeline (literature: STCDiT, Vivid-VR); single-scale + bilinear loses detail |
| v4.1 V-DUN+Predictor | V-DUN + V-JEPA Predictor (LoRA, random masking, identity-gate) | 25.4 (oscillating, then dropped to 23.8) | Random per-step masking destabilized decoder input distribution |

## Critical literature signals (March-April 2026)

- **V-JEPA 2.1** (March 2026) — dense feature SSL, not used for restoration before. Probe confirms cos=0.962 illumination invariance at layer 5. Paper niche secured.
- **ReDDiT (CVPR 2025)** — diffusion with Retinex; SDSD-indoor 29.95. Diffusion paradigm essential for SOTA.
- **STCDiT (CVPR 2026)** — VAE + DiT for video restoration; proves DiT works in restoration when paired with VAE.
- **Vivid-VR (Aug 2025)** — DiT-based restoration distilled from T2V foundation models.
- **SD VAE-ft-mse** — verified 43 dB reconstruction on SDSD frames (both LQ and HQ); free pretrained VAE that works.

## v5 architecture choices

### 1. Latent diffusion in SD-VAE space, not pixel space

Pixel-space DiT (v4.0) saturated at 24 dB because the bilinear 8× upsample at the end discards high-frequency detail. SD VAE encodes pixels at 8× downsampling with **43 dB reconstruction** — perfect fidelity, far better than learnable upsample. Operate DiT in latent space, decode with frozen VAE.

### 2. Rectified flow LQ → HQ, not denoising diffusion

Standard DDPM trains v(z_t, t) where z_t is a Gaussian-noised latent. Inference requires 50-1000 steps. **Rectified flow** uses linear interpolation z_t = (1-t)·z_LQ + t·z_HQ, target velocity = z_HQ - z_LQ. Inference: solve ODE from z_LQ to z_HQ in 5-10 Euler steps. Faster and a more direct LQ→HQ formulation.

### 3. V-JEPA 2.1 encoder as conditioning, not generator

V-JEPA 2.1's strength is **frozen feature representation** (1B+ video pretraining). Our DiT cross-attends to V-JEPA features at 4 hierarchical layers — provides illumination-invariant scene structure that flow needs but can't easily learn from 155 train clips.

### 4. V-JEPA 2.1 Predictor + LoRA as counterfactual refiner

The Predictor body (12 transformer blocks at 384-dim) was pretrained on 1M+ video hours for **masked latent completion** — predicting "what's behind the mask" in latent space. We repurpose this with LoRA (rank=16, ~5M trainable) to predict **counterfactual HQ latent** from LQ context: "if this scene were normal-lit, what would V-JEPA see?"

**Key fix from v4.1**: drop the dual-pass random masking (caused PSNR oscillation). Use deterministic full-token forward through predictor body — treat it as a feature refinement transformer using V-JEPA's pretrained body weights. The predictor's mask-token machinery is bypassed; we use only its body + LoRA + a fresh output projection (384 → 1024).

Identity-at-init via gate parameter: refined = tokens + α · transformer_output, with α = 0 at init. Decoder cross-attn sees raw E(LQ) at step 0 (no distribution shift from any warm-started decoder).

### 5. Heavy DiT (500-800M)

Latent grid is small (16×16 tokens per frame at patch=2). Memory budget at B=4-8 fits a much larger model than pixel-space DiT (which needed grad-checkpoint just to run 150M). We can spend capacity on DiT depth/width without OOM. Target 50-70 GB peak GPU utilization.

## Loss bundle

```
L = 1.0 · MSE(v_pred, z_HQ - z_LQ)              # rectified flow target
  + 0.1 · L1(refined_features, E(HQ).detach())   # counterfactual supervision
```

Optional add-ons (study after baseline):
- Pixel-space LPIPS on decoded predictions (decode every K iterations, expensive)
- Temporal consistency on decoded predictions (DoD loss)

## Out of scope

- **Retinex decomposition**: vetoed by user (dilutes V-JEPA narrative).
- **GAN losses**: not pursuing perceptual SOTA (LPIPS is enough).
- **Cross-clip recurrent state**: deferred to v5.1 if needed.
- **EMA / TTA**: deferred (low-hanging fruit but want clean baseline first).

## Story for paper

> "We propose **V-JEPA-LLVE**, the first approach that grounds low-light video enhancement in V-JEPA's pretrained world model. Specifically, we frame LLVE as a **counterfactual latent flow matching** task: given the dark observation, V-JEPA's predictor (with lightweight LoRA adaptation) reasons about what the same scene would look like under normal illumination at the level of pretrained latents, and a Diffusion Transformer learns the rectified flow between the dark and well-lit latent in SD VAE space. This combines V-JEPA 2.1's illumination-invariant structural prior (verified by our probe: cos > 0.96 at shallow layers) with the proven diffusion paradigm of recent restoration SOTA. We achieve [TBD] dB PSNR on SDSD-indoor while being significantly faster than diffusion baselines (5-10 ODE steps vs. 50-1000)."

## What we drop from earlier project

- `models/decoder.py` (U-Net + FiLM, v0/v1 era)
- `models/dun_decoder.py` (V-DUN, v3 era)
- `models/latent_predictor.py` (V-JUN, custom predictor)
- `models/dit_decoder.py` (v4.0 naive pixel DiT)
- All K-progressive unfolding logic in train.py
- All LOLv2 image-pretrain configs (vetoed: image data harms video architecture)
- All "alpha_k learnable step size" code

What we keep:
- `models/vjepa_backbone.py` (Local21 loader)
- `data/` (cleaned: drop bvi_rlv if not used)
- `utils/` (config, metrics, logging)
- `probe.py` (V-JEPA invariance probe — paper evidence)

## What we add fresh

- `models/vae_wrapper.py` (SD VAE)
- `models/predictor_lora.py` (clean Predictor refiner without random masking)
- `models/dit_velocity.py` (DiT velocity field in latent space)
- `models/vjepa_diffusion.py` (end-to-end model)
- `train_diffusion.py` (rectified flow training)
- `losses/core.py` (cleaned: flow MSE + counterfactual; remove DSS/legacy)
- `eval.py` (ODE-based inference + PSNR/SSIM/LPIPS/MABD)
- `tools/smoke_v5.py`
