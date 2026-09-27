# V-JEPA-LLVE v5

**Counterfactual Latent Flow Matching with V-JEPA 2.1 World Model for Low-Light Video Enhancement.**

---

## Core idea

We reframe LLVE as **counterfactual latent flow matching grounded in V-JEPA's pretrained world model**:

1. **V-JEPA 2.1 Encoder (frozen)** provides illumination-invariant structural features. Probe shows cos(φ(LQ), φ(HQ)) = 0.962 at layer 5.
2. **V-JEPA 2.1 Predictor (LoRA-adapted)** is repurposed for counterfactual reasoning: given dark-scene latent, predict the "imagined" well-lit-scene latent.
3. **Stable Diffusion VAE (frozen)** maps pixels to/from a 4-channel 32×32 latent grid (43 dB reconstruction PSNR on SDSD verified).
4. **DiT velocity field (~500-800M trainable)** learns the rectified flow LQ→HQ in VAE latent space, conditioned on V-JEPA features.
5. **Few-step (5-10) Euler ODE** sampling at inference.

```
LQ video (B,3,T,256,256)
    │
    ├──► VAE.encode (frozen) ──► z_LQ (B,4,T,32,32)
    │
    ├──► V-JEPA 2.1 Encoder (frozen) ──► E(LQ) at 4 hierarchical layers
    │              │
    │              ▼
    │     V-JEPA 2.1 Predictor + LoRA  (counterfactual refinement)
    │              │
    │              ▼
    │     refined_E_HQ_estimate    (cross-attn condition)
    │
    ▼
DiT Velocity Field (~500-800M trainable)
    v_pred = DiT(z_t, t, refined_features)
    Trained: ||v_pred - (z_HQ - z_LQ)||²
    │
    ▼ Rectified flow Euler ODE (5-10 steps)
    │
VAE.decode → HQ pixels
```

---

## Why this design

The previous iterations (v0 U-Net+FiLM, v3 V-DUN, v4.0 DiT regression, v4.1 V-DUN+Predictor) capped at PSNR 25.8 because **feed-forward regression on small datasets (155 SDSD train clips) has a regression-to-mean ceiling**. SOTA on SDSD (ReDDiT 29.95, MambaLLIE ~30) all use **diffusion**. We adopt that paradigm but ground it in V-JEPA's pretrained world model — a niche no prior work has explored.

See [DESIGN.md](DESIGN.md) for the full architectural rationale and the experiment history that led here.

---

## Quick start

```bash
cd /home/ma-user/workspace/zihaohe/vjepa_llve/

# 0. Activate environment
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip

# 1. Verify setup (imports, V-JEPA load, VAE load, dataset)
python tools/verify_setup.py

# 2. Probe V-JEPA prior on SDSD (GO/NO-GO before training)
bash scripts/probe.sh

# 3. Smoke test v5 model end-to-end
python tools/smoke_v5.py

# 4. Train
bash scripts/train.sh

# 5. Evaluate
bash scripts/eval.sh /cache/zihaohe/checkpoint/runs/v5/ckpts/best.pt
```

---

## Repository layout

```
vjepa_llve/
├── README.md, DESIGN.md, PROJECT_PROGRESS.md
├── requirements.txt
├── configs/
│   ├── base.yaml                    # V-JEPA 2.1 + bf16 defaults
│   ├── train_v5_diffusion.yaml      # main training config
│   ├── probe_sdsd.yaml
│   └── eval_sdsd.yaml
├── models/
│   ├── vjepa_backbone.py            # V-JEPA 2.1 ViT-L Local21 loader (frozen)
│   ├── vae_wrapper.py               # SD VAE-ft-mse wrapper (frozen)
│   ├── predictor_lora.py            # V-JEPA Predictor + LoRA + identity-gate
│   ├── dit_velocity.py              # DiT velocity field in VAE latent space
│   └── vjepa_diffusion.py           # end-to-end model
├── losses/core.py                   # flow MSE + counterfactual + LPIPS + temporal
├── data/                            # SDSD/LOL loaders, .npy support
├── utils/                           # config, metrics (PSNR/SSIM/LPIPS/MABD), logging
├── tools/                           # verify_setup, inspect_dataset, smoke_v5
├── train_diffusion.py               # main training loop (rectified flow)
├── eval.py                          # ODE-based evaluation
├── probe.py                         # V-JEPA invariance probe (GO/NO-GO)
└── scripts/                         # verify.sh, probe.sh, train.sh, eval.sh
```

---

## Storage layout (host 30138)

Big assets symlinked to `/cache/` (9.3 TB persistent):

```
/home/ma-user/workspace/zihaohe/
├── vjepa_llve/                      # this repo (small, ~15k lines)
├── datasets    -> /cache/zihaohe/datasets/      # SDSD .npy, LOLv2
├── checkpoint  -> /cache/zihaohe/checkpoint/    # V-JEPA 2.1 ckpt + run outputs
├── modelzoo    -> /cache/zihaohe/modelzoo/      # vjepa2-main clone
└── uploads     -> /cache/zihaohe/uploads/       # transient uploads

# Conda envs and HF caches all under /cache/ (see ~/.bashrc)
```

---

## License

Code: MIT. V-JEPA 2 weights: CC-BY-NC 4.0 (Meta). SDSD: research use (dvlab-research/SDSD).
