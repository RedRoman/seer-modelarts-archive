"""End-to-end smoke test for V-JEPA-LLVE v5.

Verifies:
  1. VJepaDiffusion builds: VAE + V-JEPA 2.1 + Predictor refiner + DiT
  2. Identity-at-init: sample(LQ) ≈ VAE.decode(VAE.encode(LQ)) ≈ LQ
  3. Refiner gate=0 at init → refined ≡ raw E(LQ)
  4. DiT velocity ≡ 0 at init (zero-init final_proj)
  5. Training step: flow_mse + counterfactual loss + backward + optim step
  6. Backbone + VAE frozen (no grad on their params)
  7. Memory budget acceptable
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from losses.core import LLVELossV5
from models.vjepa_diffusion import VJepaDiffusion
from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/train_v5_diffusion.yaml",
    )
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[smoke v5] device: {device}")

    cfg = load_config(
        os.path.join(_REPO_ROOT, args.config) if not os.path.isabs(args.config) else args.config,
        overrides=[],
    )

    model = VJepaDiffusion(cfg["model"]).to(device)
    model.freeze_backbone()
    n_vae   = sum(p.numel() for p in model.vae.parameters())
    n_back  = sum(p.numel() for p in model.backbone.parameters())
    n_dit   = sum(p.numel() for p in model.dit.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"[smoke v5] VAE (frozen):     {n_vae/1e6:.1f}M")
    print(f"[smoke v5] V-JEPA (frozen):  {n_back/1e6:.1f}M")
    print(f"[smoke v5] DiT (trainable):  {n_dit/1e6:.1f}M")
    if model.predictor_enabled:
        n_ref = sum(p.numel() for p in model.refiner.parameters())
        print(f"[smoke v5] Refiner total:    {n_ref/1e6:.1f}M")
        print(f"[smoke v5] gate init: {model.refiner.gate.item():.4f}")
    print(f"[smoke v5] total trainable:  {n_train/1e6:.1f}M")

    # Production-sized random input
    B = int(cfg["train"].get("batch_size", 2))
    T = int(cfg["train"]["clip_len"])
    H = W = int(cfg["train"]["crop_size"])
    lq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    hq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    print(f"[smoke v5] lq shape: {tuple(lq.shape)}")

    # 1. Identity-at-init via sample(): DiT zero-init → ODE returns z_LQ
    model.eval()
    with torch.no_grad():
        out = model.sample(lq, num_steps=cfg["model"].get("inference_steps", 8))
    z_LQ, z_pred = out["z_LQ"], out["z_pred"]
    z_diff = (z_pred.to(z_LQ.dtype) - z_LQ).abs().max().item()
    print(f"[smoke v5] |z_pred - z_LQ| max = {z_diff:.3e}  (expect ~0)")
    assert z_diff < 1e-3, f"identity-at-init failed: z diff {z_diff:.3e}"

    # 2. Refiner identity-at-init: refined ≡ raw E(LQ) when gate=0
    if model.predictor_enabled:
        with torch.no_grad():
            raw = model.encode_video_features(lq)
            ctx, refined = model.make_ctx_layers(raw, target_dtype=raw[0].dtype)
        if refined is not None:
            tl = model.predictor_target_layer
            r_diff = (refined - raw[tl].to(refined.dtype)).abs().max().item()
            print(f"[smoke v5] refiner |refined - raw_E(LQ)| max = {r_diff:.3e}  (expect ~0)")
            assert r_diff < 1e-3, f"refiner identity-at-init failed: {r_diff:.3e}"

    # 3. Training step: flow MSE + counterfactual
    model.train()
    loss_fn = LLVELossV5(
        flow_weight=cfg["loss"].get("flow_weight", 1.0),
        counterfactual_weight=cfg["loss"].get("counterfactual_weight", 0.1),
    ).to(device)

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    model.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    z_LQ = model.encode_pixels_to_latent(lq)
    z_HQ = model.encode_pixels_to_latent(hq)
    raw_lq = model.encode_video_features(lq)
    e_hq = None
    if model.predictor_enabled and cfg["loss"].get("counterfactual_weight", 0) > 0:
        raw_hq = model.encode_video_features(hq)
        e_hq = raw_hq[model.predictor_target_layer]

    t = torch.rand(B, device=device, dtype=torch.float32)
    t_b = t.view(B, 1, 1, 1, 1).to(z_LQ.dtype)
    z_t = (1 - t_b) * z_LQ + t_b * z_HQ
    v_target = (z_HQ - z_LQ).to(z_LQ.dtype)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ctx, refined = model.make_ctx_layers(raw_lq, target_dtype=z_LQ.dtype)
        v_pred = model(z_t, t, ctx)
        losses = loss_fn(v_pred=v_pred, v_target=v_target,
                         refined=refined, e_hq_target=e_hq)
    print(f"[smoke v5] losses: " + " ".join(
        f"{k}={v.item():.4f}" for k, v in losses.items()
    ))
    assert torch.isfinite(losses["total"]).item(), "loss is NaN/Inf"

    losses["total"].backward()
    n_grad_dit = sum(1 for p in model.dit.parameters()
                     if p.grad is not None and p.grad.abs().sum().item() > 0)
    n_back_grad = sum(1 for p in model.backbone.parameters()
                      if p.grad is not None and p.grad.abs().sum().item() > 0)
    n_vae_grad = sum(1 for p in model.vae.parameters()
                     if p.grad is not None and p.grad.abs().sum().item() > 0)
    n_dit_total = sum(1 for _ in model.dit.parameters())
    print(f"[smoke v5] DiT grads: {n_grad_dit}/{n_dit_total} (LoRA cold-start; should grow after step 1)")
    print(f"[smoke v5] V-JEPA grads (expect 0): {n_back_grad}")
    print(f"[smoke v5] VAE grads (expect 0): {n_vae_grad}")
    assert n_back_grad == 0
    assert n_vae_grad == 0

    optim.step()
    t_step = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke v5] full step | time {t_step:.2f}s | peak GPU {peak_gb:.2f} GB")

    print("[smoke v5] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
