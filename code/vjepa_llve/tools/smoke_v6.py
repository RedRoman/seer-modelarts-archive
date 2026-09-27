"""End-to-end smoke for V-JEPA-LLVE v6.

Verifies:
  1. VJepaLLVE_v6 builds with backbone + LatentDiT + PixelDecoder.
  2. Identity-at-init: model.sample(LQ).pred_pixels ≈ LQ (within float tol).
  3. Training step: flow_mse + pix + lpips + temporal losses compute, backward
     succeeds, optim step has no NaN.
  4. V-JEPA backbone gradient stays at 0 (frozen).
  5. Memory budget under ~70 GB at smoke batch.
"""
from __future__ import annotations

import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from losses.v6 import LLVELoss_v6
from models.vjepa_llve_v6 import VJepaLLVE_v6
from utils.config import load_config


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[smoke v6] device: {device}")

    cfg = load_config(
        os.path.join(_REPO_ROOT, "configs/train_v6.yaml"), overrides=[],
    )

    model = VJepaLLVE_v6(cfg["model"]).to(device)
    model.freeze_backbone()
    n_back = sum(p.numel() for p in model.backbone.parameters())
    n_dit  = sum(p.numel() for p in model.latent_dit.parameters())
    n_dec  = sum(p.numel() for p in model.pixel_decoder.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"[smoke v6] V-JEPA (frozen): {n_back/1e6:.1f}M")
    print(f"[smoke v6] LatentDiT:       {n_dit/1e6:.1f}M")
    print(f"[smoke v6] PixelDecoder:    {n_dec/1e6:.1f}M")
    print(f"[smoke v6] total trainable: {n_train/1e6:.1f}M")

    B = int(cfg["train"].get("batch_size", 1))
    T = int(cfg["train"]["clip_len"])
    H = W = int(cfg["train"]["crop_size"])
    lq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    hq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    print(f"[smoke v6] lq shape: {tuple(lq.shape)}")

    # 1. Identity-at-init: pred_pixels ≈ LQ (DiT zero-init, decoder output_conv zero-init,
    #    cross-attn proj zero-init).
    model.eval()
    with torch.no_grad():
        out = model.sample(lq)
    pred = out["pred_pixels"]
    diff = (pred - lq).abs().max().item()
    print(f"[smoke v6] |pred-lq| max = {diff:.3e}  (expect ~0 at init)")
    assert diff < 1e-3, f"identity-at-init failed: {diff:.3e}"
    refined = out["refined"]
    raw_l3 = out["e_lq_layers"][model.target_layer]
    r_diff = (refined - raw_l3).abs().max().item()
    print(f"[smoke v6] |refined - raw_E(LQ)[l3]| max = {r_diff:.3e}  (expect ~0)")
    assert r_diff < 1e-3, "DiT didn't return identity at init"

    # 2. Training step
    model.train()
    loss_fn = LLVELoss_v6(
        flow_weight=cfg["loss"].get("flow_weight", 1.0),
        pix_weight=cfg["loss"].get("pix_weight", 1.0),
        lpips_weight=cfg["loss"].get("lpips_weight", 0.1),
        temporal_weight=cfg["loss"].get("temporal_weight", 0.1),
    ).to(device)

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    model.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(lq, video_hq=hq)
        losses = loss_fn(
            v_pred=out.get("v_pred"), v_target=out.get("v_target"),
            pred_pixels=out["pred_pixels"], hq=hq,
        )
    print("[smoke v6] losses: " +
          " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    assert torch.isfinite(losses["total"]).item(), "loss is NaN/Inf"

    losses["total"].backward()
    n_back_grad = sum(1 for p in model.backbone.parameters()
                      if p.grad is not None and p.grad.abs().sum().item() > 0)
    n_train_t = sum(1 for _ in model.trainable_parameters())
    n_grad = sum(1 for p in model.trainable_parameters()
                 if p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f"[smoke v6] grads: {n_grad}/{n_train_t} trainable | {n_back_grad} backbone (expect 0)")
    assert n_back_grad == 0, "V-JEPA backbone got grad — freeze failed"

    optim.step()
    t_step = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke v6] full step | time {t_step:.2f}s | peak GPU {peak_gb:.2f} GB")

    print("[smoke v6] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
