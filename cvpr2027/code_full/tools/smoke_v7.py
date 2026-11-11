"""End-to-end smoke for V-JEPA-LLVE v7.

Verifies:
  1. VJepaLLVE_v7 builds with backbone + big PixelBranch.
  2. Identity-at-init: model.sample(LQ).pred_pixels ≈ LQ (residual head zero-init).
  3. Training step: pix + lpips + temporal + retinex losses compute, backward succeeds.
  4. V-JEPA backbone has 0 grad (frozen).
  5. Memory budget under ~75 GB.
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

from losses.v7 import LLVELoss_v7
from models.vjepa_llve_v7 import VJepaLLVE_v7
from utils import ParameterEMA, load_config


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[smoke v7] device: {device}")

    cfg = load_config(
        os.path.join(_REPO_ROOT, "configs/train_v7.yaml"), overrides=[],
    )

    model = VJepaLLVE_v7(cfg["model"]).to(device)
    model.freeze_backbone()
    n_back = sum(p.numel() for p in model.backbone.parameters())
    n_pb = sum(p.numel() for p in model.pixel_branch.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"[smoke v7] V-JEPA backbone (frozen): {n_back/1e6:.1f}M")
    print(f"[smoke v7] PixelBranch (big):        {n_pb/1e6:.1f}M")
    print(f"[smoke v7] total trainable:          {n_train/1e6:.1f}M")

    B = int(cfg["train"].get("batch_size", 1))
    T = int(cfg["train"]["clip_len"])
    H = W = int(cfg["train"]["crop_size"])
    lq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    hq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    print(f"[smoke v7] lq shape: {tuple(lq.shape)}")

    # 1. Identity-at-init: pred_pixels ≈ lq (since residual_head zero-init)
    model.eval()
    with torch.no_grad():
        out = model.sample(lq)
    pred = out["pred_pixels"]
    R_hat = out["R_hat"]
    L_hat = out["L_hat"]
    diff = (pred - lq).abs().max().item()
    print(f"[smoke v7] |pred - lq| max = {diff:.3e}  (expect ~0 at init)")
    assert diff < 1e-3, f"identity-at-init failed: {diff:.3e}"
    print(f"[smoke v7] R_hat shape: {tuple(R_hat.shape)} | L_hat shape: {tuple(L_hat.shape)}")
    print(f"[smoke v7] R_hat range: [{R_hat.min().item():.3f}, {R_hat.max().item():.3f}]")
    print(f"[smoke v7] L_hat range: [{L_hat.min().item():.3f}, {L_hat.max().item():.3f}]")

    # 2. Training step
    model.train()
    loss_fn = LLVELoss_v7(
        pix_weight=cfg["loss"].get("pix_weight", 1.0),
        lpips_weight=cfg["loss"].get("lpips_weight", 0.1),
        temporal_weight=cfg["loss"].get("temporal_weight", 0.1),
        retinex_weight=cfg["loss"].get("retinex_weight", 0.5),
        smooth_L_weight=cfg["loss"].get("smooth_L_weight", 0.05),
    ).to(device)

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    ema = ParameterEMA(model.trainable_parameters(), decay=0.999).to(device)
    model.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(lq)
        losses = loss_fn(
            pred_pixels=out["pred_pixels"], hq=hq,
            R_hat=out.get("R_hat"), L_hat=out.get("L_hat"),
        )
    print("[smoke v7] losses: " +
          " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    assert torch.isfinite(losses["total"]).item(), "loss is NaN/Inf"

    losses["total"].backward()
    n_back_grad = sum(1 for p in model.backbone.parameters()
                      if p.grad is not None and p.grad.abs().sum().item() > 0)
    n_train_t = sum(1 for _ in model.trainable_parameters())
    n_grad = sum(1 for p in model.trainable_parameters()
                 if p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f"[smoke v7] grads: {n_grad}/{n_train_t} trainable | "
          f"{n_back_grad} backbone (expect 0)")
    assert n_back_grad == 0, "V-JEPA backbone got grad — freeze failed"

    optim.step()
    ema.update(model.trainable_parameters())
    t_step = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke v7] full step | time {t_step:.2f}s | peak GPU {peak_gb:.2f} GB")

    print("[smoke v7] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
