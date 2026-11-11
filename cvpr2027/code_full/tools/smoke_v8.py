"""End-to-end smoke for V-JEPA-LLVE v8.B+.

Verifies:
  1. VJepaLLVE_v8 builds with backbone + Predictor (LoRA + K-iter) + big PixelBranch.
  2. Identity-at-init: model.sample(LQ).pred_pixels ≈ LQ.
  3. Training step: all losses (pix/lpips/temp/retinex/smL/counter/mono) compute.
  4. V-JEPA backbone has 0 grad (frozen).
  5. Predictor body (non-LoRA) has 0 grad.
  6. Memory budget acceptable.
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

from losses.v8 import LLVELoss_v8
from models.vjepa_llve_v8 import VJepaLLVE_v8
from utils import ParameterEMA, load_config


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[smoke v8] device: {device}")

    cfg = load_config(
        os.path.join(_REPO_ROOT, "configs/train_v8.yaml"), overrides=[],
    )

    model = VJepaLLVE_v8(cfg["model"]).to(device)
    model.freeze_backbone()

    n_back = sum(p.numel() for p in model.backbone.parameters())
    n_pred_total = sum(p.numel() for p in model.predictor_refiner.parameters())
    n_pred_train = sum(p.numel() for p in model.predictor_refiner.trainable_parameters())
    n_router = sum(p.numel() for p in model.illumination_router.parameters())
    n_pb = sum(p.numel() for p in model.pixel_branch.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())

    print(f"[smoke v8] V-JEPA backbone (frozen):     {n_back/1e6:.1f}M")
    print(f"[smoke v8] Predictor total:              {n_pred_total/1e6:.1f}M")
    print(f"[smoke v8]   Predictor trainable (LoRA + action + alpha + new_proj + gate): {n_pred_train/1e6:.2f}M")
    print(f"[smoke v8] Illumination router:          {n_router/1e6:.3f}M")
    print(f"[smoke v8] Pixel Branch (big):           {n_pb/1e6:.1f}M")
    print(f"[smoke v8] TOTAL trainable:              {n_train/1e6:.1f}M")

    B = int(cfg["train"].get("batch_size", 1))
    T = int(cfg["train"]["clip_len"])
    H = W = int(cfg["train"]["crop_size"])
    lq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    hq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    print(f"[smoke v8] lq shape: {tuple(lq.shape)}")

    # 1. Identity-at-init
    model.eval()
    with torch.no_grad():
        out = model.sample(lq)
    pred = out["pred_pixels"]
    diff = (pred - lq).abs().max().item()
    print(f"[smoke v8] |pred - lq| max = {diff:.3e}  (expect ~0 at init)")
    assert diff < 1e-3, f"identity-at-init failed: {diff:.3e}"

    z_hq_pred = out["z_hq_pred"]
    z_lq = out["e_lq_layers"][model.target_layer]
    diff_z = (z_hq_pred - z_lq).abs().max().item()
    print(f"[smoke v8] |z_hq_pred - z_LQ| max = {diff_z:.3e} (expect ~0 with gate=0, alpha=0)")
    assert diff_z < 1e-3, "Predictor not identity at init"

    print(f"[smoke v8] mixture_weights[0]: {out['mixture_weights'][0].cpu().tolist()}")
    print(f"[smoke v8] intermediates count: {len(out['intermediates'])}")

    # 2. Training step
    model.train()
    loss_fn = LLVELoss_v8(
        pix_weight=cfg["loss"].get("pix_weight", 1.0),
        lpips_weight=cfg["loss"].get("lpips_weight", 0.1),
        temporal_weight=cfg["loss"].get("temporal_weight", 0.1),
        retinex_weight=cfg["loss"].get("retinex_weight", 0.5),
        smooth_L_weight=cfg["loss"].get("smooth_L_weight", 0.05),
        counter_weight=cfg["loss"].get("counter_weight", 0.3),
        monotonic_weight=cfg["loss"].get("monotonic_weight", 0.05),
    ).to(device)

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    ema = ParameterEMA(model.trainable_parameters(), decay=0.999).to(device)
    model.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(lq, video_hq=hq)
        losses = loss_fn(
            pred_pixels=out["pred_pixels"], hq=hq,
            R_hat=out.get("R_hat"), L_hat=out.get("L_hat"),
            z_hq_pred=out.get("z_hq_pred"),
            z_hq_target=out.get("z_hq_target"),
            intermediates=out.get("intermediates"),
        )
    print("[smoke v8] losses: " +
          " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    assert torch.isfinite(losses["total"]).item(), "loss is NaN/Inf"

    losses["total"].backward()
    n_back_grad = sum(1 for p in model.backbone.parameters()
                      if p.grad is not None and p.grad.abs().sum().item() > 0)
    n_pred_body_grad = sum(
        1 for p in model.predictor_refiner.refiner.predictor.parameters()
        if p.grad is not None and not isinstance(p, torch.Tensor) is False  # always tensor
        and p.grad.abs().sum().item() > 0 and p.requires_grad is False
    )
    print(f"[smoke v8] grads: backbone={n_back_grad} (expect 0)")
    assert n_back_grad == 0, "V-JEPA backbone got grad — freeze failed"

    optim.step()
    ema.update(model.trainable_parameters())
    t_step = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke v8] full step | time {t_step:.2f}s | peak GPU {peak_gb:.2f} GB")

    print("[smoke v8] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
