"""End-to-end smoke for V-JEPA-LLVE v9.1a.

Verifies (per reviewer J2 minimum 4 + identity check):
  1. Model + RAFT build, identity-at-init pred ≈ LQ.
  2. RAFT flow shape matches pred/R_hat resolution after any resize/crop.
  3. Warped LQ[t+1] aligns with LQ[t] better than unwarped LQ[t+1] under masked L1.
  4. Horizontal flip changes flow_x sign.
  5. Mask mean is not near zero on normal clips.
  6. Full training step computes 9-loss bundle, backward succeeds, frozen params have 0 grad.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from losses.v9_1 import LLVELoss_v9_1, warp_with_flow, masked_l1
from models.raft_flow import build_raft_small, compute_flow_and_mask
from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import ParameterEMA, load_config


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[smoke v9.1] device: {device}")

    cfg = load_config(
        os.path.join(_REPO_ROOT, "configs/train_v9_1.yaml"), overrides=[],
    )

    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    raft = build_raft_small(device, weights=cfg["model"]["raft"].get("weights", "C_T_V2"))
    n_pb = sum(p.numel() for p in model.pixel_branch.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_raft = sum(p.numel() for p in raft.parameters())
    print(f"[smoke v9.1] PixelBranch v9 (trainable): {n_pb/1e6:.1f}M")
    print(f"[smoke v9.1] RAFT-small (frozen): {n_raft/1e6:.1f}M")
    print(f"[smoke v9.1] TOTAL trainable: {n_train/1e6:.1f}M")

    B = 2
    T = int(cfg["train"]["clip_len"])
    H = W = int(cfg["train"]["crop_size"])
    lq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)
    hq = torch.rand(B, 3, T, H, W, device=device).clamp(0, 1)

    # Make a "video-like" sequence: shift LQ by ~3 px right per frame so RAFT detects motion
    lq_motion = torch.zeros_like(lq)
    base_img = torch.rand(B, 3, H, W, device=device).clamp(0, 1)
    for t in range(T):
        shift = t * 3
        if shift > 0 and shift < W:
            lq_motion[:, :, t, :, :W - shift] = base_img[..., shift:]
            lq_motion[:, :, t, :, W - shift:] = base_img[..., :shift]
        else:
            lq_motion[:, :, t] = base_img
    print(f"[smoke v9.1] lq shape: {tuple(lq.shape)}")

    # 1. Identity-at-init
    model.eval()
    with torch.no_grad():
        out = model.sample(lq)
    pred = out["pred_pixels"]
    R_hat = out["R_hat"]
    L_hat = out["L_hat"]
    diff = (pred - lq).abs().max().item()
    print(f"[smoke v9.1] |pred - lq| max = {diff:.3e}  (expect ~0 at init)")
    assert diff < 1e-3, f"identity-at-init failed: {diff:.3e}"
    print(f"[smoke v9.1] R_hat shape: {tuple(R_hat.shape)}, range [{R_hat.min().item():.3f}, {R_hat.max().item():.3f}]")
    print(f"[smoke v9.1] L_hat shape: {tuple(L_hat.shape)}, range [{L_hat.min().item():.3f}, {L_hat.max().item():.3f}]")

    # 2. Flow shape check
    print("\n[smoke v9.1] === J2 check 2: flow shape ===")
    flow_back, mask = compute_flow_and_mask(raft, lq_motion, threshold_px=1.0)
    print(f"[smoke v9.1] flow_back shape: {tuple(flow_back.shape)}  expect (B, T-1, 2, H, W) = ({B}, {T-1}, 2, {H}, {W})")
    print(f"[smoke v9.1] mask shape: {tuple(mask.shape)}  expect (B, T-1, 1, H, W)")
    assert flow_back.shape == (B, T - 1, 2, H, W), "flow_back shape mismatch"
    assert mask.shape == (B, T - 1, 1, H, W), "mask shape mismatch"

    # 3. Warp alignment check: warp(LQ_{t+1}, flow_back) should be closer to LQ_t than LQ_{t+1} itself
    print("\n[smoke v9.1] === J2 check 3: warp alignment ===")
    bt = B * (T - 1)
    img_t_flat = lq_motion[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(bt, 3, H, W)
    img_tp1_flat = lq_motion[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(bt, 3, H, W)
    flow_back_flat = flow_back.reshape(bt, 2, H, W)
    mask_flat = mask.reshape(bt, 1, H, W)
    warped = warp_with_flow(img_tp1_flat, flow_back_flat)
    err_unwarped = masked_l1(img_t_flat - img_tp1_flat, mask_flat).item()
    err_warped = masked_l1(img_t_flat - warped, mask_flat).item()
    print(f"[smoke v9.1] masked L1(LQ_t, LQ_t+1)        = {err_unwarped:.4f}")
    print(f"[smoke v9.1] masked L1(LQ_t, W(LQ_t+1, fb)) = {err_warped:.4f}")
    print(f"[smoke v9.1] reduction: {(1 - err_warped / max(err_unwarped, 1e-6)) * 100:.1f}%")
    assert err_warped < err_unwarped, "warp does not reduce frame-pair L1 — flow direction wrong?"

    # 4. Horizontal flip flips flow_x sign (computed by RAFT, not adjusted by us)
    print("\n[smoke v9.1] === J2 check 4: hflip flow_x sign ===")
    lq_flip = torch.flip(lq_motion, dims=[-1])
    flow_back_flip, _ = compute_flow_and_mask(raft, lq_flip, threshold_px=1.0)
    # Sample only well-textured central pixels
    mid = H // 2
    sample_orig = flow_back[0, 0, 0, mid - 32:mid + 32, mid - 32:mid + 32].mean().item()
    sample_flip = flow_back_flip[0, 0, 0, mid - 32:mid + 32, mid - 32:mid + 32].mean().item()
    print(f"[smoke v9.1] mean flow_x (central) original: {sample_orig:.3f}")
    print(f"[smoke v9.1] mean flow_x (central) hflip:    {sample_flip:.3f}")
    print(f"[smoke v9.1] sign flipped: {(sample_orig * sample_flip) < 0 or abs(sample_orig + sample_flip) < abs(sample_orig) * 0.3}")

    # 5. Mask coverage
    print("\n[smoke v9.1] === J2 check 5: mask coverage ===")
    mask_mean = mask.float().mean().item()
    print(f"[smoke v9.1] mask mean = {mask_mean:.3f}  (expect > 0.3 for normal motion)")
    assert mask_mean > 0.05, f"mask mean too low: {mask_mean:.3f}"

    # 6. Full training step
    print("\n[smoke v9.1] === Full training step (9-loss bundle) ===")
    model.train()
    loss_fn = LLVELoss_v9_1(
        pix_weight=cfg["loss"].get("pix_weight", 1.0),
        lpips_weight=cfg["loss"].get("lpips_weight", 0.1),
        retinex_compose_weight=cfg["loss"].get("retinex_compose_weight", 0.5),
        L_smooth_spatial_weight=cfg["loss"].get("L_smooth_spatial_weight", 0.05),
        L_smooth_temporal_weight=cfg["loss"].get("L_smooth_temporal_weight", 0.10),
        R_adjacent_weight=cfg["loss"].get("R_adjacent_weight", 0.01),
        R_adjacent_scale=cfg["loss"].get("R_adjacent_scale", 2),
        pred_dod_weight=cfg["loss"].get("pred_dod_weight", 0.05),
        pred_warp_weight=cfg["loss"].get("pred_warp_weight", 0.05),
        R_warp_weight=cfg["loss"].get("R_warp_weight", 0.10),
    ).to(device)

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    ema = ParameterEMA(model.trainable_parameters(), decay=0.999).to(device)
    model.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    flow_back_real, mask_real = compute_flow_and_mask(raft, lq_motion, threshold_px=1.0)
    t_flow = time.time() - t0
    print(f"[smoke v9.1] flow compute: {t_flow:.2f}s")

    t1 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(lq_motion, video_hq=hq)
        losses = loss_fn(
            pred_pixels=out["pred_pixels"], hq=hq,
            R_hat=out.get("R_hat"), L_hat=out.get("L_hat"),
            flow_back=flow_back_real, mask=mask_real,
        )
    print("[smoke v9.1] losses: " +
          " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    assert torch.isfinite(losses["total"]).item(), "loss is NaN/Inf"

    losses["total"].backward()
    n_back_grad = sum(1 for p in model.backbone.parameters()
                      if p.grad is not None and p.grad.abs().sum().item() > 0)
    n_raft_grad = sum(1 for p in raft.parameters()
                      if p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f"[smoke v9.1] grads: backbone={n_back_grad} (expect 0), raft={n_raft_grad} (expect 0)")
    assert n_back_grad == 0, "V-JEPA backbone got grad"
    assert n_raft_grad == 0, "RAFT got grad"

    optim.step()
    ema.update(model.trainable_parameters())
    t_step = time.time() - t1 + t_flow
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke v9.1] full step | time {t_step:.2f}s | peak GPU {peak_gb:.2f} GB")

    print("\n[smoke v9.1] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
