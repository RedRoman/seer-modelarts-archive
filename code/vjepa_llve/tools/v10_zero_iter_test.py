"""v10 zero-iteration equivalence test (CH3 HARD GATE).

Loads v9.0b_mix weights into both:
  - VJepaLLVE_v9 (reference)
  - VJepaLLVE_v10 (with VJepaPosEmb at identity-init)

Forwards identical input through both, compares outputs.

PASS criteria (V10_DESIGN_SPEC.md §7):
    mean abs(pred_v10 - pred_v9)  < 1e-5
    PSNR difference                < 0.01 dB

Output: PASS / FAIL message + numeric diff. If FAIL, do not train v10.
"""
from __future__ import annotations
import os, sys, time

import numpy as np
import torch
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_llve_v9 import VJepaLLVE_v9
from models.v10 import VJepaLLVE_v10

CFG_V9 = "/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml"
CKPT_V9 = "/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt"


def load_cfg(path):
    with open(path) as f:
        full = yaml.safe_load(f)
    # The config file contains training config too; we want the model section
    return full.get("model", full)


def main():
    device = torch.device("cuda")
    cfg = load_cfg(CFG_V9)
    print(f"[zero-iter] loaded cfg keys: {list(cfg.keys())}")

    # Build v9 reference
    print("\n=== building v9 reference ===")
    model_v9 = VJepaLLVE_v9(cfg).to(device).eval()
    raw = torch.load(CKPT_V9, map_location="cpu", weights_only=False)
    # CO2/CP2: v9.0b ckpts (from train_v9_0b.save_ckpt) store top-level "pixel_branch" only.
    if isinstance(raw, dict) and "pixel_branch" in raw:
        state = {f"pixel_branch.{k}": v for k, v in raw["pixel_branch"].items()}
        loaded_via = "pixel_branch (v9.0b style)"
    elif isinstance(raw, dict) and "model" in raw:
        state = raw["model"]; loaded_via = "model"
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]; loaded_via = "state_dict"
    else:
        state = raw; loaded_via = "raw"
    print(f"  loader: ckpt unpacked via {loaded_via!r}, {len(state)} keys")
    n_pb = sum(1 for k in state if k.startswith("pixel_branch."))
    assert n_pb > 0, f"FATAL: 0 pixel_branch.* keys in unpacked state — loader broken"
    print(f"  pixel_branch keys in state: {n_pb}")
    miss, unexp = model_v9.load_state_dict(state, strict=False)
    print(f"  v9 loaded: missing={len(miss)} unexpected={len(unexp)}")
    if miss: print(f"  missing first 5: {list(miss)[:5]}")
    if unexp: print(f"  unexpected first 5: {list(unexp)[:5]}")

    # Build v10
    print("\n=== building v10 (with VJepaPosEmb identity-init) ===")
    model_v10 = VJepaLLVE_v10(cfg).to(device).eval()
    model_v10.load_v9_weights(state)
    # Verify VJepaPosEmb is at identity-init
    if model_v10.vjepa_pos_emb is not None:
        gp = model_v10.vjepa_pos_emb.gamma_pos.item()
        pt_sum = model_v10.vjepa_pos_emb.P_t.weight.sum().item()
        print(f"  v10 pos-emb: gamma_pos={gp} (expect 1.0 after CL1), P_t_sum={pt_sum} (expect 0)")

    # Forward identical input
    print("\n=== forward equivalence test ===")
    torch.manual_seed(42)
    B, C, T, H, W = 1, 3, 16, 256, 256
    x = torch.rand(B, C, T, H, W, device=device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_v9 = model_v9(x)
            out_v10 = model_v10(x)

    pred_v9 = out_v9["pred_pixels"].float()
    pred_v10 = out_v10["pred_pixels"].float()

    diff = (pred_v10 - pred_v9).abs()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()

    # PSNR diff
    mse_v9 = (pred_v9 - x).pow(2).mean().item()
    mse_v10 = (pred_v10 - x).pow(2).mean().item()
    psnr_v9 = 10 * np.log10(1.0 / max(mse_v9, 1e-12))
    psnr_v10 = 10 * np.log10(1.0 / max(mse_v10, 1e-12))
    psnr_diff = abs(psnr_v10 - psnr_v9)

    print(f"\n  pred mean abs diff:  {mean_abs:.3e}")
    print(f"  pred max abs diff:   {max_abs:.3e}")
    print(f"  PSNR v9  vs random:  {psnr_v9:.4f}")
    print(f"  PSNR v10 vs random:  {psnr_v10:.4f}")
    print(f"  PSNR abs diff:       {psnr_diff:.4f}")

    # Verdict
    pass_mean = mean_abs < 1e-5
    pass_psnr = psnr_diff < 0.01
    print(f"\n=== VERDICT ===")
    print(f"  mean_abs < 1e-5:     {pass_mean} ({mean_abs:.3e})")
    print(f"  PSNR diff < 0.01 dB: {pass_psnr} ({psnr_diff:.4f})")
    if pass_mean and pass_psnr:
        print(f"  → PASS. v10 = v9 at iteration 0. Safe to train.")
        sys.exit(0)
    else:
        print(f"  → FAIL. Identity-at-init broken. DO NOT TRAIN v10 until fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
