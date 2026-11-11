"""D1-D3 v10-spec diagnostics per CB8.

D1. pair68 T=8 + T=16 reversed-order — choose M4 mechanism (M4a vs M4b)
D2. LLVE-STCD-12 border vs center ROI PSNR — confirm B1 hypothesis
D3. Color-offset oracle PSNR on LLVE-STCD-12 + outdoor-25 — confirm B2 + justify M5
"""
from __future__ import annotations
import os, sys, csv, time
from typing import List, Dict
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import build_v9_model, rgb_psnr_uint8, load_seq_full
from tools.theta_llvestcd_eval import (
    tiled_inference, to_u8, mean_psnr, load_seq, LLVE_STCD_TEST_SEQS,
)

CFG  = "/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml"
CKPT = "/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt"
DATA_INDOOR = "/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor/indoor_static_np"


def per_frame_psnr(pred_u8, gt_u8):
    return [rgb_psnr_uint8(pred_u8[t], gt_u8[t]) for t in range(pred_u8.shape[0])]


# ------------------------------------------------------------------
# D1
# ------------------------------------------------------------------
def run_D1(model, device):
    print("\n========== D1: pair68 T=8 + T=16 reversed ==========")
    in_d = os.path.join(DATA_INDOOR, "input/pair68")
    gt_d = os.path.join(DATA_INDOOR, "GT/pair68")

    # T=16 normal (reproduce BX1 baseline 13.40)
    print("  T=16 normal order:")
    lq16, gt16, T = load_seq(in_d, gt_d, n_frames=16)
    lq16d, gt16d = lq16.to(device), gt16.to(device)
    pred16 = tiled_inference(model, lq16d, tile=512, overlap=128)
    pred16_u8 = to_u8(pred16); gt16_u8 = to_u8(gt16d)
    pf16 = per_frame_psnr(pred16_u8, gt16_u8)
    print(f"    per-frame PSNR: {' '.join(f'{p:.2f}' for p in pf16)}")
    print(f"    mean: {np.mean(pf16):.3f}, slope (lin reg): "
          f"{np.polyfit(range(16), pf16, 1)[0]:.4f} dB/frame")

    # T=8 first
    print("\n  T=8 (first 8 frames):")
    lq8 = lq16[:, :, :8].contiguous()
    gt8 = gt16[:, :, :8].contiguous()
    lq8d, gt8d = lq8.to(device), gt8.to(device)
    pred8 = tiled_inference(model, lq8d, tile=512, overlap=128)
    pred8_u8 = to_u8(pred8); gt8_u8 = to_u8(gt8d)
    pf8 = per_frame_psnr(pred8_u8, gt8_u8)
    print(f"    per-frame PSNR: {' '.join(f'{p:.2f}' for p in pf8)}")
    print(f"    mean: {np.mean(pf8):.3f}, slope: "
          f"{np.polyfit(range(8), pf8, 1)[0]:.4f} dB/frame")

    # T=16 reversed
    print("\n  T=16 reversed (frame 15..0):")
    lq16r = lq16.flip(dims=[2]).contiguous()
    gt16r = gt16.flip(dims=[2]).contiguous()
    lq16rd, gt16rd = lq16r.to(device), gt16r.to(device)
    pred16r = tiled_inference(model, lq16rd, tile=512, overlap=128)
    pred16r_u8 = to_u8(pred16r); gt16r_u8 = to_u8(gt16rd)
    pf16r = per_frame_psnr(pred16r_u8, gt16r_u8)
    print(f"    per-frame PSNR: {' '.join(f'{p:.2f}' for p in pf16r)}")
    print(f"    mean: {np.mean(pf16r):.3f}, slope: "
          f"{np.polyfit(range(16), pf16r, 1)[0]:.4f} dB/frame")

    # M4 mechanism verdict
    slope16 = np.polyfit(range(16), pf16, 1)[0]
    slope8 = np.polyfit(range(8), pf8, 1)[0]
    slope16r = np.polyfit(range(16), pf16r, 1)[0]

    print(f"\n  D1 verdict:")
    print(f"    slope T=16 normal:   {slope16:+.4f} dB/frame")
    print(f"    slope T=8 first:     {slope8:+.4f} dB/frame")
    print(f"    slope T=16 reversed: {slope16r:+.4f} dB/frame")

    if abs(slope8) < 0.5 * abs(slope16):
        m4_choice = "M4b temporal consistency loss (degradation halves at T=8 → accumulation)"
    elif slope16r > 0:  # reversed slope flips sign → degradation tied to FRAME INDEX
        m4_choice = "M4a temporal positional encoding (reversed flips slope → frame index dependence)"
    elif abs(slope16r) > 0.8 * abs(slope16):
        m4_choice = "M4b temporal consistency loss (reversed degrades same direction → scene-time-tied accumulation)"
    else:
        m4_choice = f"INDETERMINATE (slope8={slope8:.4f}, slope16r={slope16r:.4f}); recommend M4a + M4b combined"

    print(f"  → M4 mechanism choice: {m4_choice}")
    return {"T16": pf16, "T8": pf8, "T16r": pf16r,
            "slope_T16": float(slope16), "slope_T8": float(slope8),
            "slope_T16r": float(slope16r), "m4_choice": m4_choice}


# ------------------------------------------------------------------
# D2
# ------------------------------------------------------------------
def run_D2(model, device, edge_w=156):
    print(f"\n========== D2: LLVE-STCD-12 border vs center ROI (edge={edge_w}) ==========")
    rows = []
    available = []
    for seq in LLVE_STCD_TEST_SEQS:
        in_d = os.path.join(DATA_INDOOR, "input", seq)
        gt_d = os.path.join(DATA_INDOOR, "GT", seq)
        if os.path.isdir(in_d) and os.path.isdir(gt_d):
            available.append(seq)
    print(f"  available: {len(available)}/12")

    for seq in available:
        in_d = os.path.join(DATA_INDOOR, "input", seq)
        gt_d = os.path.join(DATA_INDOOR, "GT", seq)
        t0 = time.time()
        lq, gt, T = load_seq(in_d, gt_d, n_frames=16)
        H, W = lq.shape[-2:]
        lqd, gtd = lq.to(device), gt.to(device)
        pred = tiled_inference(model, lqd, tile=512, overlap=128)
        pred_u8 = to_u8(pred); gt_u8 = to_u8(gtd)

        # Full
        psnr_full = mean_psnr(pred_u8, gt_u8)

        # Center ROI (cut edge_w from each side)
        center_pred = pred_u8[:, edge_w:-edge_w, edge_w:-edge_w]
        center_gt = gt_u8[:, edge_w:-edge_w, edge_w:-edge_w]
        psnr_center = mean_psnr(center_pred, center_gt)

        # Edge mask: pixels in border ring
        # Compute by setting center to 0 in error
        # Simpler: compute MSE on border ring directly
        mse_full_per_frame = []
        mse_edge_per_frame = []
        for t in range(T):
            err = (pred_u8[t].astype(np.float64) - gt_u8[t].astype(np.float64))**2
            mse_full_per_frame.append(err.mean())
            mask = np.ones((H, W), dtype=bool)
            mask[edge_w:-edge_w, edge_w:-edge_w] = False
            mse_edge_per_frame.append(err.mean(axis=-1)[mask].mean())
        mse_edge = float(np.mean(mse_edge_per_frame))
        psnr_edge = 10 * np.log10(255**2 / max(mse_edge, 1e-8))

        rows.append({"seq": seq, "psnr_full": psnr_full,
                     "psnr_center": psnr_center, "psnr_edge": psnr_edge,
                     "diff_center_full": psnr_center - psnr_full,
                     "diff_full_edge": psnr_full - psnr_edge})
        print(f"    [{seq}] full={psnr_full:.3f}  center={psnr_center:.3f} (Δ={psnr_center-psnr_full:+.3f})  "
              f"edge={psnr_edge:.3f} (Δ_full-edge={psnr_full-psnr_edge:+.3f})  ({time.time()-t0:.1f}s)")

    n = len(rows)
    mean_full = float(np.mean([r["psnr_full"] for r in rows]))
    mean_center = float(np.mean([r["psnr_center"] for r in rows]))
    mean_edge = float(np.mean([r["psnr_edge"] for r in rows]))
    print(f"\n  D2 verdict (n={n}):")
    print(f"    mean PSNR full:    {mean_full:.3f}")
    print(f"    mean PSNR center:  {mean_center:.3f}  (center boost: {mean_center-mean_full:+.3f})")
    print(f"    mean PSNR edge:    {mean_edge:.3f}  (edge drag: {mean_full-mean_edge:+.3f})")
    if mean_center - mean_full > 1.0:
        b1_verdict = "B1 STRONGLY CONFIRMED: center is much better than full → border failure dominant"
    elif mean_center - mean_full > 0.5:
        b1_verdict = "B1 confirmed (moderate): center beats full by 0.5-1 dB"
    else:
        b1_verdict = "B1 NOT confirmed: center ~= full → edge isn't the dominant problem"
    print(f"    → {b1_verdict}")
    return {"rows": rows, "mean_full": mean_full, "mean_center": mean_center,
            "mean_edge": mean_edge, "b1_verdict": b1_verdict}


# ------------------------------------------------------------------
# D3
# ------------------------------------------------------------------
def run_D3(model, device):
    print(f"\n========== D3: color-offset oracle PSNR ==========")
    rows_indoor = []
    rows_outdoor = []

    # indoor LLVE-STCD-12
    print("\n  LLVE-STCD-12:")
    available = []
    for seq in LLVE_STCD_TEST_SEQS:
        in_d = os.path.join(DATA_INDOOR, "input", seq)
        gt_d = os.path.join(DATA_INDOOR, "GT", seq)
        if os.path.isdir(in_d) and os.path.isdir(gt_d):
            available.append((seq, in_d, gt_d))

    for seq, in_d, gt_d in available:
        lq, gt, T = load_seq(in_d, gt_d, n_frames=16)
        lqd, gtd = lq.to(device), gt.to(device)
        pred = tiled_inference(model, lqd, tile=512, overlap=128)
        pred_u8 = to_u8(pred); gt_u8 = to_u8(gtd)

        # Per-frame per-channel mean offset
        psnr_orig = mean_psnr(pred_u8, gt_u8)
        # Oracle: shift pred to match GT mean per (frame, channel)
        pred_f32 = pred_u8.astype(np.float32)
        gt_f32 = gt_u8.astype(np.float32)
        gt_mean_fc = gt_f32.mean(axis=(1,2), keepdims=True)  # (T, 1, 1, 3)
        pred_mean_fc = pred_f32.mean(axis=(1,2), keepdims=True)
        offset = gt_mean_fc - pred_mean_fc
        pred_oracle = (pred_f32 + offset).clip(0, 255).astype(np.uint8)
        psnr_oracle = mean_psnr(pred_oracle, gt_u8)

        rows_indoor.append({"seq": seq, "psnr_orig": psnr_orig, "psnr_oracle": psnr_oracle,
                             "diff": psnr_oracle - psnr_orig})
        print(f"    [{seq}] orig={psnr_orig:.3f}  oracle={psnr_oracle:.3f}  Δ={psnr_oracle-psnr_orig:+.3f}")

    # Outdoor 25 (reuse audit cache if exists, otherwise re-run)
    print("\n  outdoor-25:")
    ds = SDSDDataset(root="/home/ma-user/workspace/zihaohe/datasets/SDSD_outdoor",
                     subset="direct", split="test", clip_len=16, crop_size=256, is_train=False)
    for seq_name, lq_paths, gt_paths in list(ds.seqs):
        lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=16)
        lqd, gtd = lq.to(device), gt.to(device)
        try:
            pred = tiled_inference(model, lqd, tile=512, overlap=128)
        except Exception as e:
            print(f"    [{seq_name}] FAIL {type(e).__name__}: {str(e)[:80]}"); continue
        pred_u8 = to_u8(pred); gt_u8 = to_u8(gtd)
        psnr_orig = mean_psnr(pred_u8, gt_u8)
        pred_f32 = pred_u8.astype(np.float32); gt_f32 = gt_u8.astype(np.float32)
        gt_mean_fc = gt_f32.mean(axis=(1,2), keepdims=True)
        pred_mean_fc = pred_f32.mean(axis=(1,2), keepdims=True)
        pred_oracle = (pred_f32 + (gt_mean_fc - pred_mean_fc)).clip(0,255).astype(np.uint8)
        psnr_oracle = mean_psnr(pred_oracle, gt_u8)
        rows_outdoor.append({"seq": seq_name, "psnr_orig": psnr_orig, "psnr_oracle": psnr_oracle,
                              "diff": psnr_oracle - psnr_orig})
        print(f"    [{seq_name}] orig={psnr_orig:.3f}  oracle={psnr_oracle:.3f}  Δ={psnr_oracle-psnr_orig:+.3f}")

    mean_indoor_orig = float(np.mean([r["psnr_orig"] for r in rows_indoor]))
    mean_indoor_oracle = float(np.mean([r["psnr_oracle"] for r in rows_indoor]))
    mean_outdoor_orig = float(np.mean([r["psnr_orig"] for r in rows_outdoor]))
    mean_outdoor_oracle = float(np.mean([r["psnr_oracle"] for r in rows_outdoor]))
    delta_indoor = mean_indoor_oracle - mean_indoor_orig
    delta_outdoor = mean_outdoor_oracle - mean_outdoor_orig
    print(f"\n  D3 verdict:")
    print(f"    indoor LLVE-STCD-12: orig={mean_indoor_orig:.3f}  oracle={mean_indoor_oracle:.3f}  "
          f"Δ={delta_indoor:+.3f}")
    print(f"    outdoor-25:          orig={mean_outdoor_orig:.3f}  oracle={mean_outdoor_oracle:.3f}  "
          f"Δ={delta_outdoor:+.3f}")
    if delta_indoor > 1.5 or delta_outdoor > 1.5:
        b2_verdict = (f"B2 STRONGLY CONFIRMED: oracle color shift recovers ≥1.5 dB → "
                      "global photometric alignment loss (M5) will give material gain")
    elif delta_indoor > 0.5 or delta_outdoor > 0.5:
        b2_verdict = (f"B2 confirmed (moderate): oracle color shift recovers 0.5-1.5 dB → "
                      "M5 worth implementing")
    else:
        b2_verdict = (f"B2 NOT confirmed: oracle shift gives <0.5 dB → M5 may be lower priority")
    print(f"    → {b2_verdict}")
    return {"indoor": rows_indoor, "outdoor": rows_outdoor,
            "delta_indoor": delta_indoor, "delta_outdoor": delta_outdoor,
            "b2_verdict": b2_verdict}


def main():
    out_dir = "/cache/zihaohe/checkpoint/d1_d2_d3"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")
    print(f"[D1-D3] device: {device}")
    print(f"[D1-D3] ckpt: {CKPT}")
    model = build_v9_model(CFG, CKPT, device)

    d1 = run_D1(model, device)
    d2 = run_D2(model, device)
    d3 = run_D3(model, device)

    # Save summary
    md = os.path.join(out_dir, "d1_d2_d3_summary.md")
    with open(md, "w") as f:
        f.write("# D1-D3 v10-spec input diagnostics\n\n")
        f.write("## D1 — M4 mechanism: pair68 T=8 + T=16 reversed\n\n")
        f.write(f"  T=16 normal slope:   {d1['slope_T16']:+.4f} dB/frame\n")
        f.write(f"  T=8 first slope:     {d1['slope_T8']:+.4f} dB/frame\n")
        f.write(f"  T=16 reversed slope: {d1['slope_T16r']:+.4f} dB/frame\n\n")
        f.write(f"**Verdict:** {d1['m4_choice']}\n\n")
        f.write("## D2 — Border vs center ROI (LLVE-STCD-12)\n\n")
        f.write(f"  mean full: {d2['mean_full']:.3f}\n  mean center: {d2['mean_center']:.3f}  "
                f"(boost: {d2['mean_center']-d2['mean_full']:+.3f})\n  mean edge: {d2['mean_edge']:.3f}  "
                f"(drag: {d2['mean_full']-d2['mean_edge']:+.3f})\n\n")
        f.write(f"**Verdict:** {d2['b1_verdict']}\n\n")
        f.write("## D3 — Color-offset oracle\n\n")
        f.write(f"  indoor LLVE-STCD-12 oracle Δ: {d3['delta_indoor']:+.3f}\n")
        f.write(f"  outdoor-25 oracle Δ:           {d3['delta_outdoor']:+.3f}\n\n")
        f.write(f"**Verdict:** {d3['b2_verdict']}\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
