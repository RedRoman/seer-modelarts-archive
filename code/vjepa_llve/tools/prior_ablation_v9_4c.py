"""tools/prior_ablation_v9_4c.py — BH v9.4c tile-consistency diagnostic.

Two checks per BH3:

Check 1 — direct-vs-tiled consistency on 512x512 crops, 3 modes.
  For 8 diagnostic seqs, extract a center 512x512 crop and run inference 2 ways:
    direct: model on the 512x512 crop in one shot (no tiling)
    tiled : tiled_inference on the same 512x512 crop with tile=256, overlap=64
  Modes: full_vjepa, cross_attn_off, zero_vjepa
  Reports:
    PSNR(pred_direct vs pred_tiled)        -- self-consistency
    PSNR(pred_direct vs GT_crop)
    PSNR(pred_tiled  vs GT_crop)
    Δ_GT = tiled - direct  (PSNR drop from tiling)

Check 2 — full-frame tiled with tile=512 vs tile=256, 14 seqs, 2 modes.
  For each of 14 SDSD-indoor static_np test seqs, run full-frame tiled inference at
    tile=512, overlap=128 (new)
    tile=256, overlap=64 (already in v9.4 cache)
  Modes: full_vjepa, cross_attn_off
  Compare Δ_off across tile sizes.

Output: /cache/zihaohe/checkpoint/prior_ablation_v9_4c/
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import (
    load_seq_full, build_v9_model, rgb_psnr_uint8, make_taper_window,
)
from tools.prior_ablation_v9_4 import (
    install_cross_attn_off, uninstall_hooks,
)


def list_seqs(root: str, split: str = "test"):
    ds = SDSDDataset(root=root, subset="direct", split=split,
                     clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


def manipulate_features(feats, mode: str):
    if mode in ("full_vjepa", "cross_attn_off"):
        return feats
    if mode == "zero_vjepa":
        return [torch.zeros_like(f) for f in feats]
    raise ValueError(mode)


@torch.no_grad()
def direct_inference(model, video_lq: torch.Tensor, mode: str,
                      autocast_dtype=torch.bfloat16) -> torch.Tensor:
    """Single-pass forward on whatever video_lq's spatial size is. No tiling."""
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        feats = model.encode_video_features(video_lq)
        feats = manipulate_features(feats, mode)
        feats = [f.to(video_lq.dtype) for f in feats]
        out = model.pixel_branch(video_lq, feats)
    return out["pred_pixels"].clamp(0, 1)


@torch.no_grad()
def tiled_inference(model, video_lq: torch.Tensor, mode: str,
                     tile: int = 256, overlap: int = 64,
                     autocast_dtype=torch.bfloat16) -> torch.Tensor:
    B, C, T, H, W = video_lq.shape
    device = video_lq.device
    dtype = video_lq.dtype

    if H <= tile and W <= tile:
        return direct_inference(model, video_lq, mode, autocast_dtype)

    stride = tile - overlap
    pred_acc = torch.zeros_like(video_lq)
    weight_acc = torch.zeros(1, 1, 1, H, W, device=device, dtype=dtype)
    h_starts = list(range(0, max(H - tile, 0) + 1, stride))
    if h_starts[-1] != H - tile: h_starts.append(max(H - tile, 0))
    w_starts = list(range(0, max(W - tile, 0) + 1, stride))
    if w_starts[-1] != W - tile: w_starts.append(max(W - tile, 0))

    for h0 in h_starts:
        for w0 in w_starts:
            h1 = min(h0 + tile, H); w1 = min(w0 + tile, W)
            patch = video_lq[..., h0:h1, w0:w1]
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                feats = model.encode_video_features(patch)
                feats = manipulate_features(feats, mode)
                feats = [f.to(patch.dtype) for f in feats]
                out = model.pixel_branch(patch, feats)
            pred_patch = out["pred_pixels"].clamp(0, 1).float()
            taper = make_taper_window(
                h1 - h0, w1 - w0, overlap, device, dtype,
                taper_top=(h0 > 0), taper_bottom=(h1 < H),
                taper_left=(w0 > 0), taper_right=(w1 < W),
            )
            taper = taper.view(1, 1, 1, h1 - h0, w1 - w0)
            pred_acc[..., h0:h1, w0:w1] += pred_patch.to(dtype) * taper
            weight_acc[..., h0:h1, w0:w1] += taper

    return (pred_acc / weight_acc.clamp_min(1e-8)).clamp(0, 1)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(1, 3, T, H, W) -> (T, H, W, 3) uint8."""
    return (x[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def mean_psnr(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    return float(np.mean([rgb_psnr_uint8(pred_u8[t], gt_u8[t]) for t in range(pred_u8.shape[0])]))


# ---------------------------------------------------------------------------
# Check 1
# ---------------------------------------------------------------------------
def run_check1(model, seqs, device, out_dir: str,
               diag_seqs=("pair6_2", "pair5_2", "pair8", "pair4",
                          "pair7_2", "pair8_2", "pair9", "pair6"),
               crop: int = 512, n_frames: int = 16):
    print(f"\n===== Check 1: direct vs tiled on {crop}x{crop} crop, 3 modes, {len(diag_seqs)} seqs =====")
    seq_by_name = {n: (n, lq, gt) for (n, lq, gt) in seqs}
    rows = []
    for seq_name in diag_seqs:
        if seq_name not in seq_by_name:
            print(f"  [skip] {seq_name} not in test set")
            continue
        n, lq_paths, gt_paths = seq_by_name[seq_name]
        lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
        H, W = lq.shape[-2:]
        h0 = (H - crop) // 2; w0 = (W - crop) // 2
        lq_crop = lq[..., h0:h0+crop, w0:w0+crop].to(device)
        gt_crop = gt[..., h0:h0+crop, w0:w0+crop]
        gt_u8 = to_uint8(gt_crop)

        for mode in ("full_vjepa", "cross_attn_off", "zero_vjepa"):
            handles = install_cross_attn_off(model) if mode == "cross_attn_off" else []
            try:
                t0 = time.time()
                pred_direct = direct_inference(model, lq_crop, mode)
                t_direct = time.time() - t0
                t0 = time.time()
                pred_tiled = tiled_inference(model, lq_crop, mode, tile=256, overlap=64)
                t_tiled = time.time() - t0

                pd_u8 = to_uint8(pred_direct)
                pt_u8 = to_uint8(pred_tiled)
                psnr_consistency = mean_psnr(pd_u8, pt_u8)
                psnr_direct_gt = mean_psnr(pd_u8, gt_u8)
                psnr_tiled_gt = mean_psnr(pt_u8, gt_u8)
                mae = float((pred_direct - pred_tiled).abs().mean().item())
                rows.append({
                    "seq": seq_name, "mode": mode,
                    "direct_PSNR_GT": psnr_direct_gt,
                    "tiled_PSNR_GT": psnr_tiled_gt,
                    "tiled_minus_direct": psnr_tiled_gt - psnr_direct_gt,
                    "self_PSNR_direct_vs_tiled": psnr_consistency,
                    "mae_direct_vs_tiled": mae,
                    "t_direct_s": round(t_direct, 2),
                    "t_tiled_s": round(t_tiled, 2),
                })
                print(f"  [{seq_name}/{mode:15s}] direct→GT={psnr_direct_gt:.2f}  tiled→GT={psnr_tiled_gt:.2f}  "
                      f"Δ_tile={psnr_tiled_gt - psnr_direct_gt:+.3f}  consistency={psnr_consistency:.2f}  mae={mae:.4f}")
            finally:
                uninstall_hooks(handles)

    # Save CSV
    csv_path = os.path.join(out_dir, "check1_direct_vs_tiled.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Check 2
# ---------------------------------------------------------------------------
def run_check2(model, seqs, device, out_dir: str, n_frames: int = 16,
                tile: int = 512, overlap: int = 128):
    print(f"\n===== Check 2: full-frame tiled tile={tile} overlap={overlap}, 14 seqs, 2 modes =====")
    rows = []
    per_seq_results = {}

    for mode in ("full_vjepa", "cross_attn_off"):
        handles = install_cross_attn_off(model) if mode == "cross_attn_off" else []
        try:
            print(f"\n--- {mode} (tile={tile}) ---")
            for seq_name, lq_paths, gt_paths in seqs:
                t0 = time.time()
                lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
                T_used = lq.shape[2]
                lq, gt = lq.to(device), gt.to(device)
                pred = tiled_inference(model, lq, mode, tile=tile, overlap=overlap)
                pred_u8 = to_uint8(pred); gt_u8 = to_uint8(gt)
                psnr = mean_psnr(pred_u8, gt_u8)
                per_seq_results.setdefault(seq_name, {})[mode] = psnr
                print(f"  [{seq_name}/{mode}] PSNR={psnr:.3f} ({time.time()-t0:.1f}s)")
        finally:
            uninstall_hooks(handles)

    # Per-seq deltas
    csv_path = os.path.join(out_dir, "check2_tile512.csv")
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["seq", "full_vjepa_t512", "cross_attn_off_t512", "delta_t512"])
        for seq in sorted(per_seq_results):
            d = per_seq_results[seq]
            f_v = d.get("full_vjepa", 0)
            o_v = d.get("cross_attn_off", 0)
            w.writerow([seq, f"{f_v:.4f}", f"{o_v:.4f}", f"{o_v - f_v:+.4f}"])
            rows.append({"seq": seq, "full_vjepa_t512": f_v, "cross_attn_off_t512": o_v,
                         "delta_t512": o_v - f_v})

    # Aggregate
    n = len(rows)
    mean_full = sum(r["full_vjepa_t512"] for r in rows) / n
    mean_off = sum(r["cross_attn_off_t512"] for r in rows) / n
    mean_delta = sum(r["delta_t512"] for r in rows) / n
    print(f"\n=== Check 2 Aggregate (tile={tile}) ===")
    print(f"  full_vjepa_t512:     {mean_full:.3f}")
    print(f"  cross_attn_off_t512: {mean_off:.3f}")
    print(f"  Δ_off (t512):        {mean_delta:+.3f}")
    print(f"\nReference (tile=256 from v9.4):")
    print(f"  full_vjepa_t256:     23.489")
    print(f"  cross_attn_off_t256: 24.548")
    print(f"  Δ_off (t256):        +1.059")
    print(f"\nshrink ratio: Δ_off(t512) / Δ_off(t256) = {mean_delta / 1.059:.2f}")
    print(f"\nwrote {csv_path}")
    return rows, dict(mean_full_t512=mean_full, mean_off_t512=mean_off, mean_delta_t512=mean_delta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/prior_ablation_v9_4c")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt")
    ap.add_argument("--cfg", default="configs/train_v9.yaml")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--check", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v9.4c] device: {device}")
    seqs = list_seqs(args.val_root, "test")
    print(f"[v9.4c] {len(seqs)} indoor test seqs")
    model = build_v9_model(args.cfg, args.ckpt, device)

    check1_rows = check2_rows = check2_agg = None

    if args.check in ("1", "both"):
        check1_rows = run_check1(model, seqs, device, args.out_dir, n_frames=args.n_frames)

    if args.check in ("2", "both"):
        check2_rows, check2_agg = run_check2(model, seqs, device, args.out_dir,
                                              n_frames=args.n_frames, tile=512, overlap=128)

    # Summary MD
    md_path = os.path.join(args.out_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("# BH v9.4c-tile-consistency-diagnostic Summary\n\n")
        if check1_rows:
            f.write("## Check 1 — direct vs tiled on 512x512 center crop (8 diag seqs × 3 modes)\n\n")
            f.write("| seq | mode | direct→GT | tiled→GT | Δ_tile | self-consistency | mae |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|\n")
            for r in check1_rows:
                f.write(f"| {r['seq']} | {r['mode']} | {r['direct_PSNR_GT']:.2f} | {r['tiled_PSNR_GT']:.2f} | "
                        f"{r['tiled_minus_direct']:+.3f} | {r['self_PSNR_direct_vs_tiled']:.2f} | "
                        f"{r['mae_direct_vs_tiled']:.4f} |\n")
            # per-mode aggregate consistency
            from collections import defaultdict
            by_mode = defaultdict(list)
            for r in check1_rows: by_mode[r["mode"]].append(r)
            f.write("\n**Per-mode aggregate (8-seq mean):**\n\n")
            f.write("| mode | direct_PSNR | tiled_PSNR | Δ_tile | self-consistency PSNR | mae |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for m, rs in by_mode.items():
                avg = lambda k: sum(r[k] for r in rs) / len(rs)
                f.write(f"| {m} | {avg('direct_PSNR_GT'):.3f} | {avg('tiled_PSNR_GT'):.3f} | "
                        f"{avg('tiled_minus_direct'):+.3f} | {avg('self_PSNR_direct_vs_tiled'):.3f} | "
                        f"{avg('mae_direct_vs_tiled'):.4f} |\n")
        if check2_rows:
            f.write("\n## Check 2 — full-frame tiled tile=512 vs tile=256, 14 seqs × 2 modes\n\n")
            f.write("| seq | full_t512 | off_t512 | Δ_t512 |\n")
            f.write("|---|---:|---:|---:|\n")
            for r in check2_rows:
                f.write(f"| {r['seq']} | {r['full_vjepa_t512']:.3f} | {r['cross_attn_off_t512']:.3f} | "
                        f"{r['delta_t512']:+.3f} |\n")
            f.write(f"\n**Aggregate**: full {check2_agg['mean_full_t512']:.3f}, off "
                    f"{check2_agg['mean_off_t512']:.3f}, Δ_off {check2_agg['mean_delta_t512']:+.3f}\n")
            f.write(f"\n**Reference (tile=256, v9.4)**: full 23.489, off 24.548, Δ_off +1.059\n")
            f.write(f"\n**Shrink ratio**: Δ_off(t512) / Δ_off(t256) = "
                    f"{check2_agg['mean_delta_t512'] / 1.059:.3f}\n")
    print(f"\nwrote {md_path}")


if __name__ == "__main__":
    main()
