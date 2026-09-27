"""tools/theta_llvestcd_eval.py — BP4 θ Step 1.

Evaluate v9.0b_mix_best (PROJECT BEST) on the LLVE-STCD test split, under the
locked tile=512/overlap=128 RGB-PSNR full-frame tiled protocol.

LLVE-STCD's published 12-seq SDSD-indoor test split:
  pair13, pair15, pair21, pair23, pair31, pair33,
  pair50, pair52, pair58, pair60, pair68, pair70

Of these, 6 are available locally in our SDSD-indoor dump (the other 6 are
missing — likely from a different SDSD release version we didn't download).
So this θ Step 1 number is a 6-seq subset of the LLVE-STCD split.

This is what BP4 calls "1. Evaluate v9.0b_mix_best under locked tile512 on any
reachable LLVE-STCD SDSD-indoor split."

Output: /cache/zihaohe/checkpoint/theta_llvestcd/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import imageio.v2 as imageio
from tools.audit_llvestcd import (
    build_v9_model, rgb_psnr_uint8, make_taper_window,
)


LLVE_STCD_TEST_SEQS = [
    "pair13", "pair15", "pair21", "pair23", "pair31", "pair33",
    "pair50", "pair52", "pair58", "pair60", "pair68", "pair70",
]


def list_frames(dirpath: str) -> List[str]:
    if not os.path.isdir(dirpath): return []
    return sorted([os.path.join(dirpath, f) for f in os.listdir(dirpath)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".npy"))])


def _load_one(path: str) -> np.ndarray:
    """Load a frame; npy files are HWC float in [0,1] (SDSD static_np convention)."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.dtype != np.uint8:
            # npy is typically already in [0,1] float
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return arr  # HWC uint8
    return imageio.imread(path)


def load_seq(input_dir: str, output_dir: str, n_frames: int = 16
             ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load LQ+GT video as (1, 3, T, H, W) float in [0,1]. Use first n_frames.
    SDSD-indoor static_np convention: input/<seq>/NNNN.npy and GT/<seq>/NNNN.npy
    are frame-aligned by name (lexicographic sort of common files)."""
    lq_paths = list_frames(input_dir)
    gt_paths = list_frames(output_dir)
    # Match by basename to ensure frame alignment
    by_name_lq = {os.path.basename(p): p for p in lq_paths}
    by_name_gt = {os.path.basename(p): p for p in gt_paths}
    common = sorted(set(by_name_lq) & set(by_name_gt))[:n_frames]
    lqs = [_load_one(by_name_lq[n]) for n in common]
    gts = [_load_one(by_name_gt[n]) for n in common]
    T = len(common)
    lq_t = torch.from_numpy(np.stack(lqs, axis=0)).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    gt_t = torch.from_numpy(np.stack(gts, axis=0)).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    return lq_t, gt_t, T


@torch.no_grad()
def tiled_inference(model, video_lq: torch.Tensor, tile=512, overlap=128,
                     autocast_dtype=torch.bfloat16) -> torch.Tensor:
    B, C, T, H, W = video_lq.shape
    device = video_lq.device; dtype = video_lq.dtype
    if H <= tile and W <= tile:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model.sample(video_lq)
        return out["pred_pixels"].clamp(0, 1)
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
                out = model.sample(patch)
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


def to_u8(x): return (x[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
def mean_psnr(p, g): return float(np.mean([rgb_psnr_uint8(p[t], g[t]) for t in range(p.shape[0])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/theta_llvestcd")
    ap.add_argument("--data_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor/indoor_static_np")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt")
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--ckpt_label", default="v9.0b_mix_best")
    ap.add_argument("--n_frames", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    print(f"[θ] device: {device}")
    print(f"[θ] ckpt: {args.ckpt} ({args.ckpt_label})")

    available = []
    missing = []
    for seq in LLVE_STCD_TEST_SEQS:
        in_d = os.path.join(args.data_root, "input", seq)
        gt_d = os.path.join(args.data_root, "GT",    seq)  # CORRECTED: real GT is in GT/, not output/
        if os.path.isdir(in_d) and os.path.isdir(gt_d):
            available.append((seq, in_d, gt_d))
        else:
            missing.append(seq)
    print(f"[θ] LLVE-STCD test seqs: {len(available)}/12 available locally")
    print(f"     available: {[s for s, _, _ in available]}")
    print(f"     missing  : {missing}")

    if not available:
        print("[θ] ABORT: no LLVE-STCD seqs available locally"); return

    model = build_v9_model(args.cfg, args.ckpt, device)

    rows = []
    print(f"\n=== Running v9.0b_mix tile=512 on {len(available)} LLVE-STCD seqs ===")
    for seq, in_d, gt_d in available:
        t0 = time.time()
        lq, gt, T = load_seq(in_d, gt_d, n_frames=args.n_frames)
        H, W = lq.shape[-2:]
        lq, gt = lq.to(device), gt.to(device)
        pred = tiled_inference(model, lq, tile=512, overlap=128)
        psnr = mean_psnr(to_u8(pred), to_u8(gt))
        # LQ baseline for context
        psnr_lq = mean_psnr(to_u8(lq), to_u8(gt))
        rows.append({"seq": seq, "frames": T, "HxW": f"{H}x{W}",
                      "lq_psnr": psnr_lq, f"{args.ckpt_label}_psnr": psnr,
                      "time_s": round(time.time() - t0, 1)})
        print(f"  [{seq}] LQ={psnr_lq:.2f}  v9.0b_mix={psnr:.3f}  ({time.time()-t0:.1f}s)")

    # CSV
    csv_path = os.path.join(args.out_dir, "theta_llvestcd_psnr.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}")

    # Aggregate
    n = len(rows)
    mean_lq = float(np.mean([r["lq_psnr"] for r in rows]))
    mean_v90b = float(np.mean([r[f"{args.ckpt_label}_psnr"] for r in rows]))

    # MD summary
    md_path = os.path.join(args.out_dir, "theta_summary.md")
    LLVE_STCD_REPORTED = 28.93  # their published number on full 12-seq split
    with open(md_path, "w") as f:
        f.write("# θ — Cross-dataset SOTA benchmark alignment (BP4 Step 1)\n\n")
        f.write(f"**Model**: {args.ckpt_label}\n")
        f.write(f"**Protocol**: tile=512 / overlap=128 / RGB-PSNR uint8 full-frame tiled\n")
        f.write(f"**Test split**: LLVE-STCD's published 12-seq SDSD-indoor test split\n")
        f.write(f"**Available locally**: {len(rows)}/12 ({[r['seq'] for r in rows]})\n")
        f.write(f"**Missing (not in our local SDSD release)**: {missing}\n\n")
        f.write("## Per-seq results\n\n")
        f.write("| seq | frames | HxW | LQ PSNR | v9.0b_mix PSNR | Δ vs LQ |\n")
        f.write("|---|---:|---|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['seq']} | {r['frames']} | {r['HxW']} | "
                    f"{r['lq_psnr']:.3f} | {r[f'{args.ckpt_label}_psnr']:.3f} | "
                    f"{r[f'{args.ckpt_label}_psnr'] - r['lq_psnr']:+.3f} |\n")
        f.write(f"\n## Aggregate ({n}/12 available LLVE-STCD test seqs)\n\n")
        f.write(f"```text\n")
        f.write(f"LQ mean       : {mean_lq:.3f} dB\n")
        f.write(f"v9.0b_mix mean: {mean_v90b:.3f} dB\n")
        f.write(f"\n")
        f.write(f"Reference (LLVE-STCD published):\n")
        f.write(f"  their model on FULL 12-seq SDSD-indoor split: {LLVE_STCD_REPORTED} dB\n")
        f.write(f"\n")
        f.write(f"PROVISIONAL same-pipeline gap:\n")
        f.write(f"  LLVE-STCD - ours = {LLVE_STCD_REPORTED} - {mean_v90b:.3f} = {LLVE_STCD_REPORTED - mean_v90b:+.3f} dB\n")
        f.write(f"\n")
        f.write(f"CAVEATS:\n")
        f.write(f"  - LLVE-STCD's 28.93 is THEIR model on their full 12 seqs.\n")
        f.write(f"  - Ours is v9.0b_mix on 6/12 of those seqs.\n")
        f.write(f"  - Cannot conclude whether the missing 6 seqs would shift our number up or down.\n")
        f.write(f"  - To tighten: also run v9.0b_mix on our 14 SDSD test seqs (already done in v9.4d: 25.738),\n")
        f.write(f"    and compare LQ on both sets to estimate split difficulty.\n")
        f.write(f"```\n\n")
        f.write(f"## BP4 verdict\n\n")
        gap = LLVE_STCD_REPORTED - mean_v90b
        if gap >= 2.5:
            verdict = (f"GAP ≥ 2.5 dB ({gap:.3f}). Per BP4: skip ε/v9.5; go to v10/data planning.")
        elif gap <= 1.0:
            verdict = (f"GAP ≤ 1.0 dB ({gap:.3f}). Per BP4: paper/technical-note path becomes realistic.")
        else:
            verdict = (f"GAP in [1.0, 2.5] band ({gap:.3f}). Per BP4: run ε v9.5-large-context-finetune.")
        f.write(f"**{verdict}**\n")
    print(f"wrote {md_path}")

    print(f"\n=== AGGREGATE ===")
    print(f"  LQ mean        : {mean_lq:.3f}")
    print(f"  v9.0b_mix mean : {mean_v90b:.3f}")
    print(f"  vs LLVE-STCD's 28.93: gap = {LLVE_STCD_REPORTED - mean_v90b:+.3f} dB")
    print(f"\n=== VERDICT ===")
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
