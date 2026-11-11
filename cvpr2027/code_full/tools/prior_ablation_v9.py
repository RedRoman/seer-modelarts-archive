"""tools/prior_ablation_v9.py — AZ2/BA1 inference-only V-JEPA prior ablation.

For each of 4 modes:
  Mode 0: full_vjepa            (baseline, no manipulation)
  Mode 1: temporal_center       (replace all T'=8 planes with center plane t=3)
  Mode 2: temporal_shuffle      (shuffle T'=8 axis with fixed seed)
  Mode 3: spatial_shuffle       (shuffle 256 spatial-token positions with fixed seed)

Run on v9.0_best, 14 SDSD-indoor static_np test seqs, RGB-PSNR full-frame
tiled inference (matches AA4/AT2 protocol).

Output: /cache/zihaohe/checkpoint/prior_ablation_v9/
  ablation_summary.md   (per-mode RGB-PSNR + Δ + AZ3 fork verdict)
  per_seq.csv           (per-seq + per-mode values)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import (
    load_seq_full, build_v9_model, rgb_psnr_uint8, make_taper_window,
)


def list_seqs(root: str, split: str = "test"):
    ds = SDSDDataset(root=root, subset="direct", split=split,
                     clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


def manipulate_features(feats: List[torch.Tensor], mode: str,
                        token_grid=(8, 16, 16), seed: int = 42) -> List[torch.Tensor]:
    """Per-layer feats: (B, N=T'×H'×W', D). Manipulate per AZ2 spec."""
    if mode == "full_vjepa":
        return feats

    Tp, Hp, Wp = token_grid
    N_per_plane = Hp * Wp
    out = []
    g = torch.Generator(device="cpu").manual_seed(seed)

    for f in feats:
        B, N, D = f.shape
        assert N == Tp * N_per_plane, f"feat shape {f.shape} doesn't match grid {token_grid}"
        f5 = f.view(B, Tp, N_per_plane, D)  # (B, T', S, D)

        if mode == "temporal_center":
            center = Tp // 2  # for T'=8, center = 4 (or 3)
            center_plane = f5[:, center:center + 1].expand(-1, Tp, -1, -1).contiguous()
            f_out = center_plane

        elif mode == "temporal_shuffle":
            perm = torch.randperm(Tp, generator=g)
            f_out = f5[:, perm].contiguous()

        elif mode == "spatial_shuffle":
            perm = torch.randperm(N_per_plane, generator=g)
            f_out = f5[:, :, perm].contiguous()

        else:
            raise ValueError(f"unknown mode {mode!r}")

        out.append(f_out.view(B, N, D))
    return out


@torch.no_grad()
def tiled_inference_with_intervention(model, video_lq: torch.Tensor, mode: str,
                                       tile: int = 256, overlap: int = 64,
                                       autocast_dtype=torch.bfloat16) -> torch.Tensor:
    """Tile-based inference where V-JEPA features are intervened per `mode`."""
    B, C, T, H, W = video_lq.shape
    device = video_lq.device
    dtype = video_lq.dtype

    if H <= tile and W <= tile:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            feats = model.encode_video_features(video_lq)
            feats = manipulate_features(feats, mode)
            feats = [f.to(video_lq.dtype) for f in feats]
            out = model.pixel_branch(video_lq, feats)
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
            h1 = min(h0 + tile, H)
            w1 = min(w0 + tile, W)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/prior_ablation_v9")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt")
    ap.add_argument("--cfg", default="configs/train_v9.yaml")
    ap.add_argument("--n_frames", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prior_ablation] device: {device}")

    seqs = list_seqs(args.val_root, "test")
    print(f"[prior_ablation] {len(seqs)} indoor test seqs")

    model = build_v9_model(args.cfg, args.ckpt, device)

    modes = ["full_vjepa", "temporal_center", "temporal_shuffle", "spatial_shuffle"]
    per_seq_results = {}  # seq → {mode: psnr}

    for mode in modes:
        print(f"\n[prior_ablation] === {mode} ===")
        for seq_name, lq_paths, gt_paths in seqs:
            t0 = time.time()
            lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=args.n_frames)
            T_used = lq.shape[2]
            H, W = lq.shape[-2:]
            lq, gt = lq.to(device), gt.to(device)

            pred = tiled_inference_with_intervention(model, lq, mode=mode)

            pred_uint8 = (pred[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_uint8 = (gt[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            psnrs = [rgb_psnr_uint8(pred_uint8[t], gt_uint8[t]) for t in range(T_used)]
            seq_psnr = float(np.mean(psnrs))
            per_seq_results.setdefault(seq_name, {})[mode] = seq_psnr
            print(f"  [{mode}/{seq_name}] PSNR={seq_psnr:.3f} ({time.time()-t0:.1f}s)")

    # Aggregate
    aggregate = {}
    for mode in modes:
        vals = [per_seq_results[s][mode] for s in per_seq_results if mode in per_seq_results[s]]
        aggregate[mode] = {"mean": float(np.mean(vals)), "n": len(vals)}

    baseline = aggregate["full_vjepa"]["mean"]
    deltas = {m: aggregate[m]["mean"] - baseline for m in modes}
    print(f"\n=== AGGREGATE ===")
    for m in modes:
        print(f"  {m:20s} PSNR={aggregate[m]['mean']:.3f}  Δ vs baseline={deltas[m]:+.3f}")

    # AZ3 fork verdict
    d_temp_center = deltas["temporal_center"]
    d_temp_shuffle = deltas["temporal_shuffle"]
    d_spatial_shuffle = deltas["spatial_shuffle"]

    def cls(d):
        if abs(d) < 0.2: return "no dep"
        elif abs(d) < 0.5: return "weak"
        else: return "meaningful"

    if cls(d_temp_center) == "no dep" and cls(d_temp_shuffle) == "no dep":
        if cls(d_spatial_shuffle) == "meaningful":
            verdict = "Fork C: spatial shuffle hurts but temporal does not — V-JEPA mainly spatial semantic prior. Next: backbone control (DINOv2/VideoMAE) before BVI-RLV."
        else:
            verdict = "Fork D: all ablations barely move PSNR — V-JEPA path may be robustly bypassed at inference. noJEPA gap is likely training-dynamics, not feature-dependence."
    elif cls(d_temp_center) == "no dep" and cls(d_temp_shuffle) != "no dep":
        verdict = "Fork B-weak: temporal ordering matters but redundancy is enough. Small temporal adapter may be revisitable with this evidence."
    elif cls(d_temp_center) != "no dep":
        verdict = "Fork B: cross-frame V-JEPA context is consumed. Justifies BVI-RLV motion-rich scale-up + V-JEPA-specific paper story."
    else:
        verdict = "Mixed signal — see per-mode deltas; report and discuss."

    # === Write per-seq CSV ===
    csv_path = os.path.join(args.out_dir, "per_seq.csv")
    keys = ["seq"] + modes + [f"delta_{m}" for m in modes if m != "full_vjepa"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for seq in sorted(per_seq_results):
            d = per_seq_results[seq]
            row = {"seq": seq}
            for m in modes: row[m] = d.get(m)
            for m in modes:
                if m != "full_vjepa" and m in d and "full_vjepa" in d:
                    row[f"delta_{m}"] = d[m] - d["full_vjepa"]
            w.writerow(row)
    print(f"\nwrote {csv_path}")

    # === Write summary MD ===
    md_path = os.path.join(args.out_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("# AZ2/BA1 v9.3-prior-ablation Summary\n\n")
        f.write("4-mode V-JEPA token manipulation on v9.0_best, inference-only.\n")
        f.write("Eval: 14 SDSD-indoor static_np test seqs, RGB-PSNR uint8 full-frame tiled.\n\n")

        f.write("## Aggregate per-mode\n\n")
        f.write("| Mode | RGB-PSNR (mean) | Δ vs baseline | Classification |\n")
        f.write("|---|---:|---:|:---:|\n")
        for m in modes:
            d = deltas[m]
            f.write(f"| {m} | {aggregate[m]['mean']:.3f} | {d:+.3f} | {cls(d) if m != 'full_vjepa' else '—'} |\n")

        f.write("\n## AZ3 fork verdict\n\n")
        f.write(f"**{verdict}**\n\n")

        f.write("## Per-seq detail\n\n")
        f.write("| seq | full_vjepa | temporal_center | Δ | temporal_shuffle | Δ | spatial_shuffle | Δ |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for seq in sorted(per_seq_results):
            d = per_seq_results[seq]
            f.write(f"| {seq} | {d.get('full_vjepa', 0):.3f} | "
                    f"{d.get('temporal_center', 0):.3f} | "
                    f"{(d.get('temporal_center', 0) - d.get('full_vjepa', 0)):+.3f} | "
                    f"{d.get('temporal_shuffle', 0):.3f} | "
                    f"{(d.get('temporal_shuffle', 0) - d.get('full_vjepa', 0)):+.3f} | "
                    f"{d.get('spatial_shuffle', 0):.3f} | "
                    f"{(d.get('spatial_shuffle', 0) - d.get('full_vjepa', 0)):+.3f} |\n")
    print(f"wrote {md_path}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
