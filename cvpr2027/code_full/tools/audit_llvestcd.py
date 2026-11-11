"""tools/audit_llvestcd.py — RGB full-frame audit matching LLVE-STCD's protocol.

Per reviewer Z3/Z4/AA1/AA2 lock spec (2026-05-02):
  - PSNR: uint8 [0,255], MSE-based, RGB 3-channel (no Y conversion).
  - Crop: NONE — eval at native 1080×1920.
  - Inference: spatial tile 256×256 with 64-pixel overlap (matches train resolution),
    cosine-taper blend on overlap, sliding window across full frame.
  - Frames: first 16 of each sequence (V-JEPA grid needs T>=8; SDSD-indoor sequences
    have 19-30 frames so first 16 is always available).
  - Sequences: ALL of our test set (14 sequences, static_np layout).
    Note: this set does NOT overlap LLVE-STCD's 12-scene split (intersection = EMPTY).
    See audit_overlap_report.txt for details.

Outputs:
  /cache/zihaohe/checkpoint/audit_llvestcd/metrics_summary.csv
  /cache/zihaohe/checkpoint/audit_llvestcd/metrics_summary.md
  /cache/zihaohe/checkpoint/audit_llvestcd/per_sequence_metrics.csv
  /cache/zihaohe/checkpoint/audit_llvestcd/audit_overlap_report.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.common import read_image_01
from data.sdsd_dataset import SDSDDataset
from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import load_config, ssim


# LLVE-STCD's 12 named test scenes (SDSD_indoor_test.yml)
LLVESTCD_SCENES = [
    "pair13", "pair15", "pair21", "pair23", "pair31", "pair33",
    "pair50", "pair52", "pair58", "pair60", "pair68", "pair70",
]


# ---------------------------------------------------------------------------
# RGB PSNR (uint8 [0,255], matches LLVE-STCD utils/util.py)
# ---------------------------------------------------------------------------
def rgb_psnr_uint8(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> float:
    """pred/gt: (H, W, 3) uint8. Returns float PSNR in dB. Matches LLVE-STCD."""
    p = pred_uint8.astype(np.float64)
    g = gt_uint8.astype(np.float64)
    mse = np.mean((p - g) ** 2)
    if mse <= 1e-10:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def rgb_ssim_torch(pred_01: torch.Tensor, gt_01: torch.Tensor) -> float:
    """Wrap utils.ssim which expects (B,C,T,H,W) in [0,1]."""
    return float(ssim(pred_01.unsqueeze(0), gt_01.unsqueeze(0)))


# ---------------------------------------------------------------------------
# Spatial tiling with cosine-taper overlap blend
# ---------------------------------------------------------------------------
def make_taper_window(h: int, w: int, overlap: int, device, dtype,
                      taper_top: bool = True, taper_bottom: bool = True,
                      taper_left: bool = True, taper_right: bool = True) -> torch.Tensor:
    """2D cosine taper. Per AB3, only taper sides that overlap with another tile
    (taper_*=True). Sides at image boundary (taper_*=False) stay at weight 1.0.
    """
    win_h = torch.ones(h, device=device, dtype=dtype)
    win_w = torch.ones(w, device=device, dtype=dtype)
    if overlap > 0:
        ramp = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, overlap, device=device, dtype=dtype)))
        if taper_top:
            win_h[:overlap] = ramp
        if taper_bottom:
            win_h[-overlap:] = ramp.flip(0)
        if taper_left:
            win_w[:overlap] = ramp
        if taper_right:
            win_w[-overlap:] = ramp.flip(0)
    return (win_h[:, None] * win_w[None, :])  # (H, W)


@torch.no_grad()
def tiled_inference(model, video_lq: torch.Tensor,
                    tile: int = 256, overlap: int = 64,
                    autocast_dtype=torch.bfloat16) -> torch.Tensor:
    """Tile-based inference for full-resolution video.

    video_lq: (1, 3, T, H, W) in [0,1]. Returns same shape pred in [0,1].
    """
    if model is None:
        return video_lq.clone()
    B, C, T, H, W = video_lq.shape
    device = video_lq.device
    dtype = video_lq.dtype

    if H <= tile and W <= tile:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model.sample(video_lq)
        return out["pred_pixels"].clamp(0, 1)

    stride = tile - overlap
    pred_acc = torch.zeros_like(video_lq)
    weight_acc = torch.zeros(1, 1, 1, H, W, device=device, dtype=dtype)

    # tile starts
    h_starts = list(range(0, max(H - tile, 0) + 1, stride))
    if h_starts[-1] != H - tile:
        h_starts.append(max(H - tile, 0))
    w_starts = list(range(0, max(W - tile, 0) + 1, stride))
    if w_starts[-1] != W - tile:
        w_starts.append(max(W - tile, 0))

    for h0 in h_starts:
        for w0 in w_starts:
            h1 = min(h0 + tile, H)
            w1 = min(w0 + tile, W)
            patch = video_lq[..., h0:h1, w0:w1]
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model.sample(patch)
            pred_patch = out["pred_pixels"].clamp(0, 1).float()
            # AB3: only taper sides that overlap with another tile (have a neighbor).
            # External image boundaries keep full weight 1.0 to avoid dark borders.
            taper = make_taper_window(
                h1 - h0, w1 - w0, overlap, device, dtype,
                taper_top=(h0 > 0), taper_bottom=(h1 < H),
                taper_left=(w0 > 0), taper_right=(w1 < W),
            )
            taper = taper.view(1, 1, 1, h1 - h0, w1 - w0)
            pred_acc[..., h0:h1, w0:w1] += pred_patch.to(dtype) * taper
            weight_acc[..., h0:h1, w0:w1] += taper

    return (pred_acc / weight_acc.clamp_min(1e-8)).clamp(0, 1)


# ---------------------------------------------------------------------------
# Sequence loader (full resolution, first N frames)
# ---------------------------------------------------------------------------
def load_seq_full(lq_paths, gt_paths, n_frames: int = 16):
    """Load first n_frames at native resolution. Returns (1,3,T,H,W) in [0,1]."""
    T = min(n_frames, len(lq_paths))
    lq = np.stack([read_image_01(lq_paths[i]) for i in range(T)], axis=0)
    gt = np.stack([read_image_01(gt_paths[i]) for i in range(T)], axis=0)
    lq_t = torch.from_numpy(lq).permute(3, 0, 1, 2).unsqueeze(0).float()
    gt_t = torch.from_numpy(gt).permute(3, 0, 1, 2).unsqueeze(0).float()
    return lq_t, gt_t


# ---------------------------------------------------------------------------
# Eval one model
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_model(model, seqs, device, n_frames: int = 16,
               tile: int = 256, overlap: int = 64, label: str = ""):
    rows = []
    rgb_psnr_per_frame = []
    ssim_per_seq = []

    for seq_name, lq_paths, gt_paths in seqs:
        t0 = time.time()
        lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
        T_used = lq.shape[2]
        H, W = lq.shape[-2:]
        lq, gt = lq.to(device), gt.to(device)

        pred = tiled_inference(model, lq, tile=tile, overlap=overlap)

        # ---- RGB PSNR uint8 [0,255] per frame ----
        pred_uint8 = (pred[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_uint8 = (gt[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        seq_psnrs = []
        for t in range(T_used):
            seq_psnrs.append(rgb_psnr_uint8(pred_uint8[t], gt_uint8[t]))
        seq_psnr_mean = float(np.mean(seq_psnrs))
        rgb_psnr_per_frame.extend(seq_psnrs)

        # ---- SSIM (per-seq, on full clip) ----
        seq_ssim = rgb_ssim_torch(pred[0], gt[0])
        ssim_per_seq.append(seq_ssim)

        elapsed = time.time() - t0
        rows.append({
            "seq": seq_name, "frames": T_used, "HxW": f"{H}x{W}",
            f"{label}_rgb_psnr_seqmean": seq_psnr_mean,
            f"{label}_ssim": seq_ssim,
            f"{label}_time_s": round(elapsed, 1),
        })
        print(f"  [{label}/{seq_name}] {T_used}f @ {H}x{W} | "
              f"RGB-PSNR seq-mean = {seq_psnr_mean:.3f} | SSIM = {seq_ssim:.4f} | {elapsed:.1f}s")

    summary = {
        "model": label,
        "rgb_psnr_seqmean": float(np.mean([r[f"{label}_rgb_psnr_seqmean"] for r in rows])),
        "rgb_psnr_framemean": float(np.mean(rgb_psnr_per_frame)),
        "ssim_seqmean": float(np.mean(ssim_per_seq)),
        "n_sequences": len(rows),
        "n_frames": len(rgb_psnr_per_frame),
    }
    return {"summary": summary, "per_seq": rows}


@torch.no_grad()
def tile_consistency_sanity(model, seqs, device, n_clips: int = 2,
                            crop: int = 512, tile: int = 256, overlap: int = 64) -> Dict:
    """AB2: compare direct vs tiled inference on a crop the model can handle directly.
    Acceptance: mean abs diff <= 1e-3 OR PSNR(direct, tiled) >= 45 dB.
    """
    print(f"\n[sanity] tile-consistency check: direct {crop}x{crop} vs tiled (tile={tile}, overlap={overlap})")
    results = []
    for seq_name, lq_paths, gt_paths in seqs[:n_clips]:
        lq, _ = load_seq_full(lq_paths, gt_paths, n_frames=16)
        H, W = lq.shape[-2:]
        h0 = (H - crop) // 2
        w0 = (W - crop) // 2
        lq_crop = lq[..., h0:h0 + crop, w0:w0 + crop].to(device)
        # Direct
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_direct = model.sample(lq_crop)
        pred_direct = out_direct["pred_pixels"].clamp(0, 1).float()
        # Tiled
        pred_tiled = tiled_inference(model, lq_crop, tile=tile, overlap=overlap).float()
        diff = (pred_direct - pred_tiled).abs()
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        mse = float(((pred_direct - pred_tiled) ** 2).mean())
        psnr_dt = 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))
        results.append({"seq": seq_name, "max_d": max_d, "mean_d": mean_d, "psnr_dt": psnr_dt})
        print(f"  [{seq_name}] max|d|={max_d:.4e}  mean|d|={mean_d:.4e}  PSNR(direct,tiled)={psnr_dt:.2f} dB")
    avg_mean_d = float(np.mean([r["mean_d"] for r in results]))
    avg_psnr = float(np.mean([r["psnr_dt"] for r in results]))
    accept = (avg_mean_d <= 1e-3) or (avg_psnr >= 45.0)
    print(f"[sanity] avg mean|d|={avg_mean_d:.4e}  avg PSNR(direct,tiled)={avg_psnr:.2f} dB  → "
          f"{'ACCEPT' if accept else 'WARN'}")
    return {"accept": accept, "avg_mean_d": avg_mean_d, "avg_psnr_dt": avg_psnr, "per_clip": results}


def build_v9_model(cfg_path, ckpt_path, device):
    cfg = load_config(cfg_path, overrides=[])
    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    model.eval()
    if ckpt_path is None:
        return model
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/audit_llvestcd")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[audit_llvestcd] device: {device}")
    print(f"[audit_llvestcd] tile={args.tile}, overlap={args.overlap}, n_frames={args.n_frames}")

    # === Sequence overlap report ===
    ds = SDSDDataset(root=args.val_root, subset="direct", split="test",
                     clip_len=16, crop_size=256, is_train=False)
    seqs = list(ds.seqs)
    our_names = [s[0] for s in seqs]
    intersect = sorted(set(our_names) & set(LLVESTCD_SCENES))

    overlap_report_path = os.path.join(args.out_dir, "audit_overlap_report.txt")
    with open(overlap_report_path, "w") as f:
        f.write("=== SDSD-indoor test sequence overlap with LLVE-STCD ===\n\n")
        f.write(f"Our static_np test set ({len(our_names)} sequences):\n")
        for n in our_names:
            f.write(f"  {n}\n")
        f.write(f"\nLLVE-STCD test set ({len(LLVESTCD_SCENES)} sequences, from SDSD_indoor_test.yml):\n")
        for n in LLVESTCD_SCENES:
            f.write(f"  {n}\n")
        f.write(f"\nIntersection ({len(intersect)} sequences):\n")
        if intersect:
            for n in intersect:
                f.write(f"  {n}\n")
            f.write(f"\nWill audit on the {len(intersect)} intersection sequences.\n")
        else:
            f.write("  EMPTY — different SDSD release versions.\n")
            f.write(f"\nFalling back to ALL {len(our_names)} of our test sequences.\n")
            f.write("Numbers reported here are NOT directly comparable to LLVE-STCD's published 28.93 dB.\n")
            f.write("To enable direct comparison, download original SDSD release matching their split.\n")
    print(f"[audit_llvestcd] wrote {overlap_report_path}")
    print(f"[audit_llvestcd] sequence intersection: {intersect or 'EMPTY'}")
    print(f"[audit_llvestcd] auditing on our {len(our_names)} test sequences")

    # === Targets ===
    targets = [
        ("LQ", None, None),
        ("v9.0_best", "configs/train_v9.yaml", "/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt"),
        ("v9.0b_mix_best", "configs/train_v9_0b.yaml", "/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt"),
        ("v9.1a_best", "configs/train_v9_1.yaml", "/cache/zihaohe/checkpoint/runs/v9_1/ckpts/best.pt"),
    ]

    # === AB2 tile-consistency sanity (run once on v9.0_best, then proceed) ===
    print(f"\n[audit_llvestcd] === AB2 sanity check (will use v9.0_best) ===")
    sanity_model = build_v9_model("configs/train_v9.yaml",
                                  "/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt", device)
    sanity = tile_consistency_sanity(sanity_model, seqs, device,
                                     n_clips=2, crop=512, tile=args.tile, overlap=args.overlap)
    sanity_path = os.path.join(args.out_dir, "tile_sanity.json")
    import json
    with open(sanity_path, "w") as f:
        json.dump(sanity, f, indent=2)
    print(f"[audit_llvestcd] wrote {sanity_path}")
    del sanity_model
    torch.cuda.empty_cache()
    if not sanity["accept"]:
        print(f"[audit_llvestcd] WARNING: tile-consistency below acceptance threshold. "
              f"Audit numbers will be reported but should be labeled as tiled-inference variant.")

    summaries = []
    per_seq_combined = defaultdict(dict)

    for label, cfg_path, ckpt_path in targets:
        print(f"\n[audit_llvestcd] === {label} ===")
        t0 = time.time()
        if cfg_path is None:
            model = None
        else:
            model = build_v9_model(cfg_path, ckpt_path, device)
        result = eval_model(model, seqs, device,
                            n_frames=args.n_frames, tile=args.tile, overlap=args.overlap,
                            label=label)
        s = result["summary"]
        print(f"  AGGREGATE: RGB-PSNR seq-mean = {s['rgb_psnr_seqmean']:.3f} | "
              f"frame-mean = {s['rgb_psnr_framemean']:.3f} | "
              f"SSIM seq-mean = {s['ssim_seqmean']:.4f} | "
              f"n_seq={s['n_sequences']} n_frames={s['n_frames']} | "
              f"time {time.time()-t0:.1f}s")
        summaries.append(s)
        for row in result["per_seq"]:
            seq = row["seq"]
            for k, v in row.items():
                per_seq_combined[seq][k] = v
        if model is not None:
            del model
            torch.cuda.empty_cache()

    # === Write summary CSV ===
    summary_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    keys = ["model", "rgb_psnr_seqmean", "rgb_psnr_framemean", "ssim_seqmean",
            "n_sequences", "n_frames"]
    with open(summary_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k) for k in keys})
    print(f"\n[audit_llvestcd] wrote {summary_csv}")

    # === Write summary MD ===
    md_path = os.path.join(args.out_dir, "metrics_summary.md")
    with open(md_path, "w") as f:
        f.write("# v9 RGB-Full-Frame Audit (LLVE-STCD-style)\n\n")
        f.write(f"Eval: SDSD-indoor test, our {summaries[0]['n_sequences']} sequences "
                f"(static_np layout), first {args.n_frames} frames each.\n")
        f.write(f"PSNR: uint8 [0,255] RGB.  SSIM: 11x11 Gaussian per-channel.\n")
        f.write(f"Inference: tile {args.tile}×{args.tile} + {args.overlap}-px overlap, cosine blend, full-frame {1080}×{1920}.\n\n")
        f.write(f"⚠ Our 14-sequence test set does NOT overlap with LLVE-STCD's 12 named scenes "
                f"(pair13/15/21/.../70). These numbers are NOT directly comparable to their reported 28.93.\n\n")
        f.write("| Model | RGB PSNR (seq-mean) | RGB PSNR (frame-mean) | SSIM (seq-mean) |\n")
        f.write("|---|---:|---:|---:|\n")
        for s in summaries:
            f.write(f"| {s['model']} | {s['rgb_psnr_seqmean']:.3f} | "
                    f"{s['rgb_psnr_framemean']:.3f} | {s['ssim_seqmean']:.4f} |\n")
    print(f"[audit_llvestcd] wrote {md_path}")

    # === Write per-sequence CSV ===
    seq_csv = os.path.join(args.out_dir, "per_sequence_metrics.csv")
    all_keys = set()
    for d in per_seq_combined.values():
        all_keys.update(d.keys())
    keys = ["seq", "frames", "HxW"] + sorted(k for k in all_keys
                                              if k not in ("seq", "frames", "HxW"))
    with open(seq_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for seq in sorted(per_seq_combined):
            w.writerow({k: per_seq_combined[seq].get(k) for k in keys})
    print(f"[audit_llvestcd] wrote {seq_csv}")
    print(f"\n[audit_llvestcd] DONE")


if __name__ == "__main__":
    main()
