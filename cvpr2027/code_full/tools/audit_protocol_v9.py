"""tools/audit_protocol_v9.py — protocol & metric audit per reviewer O5.

Goal: answer "is the local 25 dB ceiling a metric/protocol artifact, an
optimization-bound, or a generalization-bound?"

Loads multiple ckpts (LQ baseline + v9.0 best/ema + v9.1a best). For each:
  RGB-PSNR @ center-256
  Y-PSNR   @ center-256
  RGB-PSNR @ center-512 (if frame size allows)
  Y-PSNR   @ center-512
  SSIM     @ center-256
  per-sequence breakdown (CSV)
  per-frame mean (in addition to per-sequence mean)

Outputs:
  metrics_summary.csv
  metrics_summary.md
  per_sequence_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
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


# ---------------------------------------------------------------------------
# PSNR helpers (per-frame, then aggregate)
# ---------------------------------------------------------------------------
def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma. img (B,3,H,W) in [0,1] → (B,1,H,W) in [0,1]."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def psnr_per_frame(pred: torch.Tensor, gt: torch.Tensor,
                   max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """pred/gt (N, C, H, W) → (N,) PSNR in dB."""
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3)).clamp_min(eps)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """x (N,C,H,W) → centered (N,C,size,size). Returns x unchanged if too small."""
    H, W = x.shape[-2:]
    if H < size or W < size:
        return x
    h0 = (H - size) // 2
    w0 = (W - size) // 2
    return x[..., h0:h0 + size, w0:w0 + size]


# ---------------------------------------------------------------------------
# Sequence loader (full clips, no random crop)
# ---------------------------------------------------------------------------
def list_sdsd_sequences(root: str, subset: str, split: str) -> List[Tuple[str, List[str], List[str]]]:
    """Use SDSDDataset's own sequence list to match the training/val split."""
    ds = SDSDDataset(root=root, subset=subset, split=split,
                     clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


def load_clip(lq_paths: List[str], gt_paths: List[str], clip_len: int = 16,
              crop_size: int = 256, mode: str = "center") -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Center-window of the sequence. Returns (lq, gt) of shape (1, 3, T, H, W).

    mode='center'  → take middle T frames.
    mode='first'   → take first T frames.
    """
    n_total = len(lq_paths)
    T = min(clip_len, n_total)
    if mode == "center":
        start = (n_total - T) // 2
    else:
        start = 0
    lq_frames = np.stack([read_image_01(lq_paths[start + i]) for i in range(T)], axis=0)
    gt_frames = np.stack([read_image_01(gt_paths[start + i]) for i in range(T)], axis=0)
    # frames are (T, H, W, 3); convert to (1, 3, T, H, W)
    lq = torch.from_numpy(lq_frames).permute(3, 0, 1, 2).unsqueeze(0).float()
    gt = torch.from_numpy(gt_frames).permute(3, 0, 1, 2).unsqueeze(0).float()
    H, W = lq.shape[-2:]
    if crop_size and (H >= crop_size and W >= crop_size):
        h0 = (H - crop_size) // 2
        w0 = (W - crop_size) // 2
        lq = lq[..., h0:h0 + crop_size, w0:w0 + crop_size]
        gt = gt[..., h0:h0 + crop_size, w0:w0 + crop_size]
    return lq, gt, f"start={start}_T={T}_HxW={H}x{W}"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def build_v9_model(cfg_path: str, ckpt_path: Optional[str], device: torch.device) -> Optional[torch.nn.Module]:
    """Build VJepaLLVE_v9 and optionally load pixel_branch state."""
    cfg = load_config(cfg_path, overrides=[])
    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    model.eval()
    if ckpt_path is None:
        return model  # untrained / identity
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "pixel_branch" not in state:
        raise RuntimeError(f"Ckpt {ckpt_path} missing 'pixel_branch' key")
    model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    return model


# ---------------------------------------------------------------------------
# Eval one model on full val set
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_model(model: Optional[torch.nn.Module], seqs: List, device: torch.device,
               clip_len: int = 16, crop_size: int = 256, label: str = "") -> Dict:
    """If model is None, treat as LQ baseline."""
    rows = []  # per-sequence
    rgb256_all = []
    y256_all = []
    rgb512_all = []
    y512_all = []
    ssim256_all = []

    for seq_name, lq_paths, gt_paths in seqs:
        lq, gt, meta = load_clip(lq_paths, gt_paths, clip_len=clip_len, crop_size=None, mode="center")
        H_full, W_full = lq.shape[-2:]
        lq, gt = lq.to(device), gt.to(device)

        # 256 center-crop view
        lq256 = center_crop(lq, 256)
        gt256 = center_crop(gt, 256)

        # 512 center-crop view (if frame allows)
        lq512 = center_crop(lq, 512) if min(H_full, W_full) >= 512 else None
        gt512 = center_crop(gt, 512) if min(H_full, W_full) >= 512 else None

        # === infer (or LQ pass-through) ===
        if model is None:
            pred256 = lq256
            pred512 = lq512
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out256 = model.sample(lq256)
            pred256 = out256["pred_pixels"].clamp(0, 1).float()
            if lq512 is not None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out512 = model.sample(lq512)
                pred512 = out512["pred_pixels"].clamp(0, 1).float()
            else:
                pred512 = None

        # === metrics @ 256 ===
        # flatten (1,3,T,H,W) → (T,3,H,W)
        p256 = pred256[0].permute(1, 0, 2, 3)
        g256 = gt256[0].permute(1, 0, 2, 3)
        rgb256 = psnr_per_frame(p256, g256)
        y256 = psnr_per_frame(rgb_to_y(p256), rgb_to_y(g256))
        ssim256 = ssim(pred256, gt256)

        rgb256_all.extend(rgb256.cpu().tolist())
        y256_all.extend(y256.cpu().tolist())
        ssim256_all.append(float(ssim256))

        row = {
            "seq": seq_name,
            "frames": p256.shape[0],
            "HxW_full": f"{H_full}x{W_full}",
            f"{label}_rgb256_seqmean": float(rgb256.mean()),
            f"{label}_y256_seqmean": float(y256.mean()),
            f"{label}_ssim256": float(ssim256),
        }

        # === metrics @ 512 ===
        if pred512 is not None and gt512 is not None:
            p512 = pred512[0].permute(1, 0, 2, 3)
            g512 = gt512[0].permute(1, 0, 2, 3)
            rgb512 = psnr_per_frame(p512, g512)
            y512 = psnr_per_frame(rgb_to_y(p512), rgb_to_y(g512))
            rgb512_all.extend(rgb512.cpu().tolist())
            y512_all.extend(y512.cpu().tolist())
            row[f"{label}_rgb512_seqmean"] = float(rgb512.mean())
            row[f"{label}_y512_seqmean"] = float(y512.mean())
        else:
            row[f"{label}_rgb512_seqmean"] = None
            row[f"{label}_y512_seqmean"] = None

        rows.append(row)

    summary = {
        "model": label,
        "rgb256_framemean": float(np.mean(rgb256_all)) if rgb256_all else None,
        "rgb256_seqmean": float(np.mean([r[f"{label}_rgb256_seqmean"] for r in rows])),
        "y256_framemean": float(np.mean(y256_all)) if y256_all else None,
        "y256_seqmean": float(np.mean([r[f"{label}_y256_seqmean"] for r in rows])),
        "ssim256_seqmean": float(np.mean(ssim256_all)),
        "rgb512_framemean": float(np.mean(rgb512_all)) if rgb512_all else None,
        "rgb512_seqmean": float(np.mean([r[f"{label}_rgb512_seqmean"] for r in rows if r[f"{label}_rgb512_seqmean"] is not None]))
                          if rgb512_all else None,
        "y512_framemean": float(np.mean(y512_all)) if y512_all else None,
        "y512_seqmean": float(np.mean([r[f"{label}_y512_seqmean"] for r in rows if r[f"{label}_y512_seqmean"] is not None]))
                        if y512_all else None,
        "n_sequences": len(rows),
        "n_frames": len(rgb256_all),
    }
    return {"summary": summary, "per_seq": rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/audit_v9")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--clip_len", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[audit] device: {device}")

    seqs = list_sdsd_sequences(args.val_root, subset="direct", split="test")
    print(f"[audit] val sequences: {len(seqs)}")

    targets = [
        ("LQ_baseline", None, None),
        ("v9.0_best", "configs/train_v9.yaml", "/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt"),
        ("v9.0_best_ema", "configs/train_v9.yaml", "/cache/zihaohe/checkpoint/runs/v9/ckpts/best_ema.pt"),
        ("v9.1a_best", "configs/train_v9_1.yaml", "/cache/zihaohe/checkpoint/runs/v9_1/ckpts/best.pt"),
    ]

    summaries = []
    per_seq_combined = defaultdict(dict)

    for label, cfg_path, ckpt_path in targets:
        print(f"\n[audit] === {label} ===")
        t0 = time.time()
        if cfg_path is None:
            model = None
        else:
            model = build_v9_model(cfg_path, ckpt_path, device)
        result = eval_model(model, seqs, device, clip_len=args.clip_len, label=label)
        s = result["summary"]
        print(f"  RGB-256 seq-mean: {s['rgb256_seqmean']:.3f}  Y-256 seq-mean: {s['y256_seqmean']:.3f}  "
              f"SSIM-256: {s['ssim256_seqmean']:.4f}")
        if s['rgb512_seqmean'] is not None:
            print(f"  RGB-512 seq-mean: {s['rgb512_seqmean']:.3f}  Y-512 seq-mean: {s['y512_seqmean']:.3f}")
        else:
            print(f"  RGB-512: skipped (frames < 512)")
        print(f"  time: {time.time()-t0:.1f}s")
        summaries.append(s)
        for row in result["per_seq"]:
            seq = row["seq"]
            for k, v in row.items():
                per_seq_combined[seq][k] = v
        if model is not None:
            del model
            torch.cuda.empty_cache()

    # === write summary CSV ===
    summary_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    with open(summary_csv, "w") as f:
        keys = ["model", "rgb256_seqmean", "y256_seqmean", "rgb512_seqmean", "y512_seqmean",
                "rgb256_framemean", "y256_framemean", "rgb512_framemean", "y512_framemean",
                "ssim256_seqmean", "n_sequences", "n_frames"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k) for k in keys})
    print(f"\n[audit] wrote {summary_csv}")

    # === write summary MD table ===
    md_path = os.path.join(args.out_dir, "metrics_summary.md")
    with open(md_path, "w") as f:
        f.write("# v9 Protocol Audit Summary\n\n")
        f.write(f"Val: SDSD-indoor test, {summaries[0]['n_sequences']} sequences, "
                f"{summaries[0]['n_frames']} frames per metric.\n\n")
        f.write("| Model | RGB-256 (seq) | Y-256 (seq) | RGB-512 (seq) | Y-512 (seq) | RGB-256 (frame) | Y-256 (frame) | SSIM-256 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for s in summaries:
            def fmt(v):
                return f"{v:.3f}" if v is not None else "—"
            f.write(f"| {s['model']} | {fmt(s['rgb256_seqmean'])} | {fmt(s['y256_seqmean'])} | "
                    f"{fmt(s['rgb512_seqmean'])} | {fmt(s['y512_seqmean'])} | "
                    f"{fmt(s['rgb256_framemean'])} | {fmt(s['y256_framemean'])} | "
                    f"{fmt(s['ssim256_seqmean'])} |\n")
    print(f"[audit] wrote {md_path}")

    # === write per-sequence CSV ===
    seq_csv = os.path.join(args.out_dir, "per_sequence_metrics.csv")
    all_keys = set()
    for d in per_seq_combined.values():
        all_keys.update(d.keys())
    keys = ["seq", "frames", "HxW_full"] + sorted(k for k in all_keys if k not in ("seq", "frames", "HxW_full"))
    with open(seq_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for seq in sorted(per_seq_combined):
            w.writerow({k: per_seq_combined[seq].get(k) for k in keys})
    print(f"[audit] wrote {seq_csv}")
    print(f"\n[audit] DONE")


if __name__ == "__main__":
    main()
