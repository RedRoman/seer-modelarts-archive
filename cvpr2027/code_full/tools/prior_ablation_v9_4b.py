"""tools/prior_ablation_v9_4b.py — BE / BF v9.4b verification.

Center-crop 256x256 inference (NO tiling, NO seam) on the same 14 SDSD-indoor
static_np test seqs, three modes:
  full_vjepa
  cross_attn_off  (forward_hook bypass on every VJepaCrossAttnBF)
  zero_vjepa      (replace e_lq_layers with zeros)

Purpose: distinguish whether v9.4's +1.06 dB cross_attn_off improvement is
- (Case A) general, also visible without tiling, OR
- (Case B) a tiled-deployment artifact specific to per-tile V-JEPA recompute.

Output: /cache/zihaohe/checkpoint/prior_ablation_v9_4b/
        ablation_summary.md, per_seq.csv, run.log
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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import (
    load_seq_full, build_v9_model, rgb_psnr_uint8,
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
    raise ValueError(f"unknown mode {mode}")


@torch.no_grad()
def center_crop_inference(model, video_lq: torch.Tensor, mode: str,
                           crop: int = 256, autocast_dtype=torch.bfloat16) -> torch.Tensor:
    """Single-pass forward on a center 256x256 crop (no tiling)."""
    B, C, T, H, W = video_lq.shape
    h0 = (H - crop) // 2
    w0 = (W - crop) // 2
    patch = video_lq[..., h0:h0 + crop, w0:w0 + crop]
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        feats = model.encode_video_features(patch)
        feats = manipulate_features(feats, mode)
        feats = [f.to(patch.dtype) for f in feats]
        out = model.pixel_branch(patch, feats)
    return out["pred_pixels"].clamp(0, 1), (h0, w0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/prior_ablation_v9_4b")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt")
    ap.add_argument("--cfg", default="configs/train_v9.yaml")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v9.4b] device: {device} crop={args.crop}")

    seqs = list_seqs(args.val_root, "test")
    print(f"[v9.4b] {len(seqs)} indoor test seqs")

    model = build_v9_model(args.cfg, args.ckpt, device)

    modes = ["full_vjepa", "cross_attn_off", "zero_vjepa"]
    per_seq_results: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        print(f"\n[v9.4b] === {mode} ===")
        hook_handles = install_cross_attn_off(model) if mode == "cross_attn_off" else []
        try:
            for seq_name, lq_paths, gt_paths in seqs:
                t0 = time.time()
                lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=args.n_frames)
                T_used = lq.shape[2]
                lq, gt = lq.to(device), gt.to(device)

                pred, (h0, w0) = center_crop_inference(model, lq, mode=mode, crop=args.crop)
                gt_crop = gt[..., h0:h0 + args.crop, w0:w0 + args.crop]

                pred_uint8 = (pred[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                gt_uint8 = (gt_crop[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                psnrs = [rgb_psnr_uint8(pred_uint8[t], gt_uint8[t]) for t in range(T_used)]
                seq_psnr = float(np.mean(psnrs))
                per_seq_results.setdefault(seq_name, {})[mode] = seq_psnr
                print(f"  [{mode}/{seq_name}] PSNR={seq_psnr:.3f} ({time.time()-t0:.1f}s)")
        finally:
            uninstall_hooks(hook_handles)

    # Aggregate
    aggregate = {m: {"mean": float(np.mean([per_seq_results[s][m] for s in per_seq_results
                                             if m in per_seq_results[s]])),
                     "n": sum(1 for s in per_seq_results if m in per_seq_results[s])}
                 for m in modes}

    baseline = aggregate["full_vjepa"]["mean"]
    deltas = {m: aggregate[m]["mean"] - baseline for m in modes}

    # SIGNED-aware classification + verdict (BE0.1 fix)
    def cls(d):
        ad = abs(d)
        if ad < 0.2: return "no dep"
        elif ad < 0.5: return "weak"
        else: return "meaningful"

    d_off = deltas["cross_attn_off"]
    d_zero = deltas["zero_vjepa"]
    cls_off, cls_zero = cls(d_off), cls(d_zero)

    # Compare against v9.4 (full-frame tiled) result
    V94_OFF_DELTA_TILED = +1.059  # from /cache/zihaohe/checkpoint/prior_ablation_v9_4/

    # BE3 case mapping (signed)
    if cls_off == "meaningful" and d_off > 0:
        if abs(d_off - V94_OFF_DELTA_TILED) < 0.5:
            verdict = (f"Case A (BE3): cross_attn_off improves under center-crop too (Δ={d_off:+.3f}). "
                       f"v9.4's +{V94_OFF_DELTA_TILED:.3f} tiled improvement is GENERAL, not a tile artifact. "
                       "Cross-attn residual is broadly net-negative at inference. "
                       "Next: v9.5-gated training (per BE4).")
        else:
            verdict = (f"Case A-partial: cross_attn_off improves under center-crop (Δ={d_off:+.3f}) "
                       f"but magnitude differs from tiled (+{V94_OFF_DELTA_TILED:.3f}). "
                       "Effect is real but partly protocol-dependent. Inspect per-seq.")
    elif cls_off in ("no dep", "weak"):
        verdict = (f"Case B (BE3): cross_attn_off does NOT meaningfully improve under center-crop "
                   f"(Δ={d_off:+.3f}, |Δ|<0.5 dB). v9.4's +{V94_OFF_DELTA_TILED:.3f} tiled improvement "
                   "is likely a TILED-DEPLOYMENT ARTIFACT — V-JEPA features per-tile are inconsistent. "
                   "Next: fix tile-consistent V-JEPA inference, NOT v10 architecture rewrite.")
    elif cls_off == "meaningful" and d_off < 0:
        verdict = (f"Case Reversed: cross_attn_off HURTS under center-crop (Δ={d_off:+.3f}). "
                   f"v9.4's +{V94_OFF_DELTA_TILED:.3f} tiled improvement is a tile-only artifact AND "
                   "cross-attn is positively useful in non-tiled mode. Strong evidence for tile-consistency fix.")
    else:
        verdict = f"Mixed: d_off={d_off:+.3f} cls={cls_off}. See per-seq."

    # === Write per-seq CSV ===
    csv_path = os.path.join(args.out_dir, "per_seq.csv")
    keys = ["seq"] + modes + ["delta_cross_attn_off", "delta_zero_vjepa"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for seq in sorted(per_seq_results):
            d = per_seq_results[seq]
            row = {"seq": seq}
            for m in modes: row[m] = d.get(m)
            row["delta_cross_attn_off"] = d.get("cross_attn_off", 0) - d.get("full_vjepa", 0)
            row["delta_zero_vjepa"] = d.get("zero_vjepa", 0) - d.get("full_vjepa", 0)
            w.writerow(row)
    print(f"\nwrote {csv_path}")

    # === Write summary MD ===
    md_path = os.path.join(args.out_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("# BE/BF v9.4b-crossattn-off-verification Summary\n\n")
        f.write(f"Center-crop {args.crop}x{args.crop} (NO tiling) inference on v9.0_best.\n")
        f.write("Eval: 14 SDSD-indoor static_np test seqs, RGB-PSNR uint8 on the center crop.\n\n")
        f.write("## Aggregate per-mode\n\n")
        f.write("| Mode | RGB-PSNR (center-crop) | Δ vs full | Class |\n")
        f.write("|---|---:|---:|:---:|\n")
        for m in modes:
            d = deltas[m]
            f.write(f"| {m} | {aggregate[m]['mean']:.3f} | {d:+.3f} | "
                    f"{cls(d) if m != 'full_vjepa' else '—'} |\n")

        f.write(f"\n**v9.4 reference (full-frame tiled, indoor 14 seqs):**\n")
        f.write(f"  cross_attn_off Δ vs full = +{V94_OFF_DELTA_TILED:.3f} dB.\n\n")

        f.write("## BE3 verdict\n\n")
        f.write(f"**{verdict}**\n\n")

        f.write("## Per-seq detail (center-crop)\n\n")
        f.write("| seq | full | cross_attn_off (Δ) | zero (Δ) |\n")
        f.write("|---|---:|---:|---:|\n")
        for seq in sorted(per_seq_results):
            d = per_seq_results[seq]
            base = d.get("full_vjepa", 0)
            f.write(f"| {seq} | {base:.3f} | "
                    f"{d.get('cross_attn_off', 0):.3f} ({d.get('cross_attn_off', 0) - base:+.3f}) | "
                    f"{d.get('zero_vjepa', 0):.3f} ({d.get('zero_vjepa', 0) - base:+.3f}) |\n")
    print(f"wrote {md_path}")
    print(f"\n=== AGGREGATE ===")
    for m in modes:
        print(f"  {m:25s} PSNR={aggregate[m]['mean']:.3f}  Δ={deltas[m]:+.3f}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
