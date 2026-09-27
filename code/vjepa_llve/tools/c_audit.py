"""tools/c_audit.py — BT2/BU3 C-audit of θ Step 1 result.

Goal: rule out protocol/eval mistakes behind the 21.80 dB number on 6
LLVE-STCD seqs (pair13/15/21/23/31/33).

Five+2 sub-checks (BT2 + BU3):

  1. Frame offset PSNR + SSIM check at {-2,-1,0,+1,+2}.
     If non-zero offset peaks, frame numbering misaligned.
     BU3.1: add SSIM as alignment proxy (low-light brightness fools raw PSNR).

  2. Visual panels (LQ / pred / GT / abs-error) for pair31 + pair23.

  3. tile512 vs 512 center-crop direct on pair31 + pair23.
     If full-frame much worse than crop/direct, tiling still an issue.

  4. Train/test membership verify.
     BU3.2: log if each seq was in train set.

  5. Per-seq SSIM as sanity vs PSNR.

Output: /cache/zihaohe/checkpoint/c_audit/{c_audit_summary.md, per_seq_metrics.csv, visual_panels/}
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
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import build_v9_model, rgb_psnr_uint8, make_taper_window
from tools.theta_llvestcd_eval import _load_one, list_frames, tiled_inference, to_u8, mean_psnr

LLVE_STCD_TEST_SEQS = ["pair13", "pair15", "pair21", "pair23", "pair31", "pair33"]


def ssim_simple(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    """11x11 Gaussian SSIM per-channel mean. Match audit_llvestcd.py convention."""
    from skimage.metrics import structural_similarity as ssim
    if a_u8.ndim == 3:
        return float(ssim(a_u8, b_u8, channel_axis=-1, data_range=255, win_size=11,
                          gaussian_weights=True, sigma=1.5))
    return float(ssim(a_u8, b_u8, data_range=255))


def edge_ncc(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    """Edge-map normalized cross correlation (alignment-sensitive, light-invariant)."""
    from scipy import ndimage
    a_g = a_u8.astype(np.float32).mean(axis=-1) if a_u8.ndim == 3 else a_u8.astype(np.float32)
    b_g = b_u8.astype(np.float32).mean(axis=-1) if b_u8.ndim == 3 else b_u8.astype(np.float32)
    a_e = ndimage.sobel(a_g, axis=0)**2 + ndimage.sobel(a_g, axis=1)**2
    b_e = ndimage.sobel(b_g, axis=0)**2 + ndimage.sobel(b_g, axis=1)**2
    a_e = (a_e - a_e.mean()) / (a_e.std() + 1e-8)
    b_e = (b_e - b_e.mean()) / (b_e.std() + 1e-8)
    return float((a_e * b_e).mean())


def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match src histogram to ref per channel."""
    matched = np.empty_like(src)
    for c in range(src.shape[-1]):
        s, r = src[..., c].ravel(), ref[..., c].ravel()
        s_v, s_idx, s_cnt = np.unique(s, return_inverse=True, return_counts=True)
        r_v, r_cnt = np.unique(r, return_counts=True)
        s_q = np.cumsum(s_cnt).astype(np.float64) / s.size
        r_q = np.cumsum(r_cnt).astype(np.float64) / r.size
        interp = np.interp(s_q, r_q, r_v.astype(np.float64))
        matched[..., c] = interp[s_idx].reshape(src[..., c].shape)
    return np.clip(matched, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Check 1 — frame offset PSNR + SSIM + edge-NCC + histogram-matched
# ---------------------------------------------------------------------------
def check1_frame_offset(in_dir, gt_dir, n_eval=6, offsets=(-2, -1, 0, 1, 2)):
    lq_paths = list_frames(in_dir)
    gt_paths = list_frames(gt_dir)
    by_lq = {os.path.basename(p): p for p in lq_paths}
    by_gt = {os.path.basename(p): p for p in gt_paths}
    common = sorted(set(by_lq) & set(by_gt))
    if len(common) < n_eval + 4:
        n_eval = max(1, len(common) - 4)

    rows = []
    for off in offsets:
        psnrs, ssims, nccs, hist_psnrs = [], [], [], []
        for i in range(2, 2 + n_eval):  # use mid frames to allow ±2 offset
            j = i + off
            if j < 0 or j >= len(common): continue
            lq_name = common[i]
            gt_name = common[j]
            lq = _load_one(by_lq[lq_name])
            gt = _load_one(by_gt[gt_name])
            psnrs.append(rgb_psnr_uint8(lq, gt))
            ssims.append(ssim_simple(lq, gt))
            nccs.append(edge_ncc(lq, gt))
            lq_hm = histogram_match(lq, gt)
            hist_psnrs.append(rgb_psnr_uint8(lq_hm, gt))
        if psnrs:
            rows.append({
                "offset": off,
                "n": len(psnrs),
                "psnr_mean": float(np.mean(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "edge_ncc_mean": float(np.mean(nccs)),
                "psnr_histmatch_mean": float(np.mean(hist_psnrs)),
            })
    return rows


# ---------------------------------------------------------------------------
# Check 3 — tile=512 full-frame vs 512 direct center-crop
# ---------------------------------------------------------------------------
@torch.no_grad()
def direct_inference_512(model, video_lq, autocast_dtype=torch.bfloat16):
    """Single-pass 512x512 center-crop inference."""
    H, W = video_lq.shape[-2:]
    h0 = (H - 512) // 2
    w0 = (W - 512) // 2
    patch = video_lq[..., h0:h0+512, w0:w0+512]
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        out = model.sample(patch)
    return out["pred_pixels"].clamp(0, 1), (h0, w0)


# ---------------------------------------------------------------------------
# Visual panel save
# ---------------------------------------------------------------------------
def save_visual_panel(lq, pred, gt, out_path: str, n_frames=4):
    """Save a panel: rows=frames, cols=[LQ, pred, GT, abs(pred-gt)]."""
    panels = []
    for t in range(min(n_frames, lq.shape[0])):
        lq_f = lq[t]; pr_f = pred[t]; gt_f = gt[t]
        err_f = np.abs(pr_f.astype(np.int16) - gt_f.astype(np.int16)).clip(0, 255).astype(np.uint8)
        # Upscale error 4x for visibility
        err_f = np.minimum(err_f * 4, 255).astype(np.uint8)
        row = np.concatenate([lq_f, pr_f, gt_f, err_f], axis=1)
        panels.append(row)
    img = np.concatenate(panels, axis=0)
    imageio.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/c_audit")
    ap.add_argument("--data_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor/indoor_static_np")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt")
    ap.add_argument("--cfg",  default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--n_frames", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "visual_panels"), exist_ok=True)
    device = torch.device("cuda")
    print(f"[C-audit] device: {device}")
    print(f"[C-audit] ckpt: {args.ckpt}")

    # =========================================================================
    # Check 4: train/test membership verify (BU3.2)
    # =========================================================================
    print("\n=== Check 4 — train/test membership ===")
    ds_train = SDSDDataset(root="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor",
                            subset="direct", split="train", clip_len=16, crop_size=256, is_train=False)
    ds_test = SDSDDataset(root="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor",
                           subset="direct", split="test",  clip_len=16, crop_size=256, is_train=False)
    train_seqs = {s[0] for s in ds_train.seqs}
    test_seqs = {s[0] for s in ds_test.seqs}
    membership = {}
    for seq in LLVE_STCD_TEST_SEQS:
        loc = "train" if seq in train_seqs else ("test" if seq in test_seqs else "neither")
        membership[seq] = loc
        print(f"  {seq}: {loc}")

    # =========================================================================
    # Check 1: frame offset (PSNR + SSIM + edge-NCC + histogram-matched)
    # =========================================================================
    print("\n=== Check 1 — frame offset PSNR/SSIM/edge-NCC/hist-matched ===")
    offset_results = {}
    for seq in LLVE_STCD_TEST_SEQS:
        in_d = os.path.join(args.data_root, "input", seq)
        gt_d = os.path.join(args.data_root, "GT", seq)
        if not (os.path.isdir(in_d) and os.path.isdir(gt_d)):
            print(f"  [skip] {seq}: dirs missing"); continue
        rows = check1_frame_offset(in_d, gt_d)
        offset_results[seq] = rows
        peak_off_psnr = max(rows, key=lambda r: r["psnr_mean"])["offset"]
        peak_off_ssim = max(rows, key=lambda r: r["ssim_mean"])["offset"]
        peak_off_ncc = max(rows, key=lambda r: r["edge_ncc_mean"])["offset"]
        peak_off_hm = max(rows, key=lambda r: r["psnr_histmatch_mean"])["offset"]
        print(f"  [{seq}] peaks: PSNR@{peak_off_psnr}, SSIM@{peak_off_ssim}, "
              f"edge-NCC@{peak_off_ncc}, hist-PSNR@{peak_off_hm}")
        for r in rows:
            print(f"     off={r['offset']:+d} n={r['n']} PSNR={r['psnr_mean']:.3f}  "
                  f"SSIM={r['ssim_mean']:.4f}  edge-NCC={r['edge_ncc_mean']:.4f}  "
                  f"hist-PSNR={r['psnr_histmatch_mean']:.3f}")

    # =========================================================================
    # Check 2 + 3 + 5: model inference + visual + tile vs direct + SSIM
    # =========================================================================
    print("\n=== Check 2/3/5 — model inference + visual + tile vs direct + SSIM ===")
    model = build_v9_model(args.cfg, args.ckpt, device)

    from tools.theta_llvestcd_eval import load_seq

    seq_metrics = {}
    for seq in LLVE_STCD_TEST_SEQS:
        in_d = os.path.join(args.data_root, "input", seq)
        gt_d = os.path.join(args.data_root, "GT", seq)
        if not (os.path.isdir(in_d) and os.path.isdir(gt_d)): continue
        print(f"\n  --- {seq} ---")
        lq, gt, T = load_seq(in_d, gt_d, n_frames=args.n_frames)
        H, W = lq.shape[-2:]
        lq_dev = lq.to(device); gt_dev = gt.to(device)

        # tile=512 (matches θ result)
        t0 = time.time()
        pred_t512 = tiled_inference(model, lq_dev, tile=512, overlap=128)
        psnr_t512 = mean_psnr(to_u8(pred_t512), to_u8(gt_dev))
        ssim_t512 = ssim_simple(to_u8(pred_t512)[0], to_u8(gt_dev)[0])  # first frame
        print(f"    tile=512:  PSNR={psnr_t512:.3f} SSIM={ssim_t512:.4f} ({time.time()-t0:.1f}s)")

        # 512 direct center-crop (Check 3)
        h0 = (H - 512) // 2; w0 = (W - 512) // 2
        gt_crop = gt_dev[..., h0:h0+512, w0:w0+512]
        t0 = time.time()
        pred_direct, _ = direct_inference_512(model, lq_dev)
        psnr_direct = mean_psnr(to_u8(pred_direct), to_u8(gt_crop))
        print(f"    direct 512 center-crop: PSNR={psnr_direct:.3f} ({time.time()-t0:.1f}s)")
        # Also compute tile512 on the same center-crop for direct comparison
        pred_t512_crop = pred_t512[..., h0:h0+512, w0:w0+512]
        psnr_t512_centercrop = mean_psnr(to_u8(pred_t512_crop), to_u8(gt_crop))

        seq_metrics[seq] = {
            "T": T, "H": H, "W": W,
            "membership": membership.get(seq, "?"),
            "psnr_t512_full": psnr_t512,
            "ssim_t512_first": ssim_t512,
            "psnr_t512_centercrop": psnr_t512_centercrop,
            "psnr_direct_centercrop": psnr_direct,
            "delta_direct_minus_t512crop": psnr_direct - psnr_t512_centercrop,
        }

        # Visual panels (Check 2) for pair31, pair23
        if seq in ("pair31", "pair23"):
            panel_path = os.path.join(args.out_dir, "visual_panels", f"{seq}_lq_pred_gt_err.png")
            save_visual_panel(to_u8(lq_dev), to_u8(pred_t512), to_u8(gt_dev), panel_path)
            print(f"    visual panel saved: {panel_path}")

    # =========================================================================
    # Write outputs
    # =========================================================================
    csv_path = os.path.join(args.out_dir, "per_seq_metrics.csv")
    with open(csv_path, "w") as f:
        keys = ["seq", "membership", "T", "H", "W",
                 "psnr_t512_full", "ssim_t512_first",
                 "psnr_t512_centercrop", "psnr_direct_centercrop", "delta_direct_minus_t512crop"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for seq, m in seq_metrics.items():
            row = {"seq": seq}; row.update(m); w.writerow(row)
    print(f"\nwrote {csv_path}")

    md_path = os.path.join(args.out_dir, "c_audit_summary.md")
    with open(md_path, "w") as f:
        f.write("# C-audit: rule out protocol/eval mistakes behind 21.80 dB θ Step 1 number\n\n")
        f.write(f"**Checkpoint**: v9.0b_mix_best  \n")
        f.write(f"**Eval seqs**: 6 LLVE-STCD test seqs available locally  \n")
        f.write(f"**Reference (θ Step 1, BQ1)**: 21.798 dB mean v9.0b_mix tile=512\n\n")

        f.write("## Check 4 — Train/test membership (BU3.2)\n\n")
        f.write("| seq | membership |\n|---|---|\n")
        for seq in LLVE_STCD_TEST_SEQS:
            f.write(f"| {seq} | {membership.get(seq, '?')} |\n")
        f.write("\n")

        f.write("## Check 1 — Frame offset alignment (BU3.1)\n\n")
        f.write("Per-seq offset peaks for 4 metrics (offset where each metric is max):\n\n")
        f.write("| seq | PSNR peak | SSIM peak | edge-NCC peak | hist-PSNR peak | misalign? |\n")
        f.write("|---|---:|---:|---:|---:|:---:|\n")
        for seq, rows in offset_results.items():
            p_psnr = max(rows, key=lambda r: r["psnr_mean"])["offset"]
            p_ssim = max(rows, key=lambda r: r["ssim_mean"])["offset"]
            p_ncc = max(rows, key=lambda r: r["edge_ncc_mean"])["offset"]
            p_hm = max(rows, key=lambda r: r["psnr_histmatch_mean"])["offset"]
            misalign = "✓ yes" if (p_ncc != 0 or p_ssim != 0) else "no"
            f.write(f"| {seq} | {p_psnr:+d} | {p_ssim:+d} | {p_ncc:+d} | {p_hm:+d} | {misalign} |\n")
        f.write("\nFull per-offset metrics:\n\n")
        for seq, rows in offset_results.items():
            f.write(f"### {seq}\n\n")
            f.write("| offset | n | PSNR | SSIM | edge-NCC | hist-PSNR |\n|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(f"| {r['offset']:+d} | {r['n']} | {r['psnr_mean']:.3f} | "
                        f"{r['ssim_mean']:.4f} | {r['edge_ncc_mean']:.4f} | "
                        f"{r['psnr_histmatch_mean']:.3f} |\n")
            f.write("\n")

        f.write("## Check 2/3/5 — Model inference + tile vs direct + SSIM\n\n")
        f.write("| seq | membership | T |  psnr_t512_full | ssim_t512 |  psnr_t512_centercrop |  psnr_direct_centercrop | Δ direct-tile (centercrop) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for seq, m in seq_metrics.items():
            f.write(f"| {seq} | {m['membership']} | {m['T']} | {m['psnr_t512_full']:.3f} | "
                    f"{m['ssim_t512_first']:.4f} | {m['psnr_t512_centercrop']:.3f} | "
                    f"{m['psnr_direct_centercrop']:.3f} | {m['delta_direct_minus_t512crop']:+.3f} |\n")

        f.write("\nVisual panels for worst seqs (pair31, pair23): see `visual_panels/`\n\n")

        # Verdict
        f.write("## Verdict\n\n")
        any_misalign = any(
            (max(rows, key=lambda r: r["edge_ncc_mean"])["offset"] != 0
             or max(rows, key=lambda r: r["ssim_mean"])["offset"] != 0)
            for rows in offset_results.values()
        )
        any_tile_artifact = any(
            abs(m["delta_direct_minus_t512crop"]) > 0.5 for m in seq_metrics.values()
        )
        all_in_train = all(membership.get(s) == "train" for s in LLVE_STCD_TEST_SEQS)
        verdict_lines = []
        if any_misalign:
            verdict_lines.append("PROTOCOL ISSUE FOUND: frame misalignment per Check 1.")
        if any_tile_artifact:
            verdict_lines.append("PROTOCOL ISSUE FOUND: residual tile artifact per Check 3.")
        if not any_misalign and not any_tile_artifact:
            verdict_lines.append("No protocol issue found. The 21.80 dB θ Step 1 result is real.")
            verdict_lines.append("Proceed with BT-1.5w-gate (BVI-RLV viability check + v10 design).")
        if all_in_train:
            verdict_lines.append("All 6 LLVE-STCD seqs are in our TRAINING set. The 21.8 dB number "
                                 "is therefore a train-time underfit number, NOT a held-out test number. "
                                 "Strong evidence for architecture/data weakness, not generalization gap.")
        for v in verdict_lines:
            f.write(f"- {v}\n")

    print(f"wrote {md_path}")
    for v in verdict_lines:
        print(f"VERDICT: {v}")


if __name__ == "__main__":
    main()
