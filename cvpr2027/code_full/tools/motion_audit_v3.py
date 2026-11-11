"""tools/motion_audit_v3.py — AP6 Test A (motion-bucket indoor) + Test B (outdoor audit).

Per AO4/AO7/AP6 spec:
  Test A: per-sequence motion score (LQ frame-diff) on 14 indoor test seqs;
          bucket into static/mild/dynamic (5/4/5);
          report RGB-PSNR per bucket for LQ + v9.0_best + v9.0-noJEPA_best;
          compute Δ(v9.0 - noJEPA) per bucket → does V-JEPA gain grow with motion?

  Test B: same RGB-full tiled audit on SDSD-outdoor 25 test seqs;
          report RGB-PSNR per ckpt + per-seq motion score;
          comparison vs Test A indoor numbers.

Re-uses tiled_inference, build_v9_model, load_seq_full, rgb_psnr_uint8 from audit_llvestcd.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import (
    load_seq_full, tiled_inference, build_v9_model, rgb_psnr_uint8,
)


def list_seqs(root: str, split: str = "test"):
    ds = SDSDDataset(root=root, subset="direct", split=split,
                     clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


@torch.no_grad()
def motion_score(lq_paths, gt_paths, n_frames: int = 16) -> Dict[str, float]:
    """Mean abs diff between adjacent LQ frames (and HQ for sanity), 256 center crop.

    Cheap proxy for motion. Higher = more motion.
    """
    lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
    # Crop to center 256 for cheap motion score (avoids resolution bias)
    H, W = lq.shape[-2:]
    crop = min(256, H, W)
    h0 = (H - crop) // 2
    w0 = (W - crop) // 2
    lq_c = lq[..., h0:h0 + crop, w0:w0 + crop]
    gt_c = gt[..., h0:h0 + crop, w0:w0 + crop]
    # adjacent diff over T axis
    lq_diff = (lq_c[:, :, 1:] - lq_c[:, :, :-1]).abs().mean().item()
    gt_diff = (gt_c[:, :, 1:] - gt_c[:, :, :-1]).abs().mean().item()
    return {"motion_lq": lq_diff, "motion_hq": gt_diff}


@torch.no_grad()
def run_audit_per_seq(model, seqs, device, n_frames=16, tile=256, overlap=64,
                     label="") -> List[Dict]:
    rows = []
    for seq_name, lq_paths, gt_paths in seqs:
        t0 = time.time()
        lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
        T_used = lq.shape[2]
        H, W = lq.shape[-2:]
        lq, gt = lq.to(device), gt.to(device)
        pred = tiled_inference(model, lq, tile=tile, overlap=overlap)
        # RGB PSNR uint8 [0,255] per-frame, then per-seq mean
        pred_uint8 = (pred[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_uint8 = (gt[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        psnrs = [rgb_psnr_uint8(pred_uint8[t], gt_uint8[t]) for t in range(T_used)]
        seq_psnr = float(np.mean(psnrs))
        rows.append({
            "seq": seq_name, "frames": T_used, "HxW": f"{H}x{W}",
            f"{label}_rgb_psnr": seq_psnr,
            f"{label}_time_s": round(time.time() - t0, 1),
        })
        print(f"  [{label}/{seq_name}] {T_used}f @ {H}x{W} | RGB-PSNR={seq_psnr:.3f} | {time.time()-t0:.1f}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/motion_audit_v3")
    ap.add_argument("--indoor_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--outdoor_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_outdoor")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[motion_audit] device: {device}")

    indoor_seqs = list_seqs(args.indoor_root, "test")
    outdoor_seqs = list_seqs(args.outdoor_root, "test")
    print(f"[motion_audit] indoor: {len(indoor_seqs)} seqs, outdoor: {len(outdoor_seqs)} seqs")

    # === Phase 0: motion scores (no model, cheap) ===
    print("\n[motion_audit] === Phase 0: motion scores ===")
    motion = {"indoor": {}, "outdoor": {}}
    for tag, seqs in (("indoor", indoor_seqs), ("outdoor", outdoor_seqs)):
        for seq_name, lq_paths, gt_paths in seqs:
            ms = motion_score(lq_paths, gt_paths, n_frames=16)
            motion[tag][seq_name] = ms
            print(f"  [{tag}/{seq_name}] motion_lq={ms['motion_lq']:.4f} motion_hq={ms['motion_hq']:.4f}")

    # === Phase 1: indoor audit on LQ + v9.0_best + v9.0-noJEPA_best ===
    print("\n[motion_audit] === Phase 1: indoor audit ===")
    indoor_results = defaultdict(dict)
    targets = [
        ("LQ", None, None),
        ("v9.0_best", "configs/train_v9.yaml", "/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt"),
        ("v9.0-noJEPA_best", "configs/train_v9_noJEPA.yaml", "/cache/zihaohe/checkpoint/runs/v9_noJEPA/ckpts/best.pt"),
    ]
    for label, cfg_path, ckpt_path in targets:
        print(f"\n[indoor/{label}]")
        if cfg_path is None:
            model = None
        else:
            model = build_v9_model(cfg_path, ckpt_path, device)
        rows = run_audit_per_seq(model, indoor_seqs, device, label=label)
        for row in rows:
            indoor_results[row["seq"]][f"{label}_rgb_psnr"] = row[f"{label}_rgb_psnr"]
            indoor_results[row["seq"]]["frames"] = row["frames"]
            indoor_results[row["seq"]]["HxW"] = row["HxW"]
        if model is not None:
            del model
            torch.cuda.empty_cache()

    # === Phase 2: outdoor audit on same 3 ckpts ===
    print("\n[motion_audit] === Phase 2: outdoor audit ===")
    outdoor_results = defaultdict(dict)
    for label, cfg_path, ckpt_path in targets:
        print(f"\n[outdoor/{label}]")
        if cfg_path is None:
            model = None
        else:
            model = build_v9_model(cfg_path, ckpt_path, device)
        rows = run_audit_per_seq(model, outdoor_seqs, device, label=label)
        for row in rows:
            outdoor_results[row["seq"]][f"{label}_rgb_psnr"] = row[f"{label}_rgb_psnr"]
            outdoor_results[row["seq"]]["frames"] = row["frames"]
            outdoor_results[row["seq"]]["HxW"] = row["HxW"]
        if model is not None:
            del model
            torch.cuda.empty_cache()

    # === Phase 3: bucket indoor by motion score ===
    print("\n[motion_audit] === Phase 3: indoor motion bucketing ===")
    indoor_seq_names = sorted(motion["indoor"].keys(), key=lambda n: motion["indoor"][n]["motion_lq"])
    n = len(indoor_seq_names)
    third = n // 3
    bucket = {}
    for i, name in enumerate(indoor_seq_names):
        if i < third + (1 if (n % 3) >= 2 else 0):
            bucket[name] = "static"
        elif i < 2 * third + (1 if (n % 3) >= 1 else 0):
            bucket[name] = "mild"
        else:
            bucket[name] = "dynamic"
    print("Bucket assignment by motion_lq score (low → high):")
    for name in indoor_seq_names:
        print(f"  {name:12s} motion_lq={motion['indoor'][name]['motion_lq']:.4f}  {bucket[name]}")

    # === Phase 4: per-bucket aggregates ===
    print("\n[motion_audit] === Phase 4: indoor per-bucket aggregates ===")
    bucket_agg = defaultdict(lambda: defaultdict(list))
    for name in indoor_seq_names:
        b = bucket[name]
        for label, _, _ in targets:
            psnr_v = indoor_results[name].get(f"{label}_rgb_psnr")
            if psnr_v is not None:
                bucket_agg[b][label].append(psnr_v)
    bucket_summary = {}
    for b in ("static", "mild", "dynamic"):
        bucket_summary[b] = {
            "n_seqs": len(bucket_agg[b]["LQ"]) if "LQ" in bucket_agg[b] else 0,
            "LQ":              float(np.mean(bucket_agg[b]["LQ"])) if bucket_agg[b]["LQ"] else None,
            "v9.0":            float(np.mean(bucket_agg[b]["v9.0_best"])) if bucket_agg[b]["v9.0_best"] else None,
            "noJEPA":          float(np.mean(bucket_agg[b]["v9.0-noJEPA_best"])) if bucket_agg[b]["v9.0-noJEPA_best"] else None,
        }
        bs = bucket_summary[b]
        if bs["v9.0"] is not None and bs["noJEPA"] is not None:
            bs["delta_v9.0_minus_noJEPA"] = bs["v9.0"] - bs["noJEPA"]
        if bs["v9.0"] is not None and bs["LQ"] is not None:
            bs["lift_v9.0"] = bs["v9.0"] - bs["LQ"]
        if bs["noJEPA"] is not None and bs["LQ"] is not None:
            bs["lift_noJEPA"] = bs["noJEPA"] - bs["LQ"]
        print(f"  {b:8s} n={bs['n_seqs']}  LQ={bs['LQ']:.3f}  v9.0={bs['v9.0']:.3f}  "
              f"noJEPA={bs['noJEPA']:.3f}  Δ(v9.0-noJEPA)={bs.get('delta_v9.0_minus_noJEPA', 0):.3f}")

    # === Phase 5: outdoor aggregate ===
    print("\n[motion_audit] === Phase 5: outdoor aggregate ===")
    outdoor_agg = {}
    motion_outdoor_avg = float(np.mean([motion["outdoor"][n]["motion_lq"] for n in motion["outdoor"]]))
    motion_indoor_avg = float(np.mean([motion["indoor"][n]["motion_lq"] for n in motion["indoor"]]))
    print(f"  Avg motion_lq: indoor={motion_indoor_avg:.4f}  outdoor={motion_outdoor_avg:.4f}")
    for label, _, _ in targets:
        vals = [outdoor_results[name].get(f"{label}_rgb_psnr") for name in outdoor_results
                if outdoor_results[name].get(f"{label}_rgb_psnr") is not None]
        outdoor_agg[label] = float(np.mean(vals)) if vals else None
        print(f"  outdoor {label}: RGB-PSNR mean={outdoor_agg[label]:.3f} (n={len(vals)})")
    if outdoor_agg.get("v9.0_best") and outdoor_agg.get("v9.0-noJEPA_best"):
        outdoor_delta = outdoor_agg["v9.0_best"] - outdoor_agg["v9.0-noJEPA_best"]
        print(f"  outdoor Δ(v9.0 - noJEPA) = {outdoor_delta:.3f}")
    if outdoor_agg.get("v9.0_best") and outdoor_agg.get("LQ"):
        outdoor_lift = outdoor_agg["v9.0_best"] - outdoor_agg["LQ"]
        print(f"  outdoor lift v9.0 over LQ = {outdoor_lift:.3f}")

    # === Phase 6: write outputs ===
    # per-seq CSV (combined indoor + outdoor)
    csv_path = os.path.join(args.out_dir, "per_seq_motion_audit.csv")
    keys = ["dataset", "seq", "frames", "HxW", "motion_lq", "motion_hq", "bucket",
            "LQ_rgb_psnr", "v9.0_best_rgb_psnr", "v9.0-noJEPA_best_rgb_psnr"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for name in indoor_seq_names:
            row = {"dataset": "indoor", "seq": name, "bucket": bucket[name],
                   "motion_lq": motion["indoor"][name]["motion_lq"],
                   "motion_hq": motion["indoor"][name]["motion_hq"],
                   "frames": indoor_results[name].get("frames"),
                   "HxW": indoor_results[name].get("HxW")}
            for label, _, _ in targets:
                row[f"{label}_rgb_psnr"] = indoor_results[name].get(f"{label}_rgb_psnr")
            w.writerow(row)
        for name in sorted(motion["outdoor"].keys()):
            row = {"dataset": "outdoor", "seq": name, "bucket": "outdoor",
                   "motion_lq": motion["outdoor"][name]["motion_lq"],
                   "motion_hq": motion["outdoor"][name]["motion_hq"],
                   "frames": outdoor_results[name].get("frames"),
                   "HxW": outdoor_results[name].get("HxW")}
            for label, _, _ in targets:
                row[f"{label}_rgb_psnr"] = outdoor_results[name].get(f"{label}_rgb_psnr")
            w.writerow(row)
    print(f"\n[motion_audit] wrote {csv_path}")

    # MD summary
    md_path = os.path.join(args.out_dir, "motion_audit_summary.md")
    with open(md_path, "w") as f:
        f.write("# AP6 Motion-Bucket + Outdoor Audit Summary\n\n")
        f.write("## Indoor: per-bucket RGB-PSNR (uint8 [0,255], full-frame tiled inference)\n\n")
        f.write("| Bucket | n | LQ | v9.0 | noJEPA | Δ(v9.0−noJEPA) | lift v9.0 | lift noJEPA |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for b in ("static", "mild", "dynamic"):
            bs = bucket_summary[b]
            f.write(f"| {b} | {bs['n_seqs']} | {bs['LQ']:.3f} | {bs['v9.0']:.3f} | "
                    f"{bs['noJEPA']:.3f} | **{bs.get('delta_v9.0_minus_noJEPA', 0):.3f}** | "
                    f"{bs.get('lift_v9.0', 0):.3f} | {bs.get('lift_noJEPA', 0):.3f} |\n")
        # Verdict
        deltas = [bucket_summary[b].get("delta_v9.0_minus_noJEPA", 0) for b in ("static", "mild", "dynamic")]
        if deltas[2] - deltas[0] >= 1.0:
            verdict = "MOTION-CORRELATED (Δ grows with motion)"
        elif abs(deltas[2] - deltas[0]) < 0.5:
            verdict = "FLAT (Δ similar across buckets — V-JEPA gives scene-prior independent of motion)"
        else:
            verdict = "AMBIGUOUS"
        f.write(f"\n**Test A verdict:** {verdict}\n")
        f.write(f"  Δ_static − Δ_dynamic = {deltas[2] - deltas[0]:+.3f}\n\n")
        f.write("## Outdoor: aggregate RGB-PSNR\n\n")
        f.write(f"Outdoor avg motion_lq: {motion_outdoor_avg:.4f} (vs indoor avg {motion_indoor_avg:.4f})\n\n")
        f.write("| Model | RGB-PSNR mean | (vs indoor v9.0=23.49) |\n")
        f.write("|---|---:|---|\n")
        for label, _, _ in targets:
            f.write(f"| {label} | {outdoor_agg[label]:.3f} | — |\n")
        if outdoor_agg.get("v9.0_best") and outdoor_agg.get("v9.0-noJEPA_best"):
            f.write(f"\n**Outdoor Δ(v9.0 − noJEPA) = {outdoor_delta:.3f}** "
                    f"(vs indoor agg Δ similar?)\n")
    print(f"[motion_audit] wrote {md_path}")
    print(f"\n[motion_audit] DONE")


if __name__ == "__main__":
    main()
