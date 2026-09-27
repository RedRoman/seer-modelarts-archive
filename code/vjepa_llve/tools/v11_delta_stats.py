"""tools/v11_delta_stats.py — CZ4 Probe A: V-JEPA z_L vs z_H delta statistics.

Falsifies (or supports) the v11 = JEPA-Delta Action Prior (JDAP) hypothesis:
"V-JEPA contains a learnable restoration action z_L -> z_H."

For each test clip:
  z_L_l = E_l(LQ)   for l in feature_layers
  z_H_l = E_l(HQ)   (training-only oracle)

Compute per-layer:
  cos(z_L, z_H)             — token-mean
  ||z_H - z_L|| / ||z_L||   — token-mean  (relative delta magnitude)
  ||z_H - z_L||             — absolute delta magnitude
  ||z_L||, ||z_H||           — sanity

Output:
  /cache/zihaohe/checkpoint/v11_probes/delta_stats.csv
  /cache/zihaohe/checkpoint/v11_probes/delta_stats_summary.md

Pass criteria (CZ Probe A):
  delta is NOT trivially zero (||delta||/||z_L|| > 0.01 in some layer)
  cos(z_L, z_H) is NOT identical to 1.0 (some learnable signal exists)
  variance of delta across sequences is non-trivial

Spec: VJEPA_LLVE_CODE_REVIEW.md §CZ4.A (2026-05-03).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import _detect_pair_dirs
from data.common import list_sequences, list_images, read_image_01
from models.vjepa_backbone import build_vjepa_backbone


# --- splits (reuse v10_audit definitions for consistency) ----------------
LLVE_STCD_12 = ["pair13","pair15","pair21","pair23","pair31","pair33",
                "pair50","pair52","pair58","pair60","pair70"]
ORIGINAL_INDOOR_14 = ["pair34_2","pair3_2","pair4","pair4_2","pair5","pair5_2",
                      "pair6","pair6_2","pair7","pair7_2","pair8","pair8_2",
                      "pair9","pair9_2"]
INDOOR_ROOT  = "/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor"
OUTDOOR_ROOT = "/home/ma-user/workspace/zihaohe/datasets/SDSD_outdoor"


def index_seqs(root: str, split: str = "test") -> dict:
    """Return {seq: (lq_paths, gt_paths)} aligned by basename intersection (CN-A fix)."""
    lq_dir, gt_dir, _ = _detect_pair_dirs(root, split, "LQ", "GT")
    out = {}
    for seq_dir in list_sequences(lq_dir):
        seq = os.path.basename(seq_dir)
        lq_frames = list_images(os.path.join(lq_dir, seq))
        gt_frames = list_images(os.path.join(gt_dir, seq))
        if not lq_frames or not gt_frames:
            continue
        lq_by = {os.path.basename(p): p for p in lq_frames}
        gt_by = {os.path.basename(p): p for p in gt_frames}
        common = sorted(set(lq_by) & set(gt_by))
        if len(common) < 16:
            continue
        out[seq] = ([lq_by[n] for n in common], [gt_by[n] for n in common])
    return out


def load_clip_for_vjepa(lq_paths: List[str], gt_paths: List[str],
                        n_frames: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load center 16 frames (3, T, H, W) in [0,1] float, full resolution.
    V-JEPA wrapper handles resize-to-256 internally."""
    n_total = len(lq_paths)
    T = min(n_frames, n_total)
    start = (n_total - T) // 2
    lq = np.stack([read_image_01(lq_paths[start+i]) for i in range(T)], axis=0)
    gt = np.stack([read_image_01(gt_paths[start+i]) for i in range(T)], axis=0)
    lq = torch.from_numpy(lq).permute(3, 0, 1, 2).unsqueeze(0).float()
    gt = torch.from_numpy(gt).permute(3, 0, 1, 2).unsqueeze(0).float()
    return lq, gt


@torch.no_grad()
def encode_pair(backbone, lq_dev: torch.Tensor, gt_dev: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    z_L = backbone(lq_dev)["feats"]
    z_H = backbone(gt_dev)["feats"]
    return z_L, z_H


def per_layer_stats(z_L: torch.Tensor, z_H: torch.Tensor) -> Dict[str, float]:
    """z_L, z_H: (B, N, D). Return cos / norm / delta-norm stats."""
    z_L = z_L.float(); z_H = z_H.float()
    delta = z_H - z_L                          # (B, N, D)
    norm_L = z_L.norm(dim=-1)                  # (B, N)
    norm_H = z_H.norm(dim=-1)                  # (B, N)
    norm_delta = delta.norm(dim=-1)            # (B, N)
    cos = F.cosine_similarity(z_L, z_H, dim=-1)  # (B, N)
    rel = norm_delta / norm_L.clamp_min(1e-8)
    return {
        "cos_mean": cos.mean().item(),
        "cos_min":  cos.min().item(),
        "cos_p10":  cos.quantile(0.10).item(),
        "rel_delta_mean": rel.mean().item(),
        "rel_delta_max":  rel.max().item(),
        "abs_delta_mean": norm_delta.mean().item(),
        "norm_L_mean":    norm_L.mean().item(),
        "norm_H_mean":    norm_H.mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/delta_stats")
    ap.add_argument("--seqs_indoor", default="LLVE_STCD_12,ORIGINAL_INDOOR_14",
                    help="comma-separated split names to probe (indoor)")
    ap.add_argument("--max_outdoor", type=int, default=10,
                    help="how many outdoor seqs to sample")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    import yaml
    with open(args.cfg) as f:
        cfg_full = yaml.safe_load(f)
    cfg = cfg_full.get("model", cfg_full)
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    layers = backbone.feature_layers
    print(f"[probe A] V-JEPA backbone built; feature_layers = {layers}")

    # build seq lists
    in_idx  = index_seqs(INDOOR_ROOT)
    out_idx = index_seqs(OUTDOOR_ROOT)
    seqs: List[Tuple[str, str, list, list]] = []  # (split, name, lq_paths, gt_paths)
    split_map = {"LLVE_STCD_12": LLVE_STCD_12, "ORIGINAL_INDOOR_14": ORIGINAL_INDOOR_14}
    for s in args.seqs_indoor.split(","):
        for name in split_map.get(s, []):
            if name in in_idx:
                seqs.append((s, name, *in_idx[name]))
    for name in list(out_idx.keys())[:args.max_outdoor]:
        seqs.append(("OUTDOOR", name, *out_idx[name]))
    print(f"[probe A] probing {len(seqs)} sequences")

    rows = []
    for split, name, lq_p, gt_p in seqs:
        t0 = time.time()
        lq, gt = load_clip_for_vjepa(lq_p, gt_p)
        lq_dev, gt_dev = lq.to(device), gt.to(device)
        z_L_layers, z_H_layers = encode_pair(backbone, lq_dev, gt_dev)
        for li, (z_L, z_H) in enumerate(zip(z_L_layers, z_H_layers)):
            stats = per_layer_stats(z_L, z_H)
            row = {"split": split, "seq": name, "layer_idx": li,
                   "layer_id": layers[li] if li < len(layers) else li,
                   **stats, "time_s": round(time.time()-t0, 1)}
            rows.append(row)
        print(f"  [{split}/{name}] {len(z_L_layers)} layers; took {time.time()-t0:.1f}s")

    # CSV
    csv_path = os.path.join(args.out_dir, "delta_stats_perlayer.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}")

    # aggregate per layer
    md_path = os.path.join(args.out_dir, "delta_stats_summary.md")
    with open(md_path, "w") as f:
        f.write("# v11 Probe A — V-JEPA delta statistics\n\n")
        f.write(f"n_sequences={len(seqs)}; backbone feature_layers={list(layers)}\n\n")
        f.write("## Per-layer aggregate (mean across all sequences)\n\n")
        f.write("| layer_idx | layer_id | cos_mean | cos_p10 | cos_min | rel_delta_mean | rel_delta_max | abs_delta_mean | ||z_L|| | ||z_H|| |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        n_layers = max(r["layer_idx"] for r in rows) + 1
        for li in range(n_layers):
            sub = [r for r in rows if r["layer_idx"] == li]
            if not sub: continue
            agg = {k: float(np.mean([r[k] for r in sub])) for k in
                   ["cos_mean","cos_p10","cos_min","rel_delta_mean","rel_delta_max",
                    "abs_delta_mean","norm_L_mean","norm_H_mean"]}
            f.write(f"| {li} | {sub[0]['layer_id']} | {agg['cos_mean']:.4f} | {agg['cos_p10']:.4f} | "
                    f"{agg['cos_min']:.4f} | {agg['rel_delta_mean']:.4f} | {agg['rel_delta_max']:.4f} | "
                    f"{agg['abs_delta_mean']:.3f} | {agg['norm_L_mean']:.3f} | {agg['norm_H_mean']:.3f} |\n")
        f.write("\n## Per-layer per-split breakdown\n\n")
        for split in sorted(set(r["split"] for r in rows)):
            f.write(f"### {split}\n\n")
            f.write("| layer_idx | cos_mean | rel_delta_mean | abs_delta_mean |\n|---:|---:|---:|---:|\n")
            for li in range(n_layers):
                sub = [r for r in rows if r["layer_idx"] == li and r["split"] == split]
                if not sub: continue
                cos = np.mean([r["cos_mean"] for r in sub])
                rel = np.mean([r["rel_delta_mean"] for r in sub])
                abs_d = np.mean([r["abs_delta_mean"] for r in sub])
                f.write(f"| {li} | {cos:.4f} | {rel:.4f} | {abs_d:.3f} |\n")
            f.write("\n")

        # CZ Probe A pass criteria
        f.write("## CZ Probe A verdict\n\n")
        any_layer_pass = False
        for li in range(n_layers):
            sub = [r for r in rows if r["layer_idx"] == li]
            if not sub: continue
            cos_mean = np.mean([r["cos_mean"] for r in sub])
            rel_mean = np.mean([r["rel_delta_mean"] for r in sub])
            ok = (rel_mean > 0.01) and (cos_mean < 0.999)
            f.write(f"- layer_idx {li}: cos_mean={cos_mean:.4f}, rel_delta={rel_mean:.4f}  "
                    f"=> {'PASS (delta exists)' if ok else 'FAIL (V-JEPA invariant — bad for JDAP)'}\n")
            any_layer_pass = any_layer_pass or ok
        verdict = "PASS — at least one layer shows learnable LQ→HQ delta" if any_layer_pass \
                  else "FAIL — V-JEPA may be too illumination-invariant for JDAP"
        f.write(f"\n**Probe A verdict: {verdict}**\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
