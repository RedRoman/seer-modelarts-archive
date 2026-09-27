"""tools/prior_ablation_v9_4d.py — BJ3 Step 2 + Step 3.

Step 2: tile=512/overlap=128 full-frame audit on 14 SDSD-indoor static_np test seqs,
        across 3 checkpoints (v9_0b, v9_noJEPA, v9_1). v9.0_best already in v9.4c.

Step 3: tile-size sweep 256/384/512 on 8 diagnostic seqs, ckpt v9.0_best, full_vjepa.

Output: /cache/zihaohe/checkpoint/prior_ablation_v9_4d/
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
    load_seq_full, build_v9_model, rgb_psnr_uint8, make_taper_window,
)


CKPTS = [
    # (label, run_dir, priority)  -- per BL2: P1 mandatory, P2 sanity, P3 optional
    ("v9_0_best",  "/cache/zihaohe/checkpoint/runs/v9",        2),
    ("v9_0b_mix",  "/cache/zihaohe/checkpoint/runs/v9_0b",     1),
    ("v9_noJEPA",  "/cache/zihaohe/checkpoint/runs/v9_noJEPA", 1),
    # ("v9_1a",      "/cache/zihaohe/checkpoint/runs/v9_1",     3),  # P3: skipped per BL
]


def list_seqs(root, split="test"):
    ds = SDSDDataset(root=root, subset="direct", split=split,
                     clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


@torch.no_grad()
def tiled_inference(model, video_lq, tile=512, overlap=128, autocast_dtype=torch.bfloat16):
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


def step2_reaudit(seqs, device, out_dir, n_frames=16):
    """Per BL2: each ckpt at BOTH tile=256 (overlap=64) AND tile=512 (overlap=128).
    Report mean AND median; CSV columns include reference + delta."""
    print(f"\n===== Step 2: tile=256 + tile=512 re-audit, {len(CKPTS)} checkpoints × 14 seqs × 2 tiles =====")
    tile_sizes = [(256, 64), (512, 128)]
    rows = []  # one row per (ckpt, seq, tile)
    for label, run_dir, prio in CKPTS:
        cfg_path = os.path.join(run_dir, "config.yaml")
        ckpt_path = os.path.join(run_dir, "ckpts", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [skip] {label}: ckpt not found at {ckpt_path}"); continue
        print(f"\n--- ckpt: {label} (P{prio}) ---")
        model = build_v9_model(cfg_path, ckpt_path, device)
        for tile, overlap in tile_sizes:
            print(f"  --- tile={tile} overlap={overlap} ---")
            for seq_name, lq_paths, gt_paths in seqs:
                t0 = time.time()
                lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
                lq, gt = lq.to(device), gt.to(device)
                pred = tiled_inference(model, lq, tile=tile, overlap=overlap)
                psnr = mean_psnr(to_u8(pred), to_u8(gt))
                rows.append({"checkpoint": label, "seq": seq_name, "tile": tile,
                             "overlap": overlap, "mode": "full_vjepa", "psnr": psnr})
                print(f"    [{label}/{seq_name}/t{tile}] PSNR={psnr:.3f} ({time.time()-t0:.1f}s)")
        del model
        torch.cuda.empty_cache()

    # Build per-(ckpt, seq) wide table with deltas
    from collections import defaultdict
    by_key = {(r["checkpoint"], r["seq"], r["tile"]): r["psnr"] for r in rows}
    wide_rows = []
    for label, _, _ in CKPTS:
        for seq_name, _, _ in seqs:
            p256 = by_key.get((label, seq_name, 256))
            p512 = by_key.get((label, seq_name, 512))
            if p256 is None or p512 is None: continue
            wide_rows.append({
                "checkpoint": label, "seq": seq_name,
                "psnr_t256": p256, "psnr_t512": p512,
                "delta_t512_minus_t256": p512 - p256,
            })

    # Save BL-schema CSV (long form: per-row checkpoint/seq/tile/...)
    csv_path = os.path.join(out_dir, "step2_audit_long.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "seq", "tile", "overlap", "mode", "psnr"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}")

    # Save wide-form CSV (per-(ckpt, seq) with delta)
    wide_path = os.path.join(out_dir, "step2_audit_wide.csv")
    with open(wide_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
        w.writeheader()
        for r in wide_rows: w.writerow(r)
    print(f"wrote {wide_path}")

    # Aggregate: mean AND median per (ckpt, tile)
    agg = defaultdict(list)
    for r in rows: agg[(r["checkpoint"], r["tile"])].append(r["psnr"])
    delta_agg = defaultdict(list)
    for wr in wide_rows: delta_agg[wr["checkpoint"]].append(wr["delta_t512_minus_t256"])
    print("\n=== Step 2 Aggregate (mean / median per ckpt × tile) ===")
    print(f"{'ckpt':12s}  {'t256_mean':>10s}  {'t256_med':>10s}  {'t512_mean':>10s}  {'t512_med':>10s}  "
          f"{'Δ_mean':>8s}  {'Δ_med':>8s}")
    aggregates = {}
    for label, _, _ in CKPTS:
        if (label, 256) not in agg: continue
        m256 = float(np.mean(agg[(label, 256)])); md256 = float(np.median(agg[(label, 256)]))
        m512 = float(np.mean(agg[(label, 512)])); md512 = float(np.median(agg[(label, 512)]))
        dm = float(np.mean(delta_agg[label])); dmed = float(np.median(delta_agg[label]))
        aggregates[label] = {"t256_mean": m256, "t256_med": md256,
                              "t512_mean": m512, "t512_med": md512,
                              "delta_mean": dm, "delta_med": dmed}
        print(f"{label:12s}  {m256:10.3f}  {md256:10.3f}  {m512:10.3f}  {md512:10.3f}  "
              f"{dm:+8.3f}  {dmed:+8.3f}")
    return rows, wide_rows, aggregates


def step3_tilesweep(seqs, device, out_dir, n_frames=16,
                     diag_seqs=("pair6_2","pair5_2","pair8","pair4",
                                "pair7_2","pair8_2","pair9","pair6"),
                     tiles=(256, 384, 512)):
    print(f"\n===== Step 3: tile-size sweep, ckpt=v9.0_best, full_vjepa, {len(diag_seqs)} diag seqs × {len(tiles)} tiles =====")
    cfg_path = "/cache/zihaohe/checkpoint/runs/v9/config.yaml"
    ckpt_path = "/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt"
    model = build_v9_model(cfg_path, ckpt_path, device)
    seq_by_name = {n: (n, lq, gt) for (n, lq, gt) in seqs}
    rows = []
    for tile in tiles:
        overlap = tile // 4  # match the v9.4 / v9.4c overlap ratio
        print(f"\n--- tile={tile} overlap={overlap} ---")
        for seq_name in diag_seqs:
            if seq_name not in seq_by_name:
                print(f"  [skip] {seq_name}"); continue
            n, lq_paths, gt_paths = seq_by_name[seq_name]
            t0 = time.time()
            lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
            lq, gt = lq.to(device), gt.to(device)
            pred = tiled_inference(model, lq, tile=tile, overlap=overlap)
            psnr = mean_psnr(to_u8(pred), to_u8(gt))
            rows.append({"seq": seq_name, "tile": tile, "psnr": psnr})
            print(f"  [tile={tile}/{seq_name}] PSNR={psnr:.3f} ({time.time()-t0:.1f}s)")

    csv_path = os.path.join(out_dir, "step3_tile_sweep.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}")

    print("\n=== Step 3 monotonicity (per-seq, tile 256 → 384 → 512) ===")
    by_seq_tile = {}
    for r in rows: by_seq_tile.setdefault(r["seq"], {})[r["tile"]] = r["psnr"]
    print(f"{'seq':10s}  {'tile=256':>10s}  {'tile=384':>10s}  {'tile=512':>10s}  {'mono?':>8s}")
    monotonic_count = 0
    for seq in sorted(by_seq_tile):
        d = by_seq_tile[seq]
        v = [d.get(t, 0) for t in tiles]
        is_mono = all(v[i] <= v[i+1] for i in range(len(v)-1))
        monotonic_count += int(is_mono)
        print(f"{seq:10s}  {v[0]:10.3f}  {v[1]:10.3f}  {v[2]:10.3f}  {'yes' if is_mono else 'no':>8s}")
    print(f"\n{monotonic_count}/{len(by_seq_tile)} seqs are monotonically increasing with tile size")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/prior_ablation_v9_4d")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--step", choices=["2", "3", "both"], default="both")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    print(f"[v9.4d] device: {device}")

    seqs = list_seqs(args.val_root, "test")
    print(f"[v9.4d] {len(seqs)} indoor test seqs")

    s2_rows = s2_wide = s2_agg = s3_rows = None
    if args.step in ("2", "both"):
        s2_rows, s2_wide, s2_agg = step2_reaudit(seqs, device, args.out_dir)
    if args.step in ("3", "both"):
        s3_rows = step3_tilesweep(seqs, device, args.out_dir, n_frames=args.n_frames)

    # Summary
    md_path = os.path.join(args.out_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("# BJ3 v9.4d-protocol-lock-and-reaudit Summary\n\n")
        f.write("## Step 1 (separate file): high-res V-JEPA sanity → PASS\n")
        f.write("  image_size=512  token_grid=(8,32,32)  vram=1.62 GB\n")
        f.write("  image_size=768  token_grid=(8,48,48)  vram=2.23 GB\n")
        f.write("  image_size=1024 token_grid=(8,64,64)  vram=3.08 GB\n\n")

        if s2_agg:
            f.write("## Step 2 — tile=256 vs tile=512 re-audit (per BL2 schema)\n\n")
            f.write("| ckpt | t256 mean | t256 med | t512 mean | t512 med | Δ mean | Δ med |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for c, a in s2_agg.items():
                f.write(f"| {c} | {a['t256_mean']:.3f} | {a['t256_med']:.3f} | "
                        f"{a['t512_mean']:.3f} | {a['t512_med']:.3f} | "
                        f"{a['delta_mean']:+.3f} | {a['delta_med']:+.3f} |\n")
            f.write("\nNote: full_vjepa mode only. Same 14 SDSD-indoor static_np test seqs.\n\n")
            # Per-seq wide table
            if s2_wide:
                f.write("### Per-(ckpt, seq) detail\n\n")
                f.write("| ckpt | seq | t256 | t512 | Δ |\n")
                f.write("|---|---|---:|---:|---:|\n")
                for r in s2_wide:
                    f.write(f"| {r['checkpoint']} | {r['seq']} | "
                            f"{r['psnr_t256']:.3f} | {r['psnr_t512']:.3f} | "
                            f"{r['delta_t512_minus_t256']:+.3f} |\n")
                f.write("\n")

        if s3_rows:
            f.write("## Step 3 — tile-size sweep (v9.0_best, full_vjepa, 8 diag seqs)\n\n")
            by_seq_tile = {}
            for r in s3_rows: by_seq_tile.setdefault(r["seq"], {})[r["tile"]] = r["psnr"]
            f.write("| seq | tile=256 | tile=384 | tile=512 | monotonic? |\n")
            f.write("|---|---:|---:|---:|:---:|\n")
            mono_count = 0
            for seq in sorted(by_seq_tile):
                d = by_seq_tile[seq]
                v = [d.get(t, 0) for t in (256, 384, 512)]
                is_mono = all(v[i] <= v[i+1] for i in range(len(v)-1))
                mono_count += int(is_mono)
                f.write(f"| {seq} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {'✓' if is_mono else '✗'} |\n")
            f.write(f"\n{mono_count}/{len(by_seq_tile)} seqs monotonic.\n")
    print(f"\nwrote {md_path}")


if __name__ == "__main__":
    main()
