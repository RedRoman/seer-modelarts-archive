"""tools/prior_ablation_v9_4e.py — BN7 v9.4e tile-consistent V-JEPA cache prototype.

Path (a): non-square padded full-frame V-JEPA encode + per-tile token crop.
Skip the wrapper's _resize_bcthw and call backbone.model directly on padded input.
Per-tile, temporarily monkey-patch pixel_branch.vjepa_grid to the cropped token shape.

Required eval (BN4):
  positive  hard cases: pair6_2, pair5_2
  negative controls   : pair7_2, pair8_2

Primary checkpoint (BN7): v9.0b_mix_best.

Acceptance (BN5):
  Strong pass: cache ≥ tile512 + 0.20 dB mean AND no negative-control drops > 0.20
  Weak pass  : within ±0.10 dB of tile512 mean AND negative controls don't degrade
  Fail       : > 0.20 dB mean drop OR negative controls regress > 0.20
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import (
    load_seq_full, build_v9_model, rgb_psnr_uint8, make_taper_window,
)


CKPT_DEFAULT = "/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt"
CFG_DEFAULT  = "/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml"


def list_seqs(root, split="test"):
    ds = SDSDDataset(root=root, subset="direct", split=split,
                     clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


def to_u8(x): return (x[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
def mean_psnr(p, g): return float(np.mean([rgb_psnr_uint8(p[t], g[t]) for t in range(p.shape[0])]))


def _normalize_bcthw(video_01):
    # Match wrapper's normalization (ImageNet mean/std)
    mean = torch.tensor([0.485, 0.456, 0.406], device=video_01.device, dtype=video_01.dtype).view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=video_01.device, dtype=video_01.dtype).view(1, 3, 1, 1, 1)
    return (video_01 - mean) / std


@torch.no_grad()
def encode_vjepa_fullframe(model, video_01: torch.Tensor, autocast_dtype=torch.bfloat16
                            ) -> Tuple[List[torch.Tensor], Tuple[int, int, int]]:
    """Bypass wrapper's _resize_bcthw. Call backbone.model directly on
    (B, 3, T, H_pad, W_pad) where H_pad, W_pad are multiples of patch_size.
    Returns: (feats per layer with shape (B, T_v*H_v*W_v, D), token_grid (T_v, H_v, W_v))
    """
    B, C, T, H, W = video_01.shape
    backbone = model.backbone  # VJepaBackboneLocal21
    patch_size = backbone.patch_size
    tubelet_size = backbone.tubelet_size
    assert H % patch_size == 0 and W % patch_size == 0, f"non-multiple-of-{patch_size}: {H}x{W}"

    # Normalize (matches wrapper). Skip wrapper's resize.
    x = _normalize_bcthw(video_01.float()).to(video_01.dtype)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        out = backbone.model(x)
    if not isinstance(out, list):
        out = [out]
    feats = out
    token_grid = (T // tubelet_size, H // patch_size, W // patch_size)
    # Verify
    expected = token_grid[0] * token_grid[1] * token_grid[2]
    assert feats[0].shape[1] == expected, \
        f"backbone returned N={feats[0].shape[1]} but token_grid={token_grid} expects {expected}"
    return feats, token_grid


def crop_vjepa_tokens(feats_full: List[torch.Tensor],
                       grid_full: Tuple[int, int, int],
                       h_start: int, h_end: int, w_start: int, w_end: int,
                       H_full: int, W_full: int):
    """Given full-frame V-JEPA features and pixel-tile coordinates, return cropped
    tokens + new grid. Pixel-coords (h_start..h_end, w_start..w_end) → token coords."""
    T_v, H_v, W_v = grid_full
    # Map pixel coords to token coords (H_v = H_full/patch, etc. exactly when divisible)
    pixel_per_tok_h = H_full / H_v
    pixel_per_tok_w = W_full / W_v
    th0 = max(0, int(round(h_start / pixel_per_tok_h)))
    th1 = min(H_v, int(round(h_end / pixel_per_tok_h)))
    tw0 = max(0, int(round(w_start / pixel_per_tok_w)))
    tw1 = min(W_v, int(round(w_end / pixel_per_tok_w)))
    if th1 <= th0: th1 = th0 + 1
    if tw1 <= tw0: tw1 = tw0 + 1

    cropped = []
    for f in feats_full:
        # f shape: (B, T_v*H_v*W_v, D)
        B, N, D = f.shape
        f5 = f.view(B, T_v, H_v, W_v, D)
        c = f5[:, :, th0:th1, tw0:tw1, :].contiguous()
        c = c.view(B, T_v * (th1 - th0) * (tw1 - tw0), D)
        cropped.append(c)
    new_grid = (T_v, th1 - th0, tw1 - tw0)
    return cropped, new_grid


@torch.no_grad()
def tiled_inference_with_cache(model, video_lq: torch.Tensor, feats_full, grid_full,
                                 H_full: int, W_full: int,
                                 tile: int = 256, overlap: int = 64,
                                 autocast_dtype=torch.bfloat16) -> torch.Tensor:
    """Tile pixel_branch only; reuse full-frame V-JEPA tokens via spatial crop.
    Per-tile, temporarily set pixel_branch.vjepa_grid to the cropped grid."""
    B, C, T, H, W = video_lq.shape
    device = video_lq.device
    dtype = video_lq.dtype
    stride = tile - overlap
    pred_acc = torch.zeros_like(video_lq)
    weight_acc = torch.zeros(1, 1, 1, H, W, device=device, dtype=dtype)
    h_starts = list(range(0, max(H - tile, 0) + 1, stride))
    if h_starts[-1] != H - tile: h_starts.append(max(H - tile, 0))
    w_starts = list(range(0, max(W - tile, 0) + 1, stride))
    if w_starts[-1] != W - tile: w_starts.append(max(W - tile, 0))

    saved_grid = model.pixel_branch.vjepa_grid
    try:
        for h0 in h_starts:
            for w0 in w_starts:
                h1 = min(h0 + tile, H); w1 = min(w0 + tile, W)
                patch = video_lq[..., h0:h1, w0:w1]
                cropped, new_grid = crop_vjepa_tokens(
                    feats_full, grid_full, h0, h1, w0, w1, H_full, W_full)
                model.pixel_branch.vjepa_grid = new_grid
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    cropped = [f.to(patch.dtype) for f in cropped]
                    out = model.pixel_branch(patch, cropped)
                pred_patch = out["pred_pixels"].clamp(0, 1).float()
                taper = make_taper_window(
                    h1 - h0, w1 - w0, overlap, device, dtype,
                    taper_top=(h0 > 0), taper_bottom=(h1 < H),
                    taper_left=(w0 > 0), taper_right=(w1 < W),
                )
                taper = taper.view(1, 1, 1, h1 - h0, w1 - w0)
                pred_acc[..., h0:h1, w0:w1] += pred_patch.to(dtype) * taper
                weight_acc[..., h0:h1, w0:w1] += taper
    finally:
        model.pixel_branch.vjepa_grid = saved_grid

    return (pred_acc / weight_acc.clamp_min(1e-8)).clamp(0, 1)


# tile=256/512 baseline (no cache, per-tile V-JEPA encode)
@torch.no_grad()
def tiled_inference_baseline(model, video_lq, tile, overlap, autocast_dtype=torch.bfloat16):
    B, C, T, H, W = video_lq.shape
    device = video_lq.device; dtype = video_lq.dtype
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/prior_ablation_v9_4e")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--cfg", default=CFG_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--n_frames", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    print(f"[v9.4e] device: {device} ckpt: {args.ckpt}")

    seqs = list_seqs(args.val_root, "test")
    seq_by_name = {n: (n, lq, gt) for (n, lq, gt) in seqs}

    target_seqs = ["pair6_2", "pair5_2", "pair7_2", "pair8_2"]  # BN4
    tile, overlap = 256, 64  # match v9.4 protocol for fair Δ vs cache
    tile_baseline_512, overlap_baseline_512 = 512, 128

    model = build_v9_model(args.cfg, args.ckpt, device)

    rows = []
    for seq_name in target_seqs:
        if seq_name not in seq_by_name:
            print(f"  [skip] {seq_name}"); continue
        n, lq_paths, gt_paths = seq_by_name[seq_name]
        print(f"\n=== {seq_name} ===")
        lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=args.n_frames)
        H, W = lq.shape[-2:]
        # Pad H to multiple of patch_size=16. W=1920 already divisible. H=1080 → pad to 1088.
        Hpad = ((H + 15) // 16) * 16
        Wpad = ((W + 15) // 16) * 16
        pad_h = Hpad - H; pad_w = Wpad - W
        # 5D F.pad doesn't support reflect; replicate via 4D reshape
        if pad_h > 0 or pad_w > 0:
            B5, C5, T5, H5, W5 = lq.shape
            lq_4d = lq.permute(0, 2, 1, 3, 4).reshape(B5 * T5, C5, H5, W5)
            lq_4d = F.pad(lq_4d, (0, pad_w, 0, pad_h), mode="reflect")
            lq_pad = lq_4d.reshape(B5, T5, C5, H5 + pad_h, W5 + pad_w).permute(0, 2, 1, 3, 4).contiguous()
        else:
            lq_pad = lq
        lq_pad = lq_pad.to(device)
        lq_dev = lq.to(device); gt_dev = gt.to(device)

        # Encode full-frame V-JEPA on padded input
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            feats_full, grid_full = encode_vjepa_fullframe(model, lq_pad)
            t_enc = time.time() - t0
            vram_enc = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  V-JEPA full-frame: token_grid={grid_full}, encode={t_enc:.1f}s, vram_peak={vram_enc:.2f}GB")
        except Exception as e:
            print(f"  ✗ V-JEPA full-frame encode FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return

        # Run cache-based tiled inference (using padded coords for V-JEPA, but pixel-branch on padded then crop back)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        pred_cache_pad = tiled_inference_with_cache(
            model, lq_pad, feats_full, grid_full, Hpad, Wpad,
            tile=tile, overlap=overlap)
        t_cache = time.time() - t0
        # Crop back to (H, W)
        pred_cache = pred_cache_pad[..., :H, :W]
        psnr_cache = mean_psnr(to_u8(pred_cache), to_u8(gt_dev))
        print(f"  cache (tile=256, full-frame V-JEPA)  PSNR={psnr_cache:.3f}  ({t_cache:.1f}s)")

        # Reference: tile=512 baseline (no cache)
        t0 = time.time()
        pred_t512 = tiled_inference_baseline(model, lq_dev, tile=512, overlap=128)
        t_t512 = time.time() - t0
        psnr_t512 = mean_psnr(to_u8(pred_t512), to_u8(gt_dev))
        print(f"  tile=512 (no cache, per-tile V-JEPA) PSNR={psnr_t512:.3f}  ({t_t512:.1f}s)")

        # Reference: tile=256 baseline (no cache, the broken protocol)
        t0 = time.time()
        pred_t256 = tiled_inference_baseline(model, lq_dev, tile=256, overlap=64)
        t_t256 = time.time() - t0
        psnr_t256 = mean_psnr(to_u8(pred_t256), to_u8(gt_dev))
        print(f"  tile=256 (no cache, per-tile V-JEPA) PSNR={psnr_t256:.3f}  ({t_t256:.1f}s)")

        rows.append({
            "seq": seq_name, "Hpad": Hpad, "Wpad": Wpad,
            "token_grid_full": str(grid_full),
            "psnr_cache_t256": psnr_cache,
            "psnr_t512_baseline": psnr_t512,
            "psnr_t256_baseline": psnr_t256,
            "delta_cache_minus_t512": psnr_cache - psnr_t512,
            "delta_cache_minus_t256": psnr_cache - psnr_t256,
            "vram_encode_gb": vram_enc,
            "t_encode_s": round(t_enc, 2),
            "t_cache_inf_s": round(t_cache, 2),
        })

    # Aggregate + verdict
    n = len(rows)
    if n == 0:
        print("no results"); return
    mean_cache = float(np.mean([r["psnr_cache_t256"] for r in rows]))
    mean_t512 = float(np.mean([r["psnr_t512_baseline"] for r in rows]))
    mean_delta = mean_cache - mean_t512
    pos_seqs = {"pair6_2", "pair5_2"}
    neg_seqs = {"pair7_2", "pair8_2"}
    neg_max_drop = max([(r["delta_cache_minus_t512"]) for r in rows if r["seq"] in neg_seqs], default=0)
    neg_min = min([(r["delta_cache_minus_t512"]) for r in rows if r["seq"] in neg_seqs], default=0)

    print(f"\n=== AGGREGATE on {[r['seq'] for r in rows]} ===")
    print(f"  cache mean (t256-tile, full-frame V-JEPA): {mean_cache:.3f}")
    print(f"  t512  mean (no cache):                     {mean_t512:.3f}")
    print(f"  Δ_mean (cache - t512):                     {mean_delta:+.3f}")
    print(f"  neg-control min Δ: {neg_min:+.3f} (worst negative-control regression)")

    # Verdict per BN5
    if mean_delta >= 0.20 and neg_min > -0.20:
        verdict = ("Strong PASS: cache mean ≥ tile512 + 0.20 dB AND no neg-control drops > 0.20.\n"
                   "→ tile-consistent V-JEPA cache recovers AND improves beyond tile=512 protocol.\n"
                   "→ Mechanism closure: V-JEPA per-tile inconsistency was the dominant artifact.")
    elif abs(mean_delta) < 0.10 and neg_min > -0.20:
        verdict = ("Weak PASS: cache within ±0.10 dB of tile=512, neg controls intact.\n"
                   "→ Cache recovers tile artifact ≈ as well as tile=512, but no extra gain.\n"
                   "→ tile=512 is sufficient as the locked protocol.")
    elif mean_delta <= -0.20 or neg_min <= -0.20:
        verdict = ("FAIL: cache loses ≥ 0.20 dB mean OR neg control regressed ≥ 0.20.\n"
                   "→ Path (a) cache does not improve over tile=512. Close v9.4e.")
    else:
        verdict = (f"INCONCLUSIVE: cache Δ_mean={mean_delta:+.3f}, neg_min={neg_min:+.3f}.\n"
                   "→ Mixed signal; inspect per-seq.")

    print(f"\n=== VERDICT (BN5): ===\n{verdict}")

    # Save CSV
    csv_path = os.path.join(args.out_dir, "v9_4e_results.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}")

    md_path = os.path.join(args.out_dir, "ablation_summary.md")
    with open(md_path, "w") as f:
        f.write("# BN7 v9.4e-tile-consistent-vjepa-cache prototype Summary\n\n")
        f.write(f"Path (a): non-square padded full-frame V-JEPA encode + per-tile token crop.\n")
        f.write(f"Primary ckpt: {os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))}\n")
        f.write(f"Eval: 4 seqs (BN4): {[r['seq'] for r in rows]}\n")
        f.write(f"V-JEPA padded shape: {rows[0]['Hpad']}x{rows[0]['Wpad']}, token_grid={rows[0]['token_grid_full']}\n")
        f.write(f"VRAM peak (encode):  {rows[0]['vram_encode_gb']:.2f} GB\n\n")
        f.write("## Per-seq results\n\n")
        f.write("| seq | cache (t256-tile + full V-JEPA) | t512 baseline | t256 baseline | Δ cache-t512 | Δ cache-t256 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['seq']} | {r['psnr_cache_t256']:.3f} | "
                    f"{r['psnr_t512_baseline']:.3f} | {r['psnr_t256_baseline']:.3f} | "
                    f"{r['delta_cache_minus_t512']:+.3f} | {r['delta_cache_minus_t256']:+.3f} |\n")
        f.write(f"\n**Aggregate (n={n}):** cache mean = {mean_cache:.3f}, t512 mean = {mean_t512:.3f}, "
                f"Δ_mean = {mean_delta:+.3f} dB, neg-control min Δ = {neg_min:+.3f}\n\n")
        f.write(f"## BN5 Verdict\n\n**{verdict}**\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
