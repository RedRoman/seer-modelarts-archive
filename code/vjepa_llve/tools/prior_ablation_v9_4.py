"""tools/prior_ablation_v9_4.py — CC4 v9.4 content-destruction probes.

Six modes per CC2/CC4 lock:
  Mode 0: full_vjepa            (baseline; reproduces 23.489)
  Mode 1: cross_attn_off        (forward_hook bypass on every VJepaCrossAttnBF)
  Mode 2: zero_vjepa            (replace e_lq_layers with zeros, same shape)
  Mode 3: stats_noise_vjepa     (Gaussian noise matched per-(layer, channel) mean+std,
                                 stats from the actual full_vjepa encoding for each tile)
  Mode 4: scene_swap_vjepa      (replace with V-JEPA from a fixed reference patch:
                                 first seq, first 16 frames, center 256x256)
  Mode 5: channel_mean_vjepa    (replace each token by per-(layer, channel) mean over tokens)

All evaluated on the same 14 SDSD-indoor static_np test seqs, same RGB-PSNR
full-frame tiled protocol as v9.3 (AA4/AT2/BA1 baseline).

Output: /cache/zihaohe/checkpoint/prior_ablation_v9_4/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# cross_attn_off via forward_hook bypass
# ---------------------------------------------------------------------------
def _bypass_hook(module, inputs, output):
    # VJepaCrossAttnBF.forward(self, x, vjepa_kv) -> x + self.proj(out)
    # Return x (the first input) to make the block an identity.
    return inputs[0]


def install_cross_attn_off(model) -> List:
    """Install forward_hooks to make every VJepaCrossAttnBF an identity. Returns handles."""
    handles = []
    for name, mod in model.pixel_branch.named_modules():
        if mod.__class__.__name__ == "VJepaCrossAttnBF":
            handles.append(mod.register_forward_hook(_bypass_hook))
    return handles


def uninstall_hooks(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Feature manipulation (zero / stats_noise / scene_swap / channel_mean)
# ---------------------------------------------------------------------------
def manipulate_features(feats: List[torch.Tensor], mode: str,
                        ref_feats: Optional[List[torch.Tensor]] = None,
                        seed: int = 42) -> List[torch.Tensor]:
    if mode == "full_vjepa":
        return feats
    if mode == "cross_attn_off":
        # No feature manipulation needed; bypass is via hook on the cross-attn module.
        return feats

    out = []
    g = torch.Generator(device="cpu").manual_seed(seed)

    for li, f in enumerate(feats):
        # f shape: (B, N, D)
        B, N, D = f.shape

        if mode == "zero_vjepa":
            f_out = torch.zeros_like(f)

        elif mode == "stats_noise_vjepa":
            # Match per-(layer, channel) mean and std using stats from f itself
            # (f IS the full_vjepa encoding for this patch, since this is called
            # on the freshly-encoded feats).
            mean = f.mean(dim=(0, 1), keepdim=True)  # (1, 1, D)
            std = f.std(dim=(0, 1), keepdim=True)    # (1, 1, D)
            noise = torch.randn(f.shape, generator=g, dtype=torch.float32)
            f_out = (noise.to(f.device) * std.float() + mean.float()).to(f.dtype)

        elif mode == "scene_swap_vjepa":
            assert ref_feats is not None, "scene_swap requires ref_feats"
            r = ref_feats[li]
            assert r.shape == f.shape, f"ref_feats[{li}].shape {r.shape} != f.shape {f.shape}"
            f_out = r.to(f.dtype).to(f.device)

        elif mode == "channel_mean_vjepa":
            # Per-(layer, channel) mean across the N=T'·H'·W' tokens; replace each token by it.
            mean_vec = f.mean(dim=1, keepdim=True)  # (B, 1, D)
            f_out = mean_vec.expand_as(f).contiguous()

        else:
            raise ValueError(f"unknown mode {mode!r}")

        out.append(f_out)
    return out


# ---------------------------------------------------------------------------
# Tiled inference with intervention
# ---------------------------------------------------------------------------
@torch.no_grad()
def tiled_inference_with_intervention(model, video_lq: torch.Tensor, mode: str,
                                       ref_feats: Optional[List[torch.Tensor]] = None,
                                       tile: int = 256, overlap: int = 64,
                                       autocast_dtype=torch.bfloat16) -> torch.Tensor:
    B, C, T, H, W = video_lq.shape
    device = video_lq.device
    dtype = video_lq.dtype

    if H <= tile and W <= tile:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            feats = model.encode_video_features(video_lq)
            feats = manipulate_features(feats, mode, ref_feats=ref_feats)
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
                feats = manipulate_features(feats, mode, ref_feats=ref_feats)
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


# ---------------------------------------------------------------------------
# Reference V-JEPA cache for scene_swap
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_ref_feats(model, val_root: str, device, n_frames: int = 16,
                    ref_seq_idx: int = 0, autocast_dtype=torch.bfloat16):
    """Compute V-JEPA features on a fixed reference patch from the TRAIN split
    (first train seq, center 256x256 crop), to ensure no overlap with any of the
    14 test seqs used in the eval protocol. Used for scene_swap_vjepa."""
    train_seqs = list_seqs(val_root, "train")
    name, lq_paths, gt_paths = train_seqs[ref_seq_idx]
    ref_lq, _ = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
    H, W = ref_lq.shape[-2:]
    h0 = (H - 256) // 2
    w0 = (W - 256) // 2
    ref_patch = ref_lq[..., h0:h0 + 256, w0:w0 + 256].to(device)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        feats = model.encode_video_features(ref_patch)
    feats = [f.detach().clone() for f in feats]
    print(f"[scene_swap] reference: TRAIN seq={name} center 256x256 frames 0-{ref_patch.shape[2]-1}")
    print(f"[scene_swap] feature shapes: {[tuple(f.shape) for f in feats]}")
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/prior_ablation_v9_4")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt")
    ap.add_argument("--cfg", default="configs/train_v9.yaml")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--ref_seq_idx", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v9.4] device: {device}")

    seqs = list_seqs(args.val_root, "test")
    print(f"[v9.4] {len(seqs)} indoor test seqs")

    model = build_v9_model(args.cfg, args.ckpt, device)

    # Build scene_swap reference cache from TRAIN split (no overlap with eval seqs)
    ref_feats = build_ref_feats(model, args.val_root, device, n_frames=args.n_frames,
                                ref_seq_idx=args.ref_seq_idx)

    modes = [
        "full_vjepa",
        "cross_attn_off",
        "zero_vjepa",
        "stats_noise_vjepa",
        "scene_swap_vjepa",
        "channel_mean_vjepa",
    ]
    per_seq_results: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        print(f"\n[v9.4] === {mode} ===")
        # Install / uninstall hooks for cross_attn_off
        hook_handles = install_cross_attn_off(model) if mode == "cross_attn_off" else []
        try:
            for seq_name, lq_paths, gt_paths in seqs:
                t0 = time.time()
                lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=args.n_frames)
                T_used = lq.shape[2]
                lq, gt = lq.to(device), gt.to(device)

                pred = tiled_inference_with_intervention(
                    model, lq, mode=mode, ref_feats=ref_feats,
                )

                pred_uint8 = (pred[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                gt_uint8 = (gt[0].permute(1, 2, 3, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                psnrs = [rgb_psnr_uint8(pred_uint8[t], gt_uint8[t]) for t in range(T_used)]
                seq_psnr = float(np.mean(psnrs))
                per_seq_results.setdefault(seq_name, {})[mode] = seq_psnr
                print(f"  [{mode}/{seq_name}] PSNR={seq_psnr:.3f} ({time.time()-t0:.1f}s)")
        finally:
            uninstall_hooks(hook_handles)

    # Aggregate
    aggregate = {}
    for mode in modes:
        vals = [per_seq_results[s][mode] for s in per_seq_results if mode in per_seq_results[s]]
        aggregate[mode] = {"mean": float(np.mean(vals)), "n": len(vals)}

    baseline = aggregate["full_vjepa"]["mean"]
    deltas = {m: aggregate[m]["mean"] - baseline for m in modes}
    print(f"\n=== AGGREGATE ===")
    for m in modes:
        print(f"  {m:25s} PSNR={aggregate[m]['mean']:.3f}  Δ vs baseline={deltas[m]:+.3f}")

    # CC3 verdict
    def cls(d):
        ad = abs(d)
        if ad < 0.2: return "no dep"
        elif ad < 0.5: return "weak"
        else: return "meaningful"

    d_off = deltas["cross_attn_off"]
    d_zero = deltas["zero_vjepa"]
    d_noise = deltas["stats_noise_vjepa"]
    d_swap = deltas["scene_swap_vjepa"]
    d_mean = deltas["channel_mean_vjepa"]

    # Root: cross_attn_off
    if cls(d_off) == "meaningful":
        # Case 1: cross-attn matters; further interpret
        if cls(d_swap) == "meaningful":
            verdict = ("Case 1A: cross_attn_off hurts AND scene_swap hurts. "
                       "LQ-aligned V-JEPA content is causally used. "
                       "Strongest result for keeping V-JEPA-specific architecture work; "
                       "BVI-RLV scale-up + V-JEPA paper story justified.")
        elif cls(d_zero) == "meaningful" and cls(d_noise) != "meaningful":
            verdict = ("Case 1B: cross_attn_off + zero hurt; stats_noise close to full. "
                       "First/second-order statistics matter, not token content. "
                       "Cross-attn acts as a content-agnostic 'valid prior present' gate.")
        elif cls(d_zero) != "meaningful" and cls(d_noise) != "meaningful":
            verdict = ("Case 1C: cross_attn_off hurts but zero/noise are close to full. "
                       "Branch is useful but actual V-JEPA content is not necessary. "
                       "Next: replace V-JEPA with learned latent tokens or cheaper prior.")
        else:
            verdict = ("Case 1-mixed: cross_attn_off hurts. See per-mode deltas below for nuance.")
    elif cls(d_off) == "no dep":
        verdict = ("Case 2: cross_attn_off barely moves PSNR. "
                   "V-JEPA cross-attn is effectively bypassed at inference on SDSD-indoor. "
                   "noJEPA gap is training-dynamics, NOT inference-time feature-dep. "
                   "Architecture finding: current cross-attn design is not exploiting V-JEPA at inference.")
    else:
        verdict = ("Case 3: cross_attn_off in 0.2-0.5 dB band. Weak inference dependence; "
                   "report per-seq spread before fork choice.")

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
        f.write("# CC4 v9.4-prior-ablation Summary\n\n")
        f.write("6-mode V-JEPA content-destruction probe on v9.0_best, inference-only.\n")
        f.write("Eval: 14 SDSD-indoor static_np test seqs, RGB-PSNR uint8 full-frame tiled.\n\n")

        f.write("## Aggregate per-mode\n\n")
        f.write("| Mode | RGB-PSNR (mean) | Δ vs baseline | Classification |\n")
        f.write("|---|---:|---:|:---:|\n")
        for m in modes:
            d = deltas[m]
            f.write(f"| {m} | {aggregate[m]['mean']:.3f} | {d:+.3f} | {cls(d) if m != 'full_vjepa' else '—'} |\n")

        f.write("\n## CC3 verdict\n\n")
        f.write(f"**{verdict}**\n\n")

        f.write("## Per-seq detail\n\n")
        f.write("| seq | full_vjepa | cross_attn_off (Δ) | zero (Δ) | stats_noise (Δ) | scene_swap (Δ) | channel_mean (Δ) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for seq in sorted(per_seq_results):
            d = per_seq_results[seq]
            base = d.get("full_vjepa", 0)
            row = [
                f"{base:.3f}",
                f"{d.get('cross_attn_off', 0):.3f} ({d.get('cross_attn_off', 0)-base:+.3f})",
                f"{d.get('zero_vjepa', 0):.3f} ({d.get('zero_vjepa', 0)-base:+.3f})",
                f"{d.get('stats_noise_vjepa', 0):.3f} ({d.get('stats_noise_vjepa', 0)-base:+.3f})",
                f"{d.get('scene_swap_vjepa', 0):.3f} ({d.get('scene_swap_vjepa', 0)-base:+.3f})",
                f"{d.get('channel_mean_vjepa', 0):.3f} ({d.get('channel_mean_vjepa', 0)-base:+.3f})",
            ]
            f.write(f"| {seq} | " + " | ".join(row) + " |\n")
    print(f"wrote {md_path}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
