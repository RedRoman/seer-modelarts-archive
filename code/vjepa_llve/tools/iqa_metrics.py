"""Shared IQA metrics for V-JEPA-LLVE eval (LPIPS / DISTS / SSIM / CLIP-IQA).

All trainer eval blocks call compute_iqa(pred, gt). Metric models are lazily
created and cached so the per-iter cost is just the forward pass.

Tensor convention (matches tools/v11_oracle_delta_probe.load_clip):
  pred / gt: (B=1, 3, T, H, W) float in [0,1] on CUDA.
"""
from __future__ import annotations
import torch

_CACHE = {}


def _get(name, device):
    import pyiqa
    key = (name, str(device))
    if key not in _CACHE:
        _CACHE[key] = pyiqa.create_metric(name, device=device, as_loss=False).eval()
    return _CACHE[key]


def _to_iqa(t):
    """(B,3,T,H,W) float[0,1] -> (B*T, 3, H, W) clamped."""
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    if t.dim() == 5:
        b, c, T, h, w = t.shape
        t = t.permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
    return t.clamp(0, 1)


@torch.no_grad()
def compute_iqa(pred, gt, device=None,
                metrics=("ssim", "lpips", "dists", "clipiqa")):
    """Run FR (ssim/lpips/dists) + NR (clipiqa) metrics. Returns {name: float}."""
    if device is None:
        device = pred.device if pred.is_cuda else torch.device("cuda")
    p = _to_iqa(pred).to(device)
    g = _to_iqa(gt).to(device)
    fr = {"ssim", "lpips", "dists", "topiq_fr", "ahiq"}
    out = {}
    for m in metrics:
        fn = _get(m, device)
        v = fn(p, g) if m in fr else fn(p)
        out[m] = float(v.mean().item())
    return out


def fmt_iqa(d):
    parts = []
    for k in ("ssim", "lpips", "dists", "clipiqa"):
        if k in d:
            parts.append(f"{k.upper()}={d[k]:.4f}")
    return " ".join(parts)


def save_compare_frame(out_path, lq, pred, gt, frame_indices=(0, 7, 15), labels=None):
    """Save a side-by-side LQ | Pred | GT visualization grid as PNG.

    Args:
      out_path: target PNG path (parent dirs created).
      lq, pred, gt: tensors (B=1, 3, T, H, W) float [0,1] (or uint8).
      frame_indices: which time indices to render as rows.
      labels: optional list of column labels (default: LQ, PRED, GT).
    """
    import os
    import numpy as np
    from PIL import Image

    def grab(t, fi):
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        a = t[0, :, fi].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (a * 255).astype(np.uint8)

    pad_w = 8
    rows = []
    for fi in frame_indices:
        if fi >= lq.shape[2]:
            continue
        a, b, c = grab(lq, fi), grab(pred, fi), grab(gt, fi)
        H, W, _ = a.shape
        sep = np.full((H, pad_w, 3), 128, dtype=np.uint8)
        rows.append(np.concatenate([a, sep, b, sep, c], axis=1))
    H, W, _ = rows[0].shape
    gap = np.full((pad_w, W, 3), 128, dtype=np.uint8)
    grid_parts = []
    for i, r in enumerate(rows):
        if i > 0:
            grid_parts.append(gap)
        grid_parts.append(r)
    grid = np.concatenate(grid_parts, axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(grid).save(out_path)


def save_compare_video(out_path, lq, pred, gt, fps=10):
    """Save MP4 of side-by-side LQ|PRED|GT for ALL frames.

    Args:
      out_path: target .mp4 path (parent dirs created).
      lq, pred, gt: tensors (B=1, 3, T, H, W) float [0,1] (or uint8).
      fps: video framerate.
    """
    import os
    import numpy as np
    import imageio.v2 as imageio

    def pp(t, fi):
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        a = t[0, :, fi].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (a * 255).astype(np.uint8)

    T = lq.shape[2]
    pad_w = 8
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    for fi in range(T):
        a, b, c = pp(lq, fi), pp(pred, fi), pp(gt, fi)
        H = a.shape[0]
        sep = np.full((H, pad_w, 3), 128, dtype=np.uint8)
        frame = np.concatenate([a, sep, b, sep, c], axis=1)
        writer.append_data(frame)
    writer.close()


def write_eval_md(path, header, body_lines, mean_psnr, mean_iqa, baseline_line=None):
    """Standard EVAL.md writer. body_lines is a list of strings rendered before metrics."""
    with open(path, "w") as f:
        f.write(header.rstrip() + "\n\n")
        for ln in body_lines:
            f.write(ln.rstrip() + "\n")
        if body_lines:
            f.write("\n")
        f.write("## SDSD held-out\n\n")
        f.write(f"- PSNR    = {mean_psnr:.3f}\n")
        for k in ("ssim", "lpips", "dists", "clipiqa"):
            if k in mean_iqa:
                f.write(f"- {k.upper():<7} = {mean_iqa[k]:.4f}\n")
        if baseline_line:
            f.write(f"\n{baseline_line}\n")
