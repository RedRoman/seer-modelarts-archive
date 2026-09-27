"""Extended eval harness that adds DISTS + CLIPIQA on top of eval_lib.

Same protocol (video: center T=16 512x512, images: full-image). PSNR uint8 RGB,
LPIPS + DISTS + CLIPIQA via pyiqa. For very large frames, DISTS/CLIPIQA can be
tile-averaged with the 512 crop already applied by the loader so a single 512x512
call fits in memory easily on 80GB A100.

The evaluate_method_v2 function writes per-seq JSON entries with keys
{psnr_rgb, lpips, dists, clipiqa, n_frames} and per-dataset summary with
{psnr_rgb, lpips, dists, clipiqa, n_seqs}.
"""
from __future__ import annotations
import os, sys, json, time
from typing import Callable, Dict, List, Optional
import numpy as np
import torch

# Reuse everything from the original library.
from common.eval_lib import (
    DATASET_ITERS,
    rgb_psnr_uint8,
    get_lpips_model,
    lpips_batch,
)


_DISTS_MODEL = None
_CLIPIQA_MODEL = None


def get_dists_model(device):
    global _DISTS_MODEL
    if _DISTS_MODEL is None:
        import pyiqa
        _DISTS_MODEL = pyiqa.create_metric("dists", device=device, as_loss=False).eval()
    return _DISTS_MODEL


def get_clipiqa_model(device):
    global _CLIPIQA_MODEL
    if _CLIPIQA_MODEL is None:
        import pyiqa
        _CLIPIQA_MODEL = pyiqa.create_metric("clipiqa", device=device, as_loss=False).eval()
    return _CLIPIQA_MODEL


@torch.no_grad()
def dists_batch(pred_01: torch.Tensor, gt_01: torch.Tensor) -> float:
    """pred/gt: (B, 3, H, W) float [0,1]. Full-reference DISTS."""
    dev = pred_01.device if pred_01.is_cuda else torch.device("cuda")
    if not pred_01.is_cuda:
        pred_01 = pred_01.to(dev); gt_01 = gt_01.to(dev)
    m = get_dists_model(dev)
    v = m(pred_01.clamp(0, 1), gt_01.clamp(0, 1))
    return float(v.mean().item())


@torch.no_grad()
def clipiqa_batch(pred_01: torch.Tensor) -> float:
    """pred: (B, 3, H, W) float [0,1]. No-reference CLIP-IQA."""
    dev = pred_01.device if pred_01.is_cuda else torch.device("cuda")
    if not pred_01.is_cuda:
        pred_01 = pred_01.to(dev)
    m = get_clipiqa_model(dev)
    v = m(pred_01.clamp(0, 1))
    return float(v.mean().item())


def evaluate_method_v2(
    method_name: str,
    infer_fn: Callable[[torch.Tensor], torch.Tensor],
    datasets: Optional[List[str]] = None,
    out_path: str = "/cache/zihaohe/baseline_ckpts/results_v2.json",
    notes: str = "",
    device=None,
    n_frames: int = 16,
    crop: int = 512,
    limit_seqs: Optional[int] = None,
    dtype=torch.float32,
):
    """
    Same as evaluate_method but also computes DISTS (FR) + CLIPIQA (NR).
    Per-seq schema: {psnr_rgb, lpips, dists, clipiqa, n_frames}
    Per-dataset summary: {psnr_rgb, lpips, dists, clipiqa, n_seqs}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if datasets is None:
        datasets = list(DATASET_ITERS.keys())

    per_seq: Dict[str, Dict[str, float]] = {}
    summary: Dict[str, Dict[str, float]] = {}

    for ds_name in datasets:
        kind, iter_fn = DATASET_ITERS[ds_name]
        if kind == "video":
            iter_ = iter_fn(n_frames=n_frames, crop=crop)
        else:
            iter_ = iter_fn()
        psnrs_ds: List[float] = []; lpips_ds: List[float] = []
        dists_ds: List[float] = []; clip_ds: List[float] = []
        print(f"\n=== {method_name} / {ds_name} ({kind}) ===", flush=True)
        t0 = time.time()
        for i, (seq_name, lq_arr, gt_arr) in enumerate(iter_):
            if limit_seqs is not None and i >= limit_seqs: break
            T = lq_arr.shape[0]
            ps_seq: List[float] = []; lp_seq: List[float] = []
            di_seq: List[float] = []; ci_seq: List[float] = []
            for t in range(T):
                lq_t = torch.from_numpy(lq_arr[t]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                gt_t = torch.from_numpy(gt_arr[t]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                if dtype != torch.float32:
                    lq_t = lq_t.to(dtype)
                out = infer_fn(lq_t)
                out = out.clamp(0, 1).float()
                out_u8 = (out[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                gt_u8 = (gt_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                ps_seq.append(rgb_psnr_uint8(out_u8, gt_u8))
                lp_seq.append(lpips_batch(out.float(), gt_t.float()))
                di_seq.append(dists_batch(out.float(), gt_t.float()))
                ci_seq.append(clipiqa_batch(out.float()))
            mp = float(np.mean(ps_seq)); ml = float(np.mean(lp_seq))
            md = float(np.mean(di_seq)); mc = float(np.mean(ci_seq))
            key = f"{ds_name}__{seq_name}"
            per_seq[key] = {"psnr_rgb": mp, "lpips": ml, "dists": md, "clipiqa": mc, "n_frames": T}
            psnrs_ds.append(mp); lpips_ds.append(ml); dists_ds.append(md); clip_ds.append(mc)
            print(f"  [{i+1:3d}] {seq_name}: PSNR={mp:.3f} LPIPS={ml:.4f} DISTS={md:.4f} CLIPIQA={mc:.4f}  ({T}f)", flush=True)
        if psnrs_ds:
            summary[ds_name] = {
                "psnr_rgb": float(np.mean(psnrs_ds)),
                "lpips":    float(np.mean(lpips_ds)),
                "dists":    float(np.mean(dists_ds)),
                "clipiqa":  float(np.mean(clip_ds)),
                "n_seqs":   len(psnrs_ds),
            }
        else:
            summary[ds_name] = {"psnr_rgb": float("nan"), "lpips": float("nan"),
                                "dists": float("nan"), "clipiqa": float("nan"), "n_seqs": 0}
        dt = time.time() - t0
        s = summary[ds_name]
        print(f"  MEAN {ds_name}: PSNR={s['psnr_rgb']:.3f} LPIPS={s['lpips']:.4f} "
              f"DISTS={s['dists']:.4f} CLIPIQA={s['clipiqa']:.4f}  ({dt:.1f}s, {s['n_seqs']} seqs)",
              flush=True)

    out = {
        "method": method_name,
        "notes": notes,
        "protocol": f"video: center T={n_frames}, {crop}x{crop} center-crop; images: full-image. "
                    f"PSNR uint8 RGB, LPIPS/DISTS/CLIPIQA pyiqa.",
        "per_seq": per_seq,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {out_path}", flush=True)
    return out
