"""GT-mean diagnostic for Seer-family checkpoints.

Same clips, crops and metric functions as seer_eval_unified.py. For every output
frame we additionally report the scores after rescaling the frame so that its mean
BT.601 luma equals that of the ground-truth frame ("GT-mean", as reported by several
LLIE papers): pred_gm = clip(pred * mean(Y(gt)) / mean(Y(pred)), 0, 1).
This removes the global exposure level, which is dataset-specific and not
recoverable from the input, and leaves structure / colour / noise errors.
It is a diagnostic only and uses ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/temp/zihaohe/workspace/vjepa_llve/tools")
import seer_eval_unified as E  # noqa: E402


def luma(x):  # (T,3,H,W)
    return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


def gt_mean(pred, gt):
    s = luma(gt).mean(dim=(1, 2)) / luma(pred).mean(dim=(1, 2)).clamp_min(1e-4)
    return (pred * s.view(-1, 1, 1, 1)).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_vjepa", action="store_true")
    ap.add_argument("--splits", nargs="+", default=list(E.SPLITS.keys()))
    args = ap.parse_args()
    dev = torch.device("cuda")
    model = E.Seer(args.ckpt, args.no_vjepa, dev)
    per_seq, summary = {}, {}
    for split in args.splits:
        kind, it_fn = E.SPLITS[split]
        per_seq[split] = {"HMP-raw": {}, "HMP-gtmean": {}}
        t0 = time.time()
        for name, lq_u8, gt_u8 in it_fn():
            lq = torch.from_numpy(lq_u8).float().div(255).permute(3, 0, 1, 2).unsqueeze(0).to(dev)
            gt = torch.from_numpy(gt_u8).float().div(255).permute(0, 3, 1, 2).to(dev)
            if kind == "image":
                lq = lq.repeat(1, 1, E.CLIP_LEN, 1, 1)
            lq_p, H, W = E.pad16(lq)
            pred = model(lq_p)[..., :H, :W]
            if kind == "image":
                pred = pred[:, :, E.CLIP_LEN // 2:E.CLIP_LEN // 2 + 1]
            pred_t = pred[0].permute(1, 0, 2, 3).contiguous()
            per_seq[split]["HMP-raw"][name] = E.score_frames(pred_t, gt)
            per_seq[split]["HMP-gtmean"][name] = E.score_frames(gt_mean(pred_t, gt), gt)
        summary[split] = {v: {k: float(np.mean([r[k] for r in d.values()]))
                              for k in ("psnr_rgb", "lpips", "dists", "clipiqa")}
                          for v, d in per_seq[split].items()}
        s = summary[split]
        print(f"{split:16s} raw {s['HMP-raw']['psnr_rgb']:.3f}/{s['HMP-raw']['lpips']:.4f}  "
              f"gtmean {s['HMP-gtmean']['psnr_rgb']:.3f}/{s['HMP-gtmean']['lpips']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "variant": model.variant, "doc": __doc__,
                       "per_seq": per_seq, "summary": summary}, f, indent=1)
    print("[SAVE]", args.out)


if __name__ == "__main__":
    main()
