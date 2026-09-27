"""Per-clip difficulty statistics for stratified analysis (darkness, motion, colour cast).

For every evaluated clip (same iterators and crops as seer_eval_unified.py):
  lq_luma     mean BT.601 luma of the input
  gt_luma     mean luma of the ground truth
  gain        gt_luma / lq_luma (how much brightening is needed)
  motion      mean RAFT flow magnitude on the GT clip (pixels / frame, 512 crop); video only
  lq_cast     |mean(B) - mean(R)| / mean luma of the input (colour cast proxy)
Output: /temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals/clip_stats.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/temp/zihaohe/workspace/vjepa_llve/tools")
import seer_eval_unified as E  # noqa: E402
from temporal_metrics import gt_flows  # noqa: E402

OUT = "/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals/clip_stats.json"


def luma(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def main():
    dev = torch.device("cuda")
    out = {}
    for split, (kind, it_fn) in E.SPLITS.items():
        out[split] = {}
        for name, lq_u8, gt_u8 in it_fn():
            lq = lq_u8.astype(np.float32) / 255.0
            gt = gt_u8.astype(np.float32) / 255.0
            lql, gtl = float(luma(lq).mean()), float(luma(gt).mean())
            rec = {"lq_luma": lql, "gt_luma": gtl, "gain": gtl / max(lql, 1e-4),
                   "lq_cast": float(abs(lq[..., 2].mean() - lq[..., 0].mean()) / max(lql, 1e-4))}
            if kind == "video":
                g = torch.from_numpy(gt).permute(0, 3, 1, 2).to(dev)
                with torch.no_grad():
                    fwd, mask = gt_flows(g)
                rec["motion"] = float(fwd.norm(dim=1).mean())
            out[split][name] = rec
        print(split, len(out[split]), flush=True)
    with open(OUT, "w") as f:
        json.dump({"doc": __doc__, "per_seq": out}, f, indent=1)
    print("[SAVE]", OUT)


if __name__ == "__main__":
    main()
