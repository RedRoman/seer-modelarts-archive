"""E13: does the prior's benefit actually fall with motion, or only with dataset?

The motion account so far rests on three datasets ordered by RAFT flow
(LOL 0.00, SDSD 0.18, BVI 2.63 px). Three points, and the x-axis is collinear
with everything else that distinguishes those corpora: camera, noise, capture
protocol, resolution, and whether the data is video at all. That ordering is a
hypothesis, not evidence for motion specifically.

This intervenes instead of correlating. The same BVI clips, the same
checkpoints, the same crop and metric; only the temporal stride changes. Taking
16 frames spaced s apart multiplies apparent inter-frame motion by roughly s
while holding content, sensor and noise fixed, so dataset identity is no longer
free to vary with the quantity under test.

Falsifiable in the obvious way: if the prior's benefit is flat in s, motion is
not what governs it and the three-dataset ordering was a coincidence of corpora.

  python tools/stride_test.py --strides 1 2 4 8 --out results/e13_stride.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = os.path.join(os.path.dirname(CODE), "baselines")
for p in (CODE, os.path.join(CODE, "tools"), BASELINES):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.eval_lib import _load_img_any, center_crop  # noqa: E402
from data.common import list_images  # noqa: E402
from eval_e4 import build_model, psnr_y_uint8, remap, run_clip  # noqa: E402
from temporal_metrics import _get_raft  # noqa: E402

ROOT = os.environ.get("SEER_ROOT", "/cache/zihaohe/seer_cvpr")
SPLITS = f"{ROOT}/datasets/splits_cvpr.json"


def bvi_clips(split, n_frames, crop, stride):
    """BVI clips sampled with a temporal stride, centred on the same midpoint.

    The window is centred so that every stride looks at the same part of the
    sequence; only its temporal extent, and hence the motion between adjacent
    sampled frames, changes.
    """
    sp = json.load(open(SPLITS))
    e = remap(sp["BVI_RLV"])
    span = (n_frames - 1) * stride + 1
    for n in e[split]:
        base, fps = n.split("__")
        fps = fps.replace("fps", "")
        lq_p = list_images(os.path.join(e["root"], base, f"low_light_{fps}"))
        gt_p = list_images(os.path.join(e["root"], base, f"normal_light_{fps}"))
        k = min(len(lq_p), len(gt_p))
        if k < span:                      # not enough frames for this stride
            continue
        s0 = max(0, (k - span) // 2)
        idx = [s0 + i * stride for i in range(n_frames)]
        lq = np.stack([center_crop(_load_img_any(lq_p[i]), crop) for i in idx])
        gt = np.stack([center_crop(_load_img_any(gt_p[i]), crop) for i in idx])
        yield f"BVI/{n}", lq, gt


@torch.no_grad()
def flow_of(gt_u8, dev):
    g = torch.from_numpy(gt_u8.copy()).float().div(255.).permute(0, 3, 1, 2).to(dev)
    f = _get_raft(dev)(g[:-1] * 2 - 1, g[1:] * 2 - 1)[-1]
    return float((f ** 2).sum(1).sqrt().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_pretrained", required=True)
    ap.add_argument("--ckpt_none", required=True)
    args = ap.parse_args()

    dev = torch.device(f"{args.device}:0")
    amp = torch.bfloat16
    models = {"pretrained": build_model(args.ckpt_pretrained, dev),
              "none": build_model(args.ckpt_none, dev)}

    out = {"split": args.split, "crop": args.crop, "n_frames": args.n_frames,
           "ckpt_pretrained": args.ckpt_pretrained, "ckpt_none": args.ckpt_none,
           "by_stride": {}}
    print("%6s %6s %10s %10s %10s %10s" % (
        "stride", "n", "flow(px)", "pretrained", "none", "difference"), flush=True)
    for st in args.strides:
        rows = []
        for name, lq, gt in bvi_clips(args.split, args.n_frames, args.crop, st):
            rec = {"name": name, "flow": flow_of(gt, dev)}
            for arm, (pb, bb, refiner, a, clip_len, _) in models.items():
                pred = run_clip(pb, bb, refiner, a, lq, clip_len, dev, amp)
                n = min(pred.shape[0], gt.shape[0])
                rec[arm] = float(np.mean([psnr_y_uint8(pred[j], gt[j])
                                          for j in range(n)]))
            rec["diff"] = rec["pretrained"] - rec["none"]
            rows.append(rec)
        if not rows:
            print("%6d  (no clip long enough)" % st, flush=True)
            continue
        f = float(np.mean([r["flow"] for r in rows]))
        p = float(np.mean([r["pretrained"] for r in rows]))
        q = float(np.mean([r["none"] for r in rows]))
        d = np.array([r["diff"] for r in rows])
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out["by_stride"][st] = {"n": len(rows), "flow": f, "pretrained": p,
                                "none": q, "diff": float(d.mean()),
                                "ci": [float(lo), float(hi)], "per_clip": rows}
        print("%6d %6d %10.3f %10.3f %10.3f %+10.3f [%+.2f,%+.2f]" % (
            st, len(rows), f, p, q, d.mean(), lo, hi), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    ks = sorted(out["by_stride"])
    if len(ks) > 2:
        fl = [out["by_stride"][k]["flow"] for k in ks]
        df = [out["by_stride"][k]["diff"] for k in ks]
        print("\ncorr(flow, benefit) = %+.3f over %d stride levels" % (
            np.corrcoef(fl, df)[0, 1], len(ks)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
