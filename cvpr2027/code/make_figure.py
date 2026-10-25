"""Side-by-side qualitative comparison on SDSD-val.

Columns: input, Retinexformer retrained on our split, Seer-v2 without the
V-JEPA tokens, Seer-v2 with them, ground truth. The two Seer columns differ in
the prior alone, so the figure shows the same ablation the numbers report.

Everything runs through the evaluation path used for the tables: crop 256,
centre 16 frames, the frozen split lists.

  python tools/make_figure.py --out /tmp/compare.png --seqs 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = os.path.join(os.path.dirname(CODE), "baselines")
for p in (CODE, os.path.join(CODE, "tools"), BASELINES):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_e4 import build_model, held_out_sequences, psnr_y_uint8, run_clip  # noqa: E402
from eval_baseline import LOADERS, run_frames  # noqa: E402

ROOT = os.environ.get("SEER_ROOT", "/cache/zihaohe/seer_cvpr")
RUNS = f"{ROOT}/runs_cuda"
RF = (f"{ROOT}/weights_small/checkpoint/runs/retinexformer_seer_protocol/"
      f"experiments/RetinexFormer_SDSD_union_seer_protocol/models/net_g_160000.pth")


def label_strip(width, text, height=26):
    """A white caption bar with the text drawn as coarse pixels, no font deps."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((6, 6), text, fill=(0, 0, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/compare.png")
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--frame", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
    dev = torch.device(f"{args.device}:0")
    amp = torch.bfloat16

    models = {}
    for name, tag in (("ours_full", "v2_pretrained_s0"), ("ours_novjepa", "v2_none_s0")):
        ck = f"{RUNS}/{tag}/final.pt"
        if not os.path.exists(ck):
            ck = f"{RUNS}/{tag}/step_2000.pt"
        models[name] = build_model(ck, dev)
        print(f"loaded {name} from {os.path.basename(ck)}", flush=True)
    rf_model, rf_mult = LOADERS["retinexformer"](RF, dev)
    print("loaded retinexformer", flush=True)

    rows, titles = [], ["Input", "Retinexformer", "Ours w/o V-JEPA", "Ours", "Ground truth"]
    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences("sdsd", "val", 16, args.crop)):
        if i >= args.seqs:
            break
        f = min(args.frame, lq.shape[0] - 1)
        panels = [lq[f]]

        pred_rf = run_frames(rf_model, rf_mult, lq, dev)
        panels.append(pred_rf[f])

        for key in ("ours_novjepa", "ours_full"):
            pb, bb, refiner, arm, clip_len, _ = models[key]
            pred = run_clip(pb, bb, refiner, arm, lq, clip_len, dev, amp)
            panels.append(pred[f])

        panels.append(gt[f])
        p_rf = psnr_y_uint8(pred_rf[f], gt[f])
        print(f"  {name}: Retinexformer {p_rf:.2f} dB", flush=True)
        rows.append(np.concatenate(panels, axis=1))

    grid = np.concatenate(rows, axis=0)
    strip = np.concatenate(
        [label_strip(args.crop, t) for t in titles], axis=1)
    grid = np.concatenate([strip, grid], axis=0)

    from PIL import Image
    Image.fromarray(grid).save(args.out)
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
