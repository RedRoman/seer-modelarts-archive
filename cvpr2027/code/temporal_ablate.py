"""E14: is there any temporal component in what the prior supplies?

The stride intervention failed as a controlled test: widening the window also
made the task easier (input PSNR rose 11.06 -> 11.58 dB across strides), so
motion was still confounded, this time with difficulty rather than with dataset.

This removes that confound by never changing which frames are involved. The same
16 frames are fed to the model, and only their temporal arrangement changes:

  intact    the frames in order                      (reference)
  shuffled  the same frames in a random order        (temporal structure broken,
                                                      frame content identical)
  frozen    the centre frame repeated 16 times       (no temporal information)

Scoring is always against the intact ground truth for the frames the model was
given, so the target and the per-frame difficulty are pixel-identical across
conditions. Only the temporal coherence of the input differs.

The prediction that distinguishes the two accounts: if the prior carries
temporal structure, breaking it should hurt the pretrained arm more than the
arm that never had tokens. If the prior is spatial-semantic, both arms should
degrade by a similar amount.

For shuffled we score frame-wise against the correspondingly shuffled ground
truth, so a model that simply enhances each frame independently is not penalised
for the shuffling itself.
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

from eval_e4 import build_model, held_out_sequences, psnr_y_uint8, run_clip  # noqa: E402

ROOT = os.environ.get("SEER_ROOT", "/cache/zihaohe/seer_cvpr")


def conditions(lq, gt, rng):
    """(name, lq_in, gt_ref) triples that hold frame content fixed."""
    t = lq.shape[0]
    yield "intact", lq, gt
    perm = rng.permutation(t)
    yield "shuffled", lq[perm], gt[perm]
    mid = t // 2
    yield "frozen", np.repeat(lq[mid:mid + 1], t, axis=0), \
          np.repeat(gt[mid:mid + 1], t, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bvi", choices=["bvi", "sdsd"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_pretrained", required=True)
    ap.add_argument("--ckpt_none", required=True)
    args = ap.parse_args()

    dev = torch.device(f"{args.device}:0")
    amp = torch.bfloat16
    models = {"pretrained": build_model(args.ckpt_pretrained, dev),
              "none": build_model(args.ckpt_none, dev)}
    rng = np.random.default_rng(args.seed)

    recs = []
    for name, lq, gt, kind in held_out_sequences(
            args.domain, args.split, args.n_frames, args.crop):
        if kind != "video":
            continue
        rec = {"name": name}
        for cond, lq_in, gt_ref in conditions(lq, gt, rng):
            for arm, (pb, bb, refiner, a, clip_len, _) in models.items():
                pred = run_clip(pb, bb, refiner, a, lq_in, clip_len, dev, amp)
                k = min(pred.shape[0], gt_ref.shape[0])
                rec[f"{cond}/{arm}"] = float(np.mean(
                    [psnr_y_uint8(pred[j], gt_ref[j]) for j in range(k)]))
        recs.append(rec)
        print("  %-28s intact p/n %.2f/%.2f  shuffled %.2f/%.2f  frozen %.2f/%.2f" % (
            name.split("/")[-1], rec["intact/pretrained"], rec["intact/none"],
            rec["shuffled/pretrained"], rec["shuffled/none"],
            rec["frozen/pretrained"], rec["frozen/none"]), flush=True)

    print("\n%-10s %12s %12s %12s" % ("condition", "pretrained", "none", "prior effect"))
    agg = {}
    for cond in ("intact", "shuffled", "frozen"):
        p = np.array([r[f"{cond}/pretrained"] for r in recs])
        q = np.array([r[f"{cond}/none"] for r in recs])
        agg[cond] = {"pretrained": float(p.mean()), "none": float(q.mean()),
                     "diff": float((p - q).mean())}
        print("%-10s %12.3f %12.3f %+12.3f" % (cond, p.mean(), q.mean(), (p - q).mean()))

    print("\n%-10s %14s %14s" % ("", "pretrained drop", "none drop"))
    rng2 = np.random.default_rng(1)
    for cond in ("shuffled", "frozen"):
        dp = np.array([r[f"{cond}/pretrained"] - r["intact/pretrained"] for r in recs])
        dq = np.array([r[f"{cond}/none"] - r["intact/none"] for r in recs])
        gap = dp - dq
        bs = np.array([rng2.choice(gap, len(gap), replace=True).mean()
                       for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "prior-specific" if hi < 0 else ("none-specific" if lo > 0 else "n.s.")
        agg[cond]["drop_pretrained"] = float(dp.mean())
        agg[cond]["drop_none"] = float(dq.mean())
        agg[cond]["gap_ci"] = [float(lo), float(hi)]
        print("%-10s %14.3f %14.3f   gap %+.3f [%+.3f,%+.3f]  %s" % (
            cond, dp.mean(), dq.mean(), gap.mean(), lo, hi, verdict))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"domain": args.domain, "split": args.split, "n": len(recs),
               "aggregate": agg, "per_clip": recs}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
