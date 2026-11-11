"""Frames for the qualitative figure, from the same evaluator as the tables.

The ICLR-era `dump_qualitative.py` runs the old 512-crop stack with the old
baselines and the pre-de-leak split, so its numbers would not match this
paper's tables. This one goes through `eval_e4` and `eval_baseline` so a number
printed under a crop is the same number the table reports for that sequence.

Sequences are chosen by a stated rule rather than by eye: the test sequences at
the 10th, 35th, 60th and 85th percentile of our margin over the retrained
Retinexformer. That deliberately includes a sequence we lose on, because the
margin's median is near zero and a figure made of wins would misrepresent the
distribution the paper reports.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

CODE = "/cache/zihaohe/seer_cvpr/code/vjepa_llve"
for p in (CODE, os.path.join(CODE, "tools"),
          "/cache/zihaohe/seer_cvpr/code/baselines"):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_e4 import (build_model, held_out_sequences, run_clip,  # noqa: E402
                     psnr_y_uint8, lpips_of)
import eval_baseline as EB  # noqa: E402

PCTS = (10, 35, 60, 85)


def save(arr_u8, path):
    Image.fromarray(arr_u8).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours", required=True, help="converged prior checkpoint")
    ap.add_argument("--none", required=True, help="converged no-prior checkpoint")
    ap.add_argument("--retinexformer", required=True, help="retrained .pth")
    ap.add_argument("--margins", required=True,
                    help="JSON with our and Retinexformer per-sequence psnr_y")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    amp = torch.bfloat16

    M = json.load(open(a.margins))
    order = sorted(M.items(), key=lambda kv: kv[1])
    picks = []
    for pc in PCTS:
        i = int(round(pc / 100 * (len(order) - 1)))
        if order[i][0] not in picks:
            picks.append(order[i][0])
    print("picked by margin percentile %s:" % (PCTS,))
    for n in picks:
        print("  %-28s margin %+.3f dB" % (n, M[n]))

    models = {}
    for tag, ck in (("ours", a.ours), ("none", a.none)):
        models[tag] = build_model(ck, dev)
    rf_model, rf_mult = EB.LOADERS["retinexformer"](a.retinexformer, dev)

    rows = {}
    for name, lq, gt, kind in held_out_sequences("sdsd", "test", 16, a.crop):
        if name not in picks:
            continue
        mid = lq.shape[0] // 2
        out = {"lq": lq, "gt": gt}
        for tag in ("ours", "none"):
            pb, bb, refiner, exp_head, arm, clip_len, _ = models[tag]
            out[tag] = run_clip(pb, bb, refiner, exp_head, arm, lq,
                                clip_len, dev, amp)
        out["retinexformer"] = EB.run_frames(rf_model, rf_mult, lq, dev)

        rows[name] = {}
        safe = name.replace("/", "__")
        for tag, img in out.items():
            f = img[mid]
            save(f, os.path.join(a.out, f"{safe}__{tag}.png"))
            if tag != "gt":
                # lpips_of wants (B,3,H,W) float tensors in [0,1], the same
                # form eval_e4 feeds it, so the figure's numbers and the
                # table's come from identical inputs
                def to_t(u8):
                    return (torch.from_numpy(np.ascontiguousarray(u8[None]))
                            .permute(0, 3, 1, 2).float().div(255.))
                rows[name][tag] = {
                    "psnr_y": psnr_y_uint8(f, gt[mid]),
                    "lpips": lpips_of(to_t(f), to_t(gt[mid]), dev)}
        print("%-28s " % name + "  ".join(
            "%s %.2f/%.3f" % (t, v["psnr_y"], v["lpips"])
            for t, v in rows[name].items()))

    meta = {"crop": a.crop, "frame": "middle of the centre 16-frame clip",
            "selection": "test sequences at margin percentiles %s over the "
                         "retrained Retinexformer" % (PCTS,),
            "picked": picks, "per_image": rows}
    json.dump(meta, open(os.path.join(a.out, "qualitative.json"), "w"), indent=1)
    print("\nwrote", os.path.join(a.out, "qualitative.json"))


if __name__ == "__main__":
    main()
