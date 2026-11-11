"""Is the temporal-delta term learnable, or is 1.0 just the collapse value?

E22's loss reads |dp - dg| / |dg| over tubelet first differences, and it is
settling near 1.02. That number is ambiguous on its own: a model that emits
temporally *constant* latents scores exactly 1.0, so an apparent improvement
from 1.40 to 1.02 could be the term learning the dynamics or the term killing
them. The two are separable by looking at the parts of the ratio:

  mag   |dp| / |dg|     how much temporal variation the prediction has at all
  cos   <dp, dg>        whether that variation points where the target's does

Collapse looks like mag -> 0 with td -> 1. Learning looks like mag ~ 1 with cos
rising and td falling below 1. The low-light input is scored the same way as a
floor, since whatever the input already carries is free.
"""
import argparse, json, os, sys
import numpy as np
import torch

CODE = "/cache/zihaohe/seer_cvpr/code/vjepa_llve"
for p in (CODE, os.path.join(CODE, "tools"), "/cache/zihaohe/seer_cvpr/code/baselines"):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_e4 import (build_model, held_out_sequences, run_clip, REF_LAYER)  # noqa: E402


def deltas(tok, grid):
    t_v, h_v, w_v = grid
    x = tok.reshape(tok.shape[0], t_v, h_v * w_v, -1)
    return x[:, 1:] - x[:, :-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain", default="sdsd")
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = torch.device(a.device)
    amp = torch.bfloat16
    pb, bb, refiner, exp_head, arm, clip_len, _ = build_model(a.ckpt, dev)
    if bb is None:
        raise SystemExit("this probe needs the encoder; the checkpoint has arm=none")
    grid = (clip_len // 2, a.crop // 16, a.crop // 16)

    def tok(x01):
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp):
            return bb(x01)["feats"][REF_LAYER].float()

    rows = []
    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences(a.domain, a.split, clip_len, a.crop)):
        if i >= a.clips:
            break
        pred_u8 = run_clip(pb, bb, refiner, exp_head, arm, lq, clip_len, dev, amp)
        def t(u8):
            x = torch.from_numpy(np.ascontiguousarray(u8)).float().div(255.)
            return x.permute(3, 0, 1, 2)[None].to(dev)
        dp = deltas(tok(t(pred_u8)), grid)
        dg = deltas(tok(t(gt)), grid)
        dl = deltas(tok(t(lq)), grid)
        g = dg.abs().mean().clamp_min(1e-6)
        row = dict(
            name=name,
            td_pred=float((dp - dg).abs().mean() / g),
            td_input=float((dl - dg).abs().mean() / g),
            mag_pred=float(dp.abs().mean() / g),
            mag_input=float(dl.abs().mean() / g),
            cos_pred=float(torch.nn.functional.cosine_similarity(
                dp.flatten(0, 2), dg.flatten(0, 2), dim=-1).mean()),
            cos_input=float(torch.nn.functional.cosine_similarity(
                dl.flatten(0, 2), dg.flatten(0, 2), dim=-1).mean()))
        rows.append(row)
        print("  %2d %-24s td %.3f  mag %.3f  cos %+.3f" %
              (i + 1, name[:24], row["td_pred"], row["mag_pred"], row["cos_pred"]),
              flush=True)

    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "name"}
    print("\n%s  (%d clips, %s)" % (os.path.basename(a.ckpt), len(rows), arm))
    print("            td     mag    cos")
    print("prediction  %.3f  %.3f  %+.3f" % (m["td_pred"], m["mag_pred"], m["cos_pred"]))
    print("low-light   %.3f  %.3f  %+.3f" % (m["td_input"], m["mag_input"], m["cos_input"]))
    print("zero-delta  1.000  0.000   0.000      <- what collapse scores")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(ckpt=a.ckpt, arm=arm, mean=m, per_clip=rows),
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
