"""How much of the token field does the pixel branch actually need?

Every architectural question asked so far has been about *how* the tokens are
delivered -- positional grounding, cross-frame keys, multi-layer keys, exposure
decoupling, a trust gate -- and all five land within noise or worse at
convergence. Meanwhile *what is in* the tokens is decisive: random tokens cost
4.8 dB against correct ones. That asymmetry says the open question is not
delivery but content, and the content question has never been asked: of a
2048 x 1024 field per clip, how much is load-bearing?

This applies rank- and structure-reducing transforms to the tokens at inference
on a model trained with the full field:

  identity        control
  spatial mean    each tubelet's tokens replaced by their spatial mean,
                  broadcast back -- removes the spatial layout, keeps semantics
  temporal mean   each spatial position averaged over tubelets, broadcast back
                  -- removes what little temporal variation is used
  rank k          projection onto the top-k principal directions of this clip's
                  own tokens, k in {4, 16, 64, 256} of 1024
  random          control, known to cost 4.8 dB

The reading is one-sided and that is stated rather than finessed: a reduction
that loses nothing proves the discarded part was surplus, because the model is
demonstrably sensitive to token content. A reduction that loses a lot is
ambiguous between "the information was needed" and "the model is brittle to a
shift in its conditioning it never trained on", and settling that needs a
retrain, which this screen is meant to decide whether to spend.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

CODE = "/cache/zihaohe/seer_cvpr/code/vjepa_llve"
for p in (CODE, os.path.join(CODE, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
from eval_e4 import (build_model, held_out_sequences, psnr_y_uint8,  # noqa: E402
                     lpips_of, REF_LAYER)
import eval_e4 as E4  # noqa: E402

GRID_T, GRID_H, GRID_W = 8, 16, 16


def spatial_mean(z):
    b, n, d = z.shape
    x = z.reshape(b, GRID_T, GRID_H * GRID_W, d)
    return x.mean(2, keepdim=True).expand_as(x).reshape(b, n, d)


def temporal_mean(z):
    b, n, d = z.shape
    x = z.reshape(b, GRID_T, GRID_H * GRID_W, d)
    return x.mean(1, keepdim=True).expand_as(x).reshape(b, n, d)


def low_rank(z, k):
    b, n, d = z.shape
    x = z.reshape(-1, d)
    mu = x.mean(0, keepdim=True)
    xc = (x - mu).float()
    # economy SVD on (N, D); k << D so this is cheap
    _, _, V = torch.linalg.svd(xc, full_matrices=False)
    Vk = V[:k]
    return ((xc @ Vk.T) @ Vk + mu).reshape(b, n, d).to(z.dtype)


TRANSFORMS = {
    "identity":      lambda z: z,
    "spatial mean":  spatial_mean,
    "temporal mean": temporal_mean,
    "rank 4":        lambda z: low_rank(z, 4),
    "rank 16":       lambda z: low_rank(z, 16),
    "rank 64":       lambda z: low_rank(z, 64),
    "rank 256":      lambda z: low_rank(z, 256),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", type=int, default=39)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device(a.device)
    amp = torch.bfloat16
    pb, bb, refiner, exp_head, arm, clip_len, _ = build_model(a.ckpt, dev)
    if bb is None:
        raise SystemExit("needs the encoder")

    acc = {k: {"psnr": [], "lpips": []} for k in TRANSFORMS}
    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences("sdsd", "test", clip_len, a.crop)):
        if i >= a.clips:
            break
        x = torch.from_numpy(np.ascontiguousarray(lq)).float().div(255.)
        x = x.permute(3, 0, 1, 2)[None].to(dev)
        g = torch.from_numpy(np.ascontiguousarray(gt)).float().div(255.)
        g = g.permute(3, 0, 1, 2)[None].to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp):
            zl = bb(x)["feats"]
        zl = [z.float() for z in zl]
        for tname, fn in TRANSFORMS.items():
            zp = [fn(z) for z in zl]
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp):
                out = pb(x, zp)
            pred = out["pred_pixels"].float().clamp(0, 1)
            pu8 = (pred[0].permute(1, 2, 3, 0).cpu().numpy() * 255
                   ).round().astype(np.uint8)
            acc[tname]["psnr"].append(float(np.mean(
                [psnr_y_uint8(pu8[j], gt[j]) for j in range(pu8.shape[0])])))
            acc[tname]["lpips"].append(lpips_of(
                pred[0].permute(1, 0, 2, 3), g[0].permute(1, 0, 2, 3), dev))
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, a.clips), flush=True)

    base_p = float(np.mean(acc["identity"]["psnr"]))
    base_l = float(np.mean(acc["identity"]["lpips"]))
    print("\ntoken sufficiency, SDSD test, %d clips, inference-time only" % a.clips)
    print("%-14s %8s %8s   %8s %8s" % ("transform", "PSNR-Y", "d", "LPIPS", "d"))
    res = {}
    for t in TRANSFORMS:
        p = float(np.mean(acc[t]["psnr"])); l = float(np.mean(acc[t]["lpips"]))
        res[t] = {"psnr_y": p, "lpips": l, "d_psnr": p - base_p, "d_lpips": l - base_l}
        print("%-14s %8.3f %+8.3f   %8.4f %+8.4f" % (t, p, p - base_p, l, l - base_l))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        # per-clip values so the differences can be paired and bootstrapped;
        # a mean on its own is the kind of number this project has been told
        # not to report
        json.dump({"ckpt": a.ckpt, "clips": a.clips, "result": res,
                   "per_clip": {t: acc[t] for t in TRANSFORMS}},
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
