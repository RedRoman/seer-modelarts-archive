"""Per-clip versus fixed-basis token projection, on the same model and clips.

Amendment 19 rests on one contrast: four of each clip's own principal
directions preserve the prior, four fixed directions estimated on the training
split destroy it. The fixed-basis number was measured on one seed whose JSON
lives on a host that is now unreachable, so this re-measures the contrast on
the two converged seeds held elsewhere, with both projections computed in the
same pass so they share every other condition.

Inference-time only, on models trained with the full token field. Per-clip
values are saved so differences can be paired over sequences.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

CODE = os.environ.get("SEER_CODE", "/temp/zihaohe/seer_cvpr/code/vjepa_llve")
for p in (CODE, os.path.join(CODE, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
from eval_e4 import build_model, held_out_sequences, psnr_y_uint8, lpips_of  # noqa: E402


def clip_rank(z, k):
    b, n, d = z.shape
    x = z.reshape(-1, d)
    mu = x.mean(0, keepdim=True)
    xc = (x - mu).float()
    V = torch.linalg.svd(xc.cpu(), full_matrices=False)[2][:k].to(z.device)
    return ((xc @ V.T) @ V + mu).reshape(b, n, d)


def global_rank(z, k, mu, comp):
    b, n, d = z.shape
    x = z.reshape(-1, d) - mu
    V = comp[:k]
    return ((x @ V.T) @ V + mu).reshape(b, n, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--clips", type=int, default=39)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device.startswith("npu"):
        import torch_npu  # noqa: F401
    dev = torch.device(a.device)
    amp = torch.bfloat16

    pb, bb, refiner, exp_head, arm, clip_len, _ = build_model(a.ckpt, dev)
    bj = json.load(open(a.basis))
    mus = [torch.tensor(m, device=dev).float() for m in bj["mean"]]
    comps = [torch.tensor(c, device=dev).float() for c in bj["components"]]

    conds = {"identity": None}
    for k in (4, 16, 64):
        conds[f"clip rank {k}"] = ("clip", k)
        conds[f"global rank {k}"] = ("global", k)
    acc = {c: {"psnr": [], "lpips": []} for c in conds}

    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences("sdsd", "test", clip_len, a.crop)):
        if i >= a.clips:
            break
        x = torch.from_numpy(np.ascontiguousarray(lq)).float().div(255.)
        x = x.permute(3, 0, 1, 2)[None].to(dev)
        g = torch.from_numpy(np.ascontiguousarray(gt)).float().div(255.)
        g = g.permute(3, 0, 1, 2)[None].to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp):
            zl = [z.float() for z in bb(x)["feats"]]
        for c, spec in conds.items():
            if spec is None:
                zp = zl
            elif spec[0] == "clip":
                zp = [clip_rank(z, spec[1]) for z in zl]
            else:
                zp = [global_rank(z, spec[1], mus[li], comps[li])
                      for li, z in enumerate(zl)]
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp):
                pred = pb(x, zp)["pred_pixels"].float().clamp(0, 1)
            pu8 = (pred[0].permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)
            acc[c]["psnr"].append(float(np.mean(
                [psnr_y_uint8(pu8[j], gt[j]) for j in range(pu8.shape[0])])))
            acc[c]["lpips"].append(lpips_of(
                pred[0].permute(1, 0, 2, 3), g[0].permute(1, 0, 2, 3), dev))
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, a.clips), flush=True)

    base = np.array(acc["identity"]["psnr"])
    print("\n%s  (%d clips)" % (os.path.basename(os.path.dirname(a.ckpt)), len(base)))
    print("%-16s %8s %8s" % ("projection", "PSNR-Y", "d"))
    for c in conds:
        v = np.array(acc[c]["psnr"])
        print("%-16s %8.3f %+8.3f" % (c, v.mean(), (v - base).mean()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"ckpt": a.ckpt, "basis": a.basis, "per_clip": acc},
              open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
