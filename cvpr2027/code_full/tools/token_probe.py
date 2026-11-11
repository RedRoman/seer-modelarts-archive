"""Do V-JEPA's tokens carry temporal structure on this data at all?

Our converged measurements say the prior behaves as single-frame semantics:
shuffling the input frames changes the output by 0.05 dB, and freezing the clip
to one repeated frame costs the prior arm 0.76 dB against the no-prior arm's
2.61. Two explanations survive that, and they call for different responses:

  A  the encoder does not encode cross-frame structure on low-light,
     near-static footage -- there is nothing for any model to use
  B  it does encode it, and our model has learned to ignore it, because the
     training objective is per-frame reconstruction and nothing rewards
     temporal use

This looks at the tokens directly, before any pixel branch is involved, so the
two are separable. Three quantities per tapped block:

  across-tubelet  mean cosine distance between spatially matched tokens of
                  different tubelets. If a clip's tokens hardly vary over time,
                  the encoder is not representing motion
  vs-frozen       how much the across-tubelet variation drops when the clip is
                  replaced by one frame repeated. A large drop means the
                  variation in the intact clip was genuinely temporal rather
                  than positional
  shuffle-delta   relative change of the token tensor when the input frames are
                  randomly reordered. Near zero means the encoder's output is
                  close to order-invariant on this data

Each is reported for the low-light input and for the ground truth, since a
further possibility is that the features are fine on well-exposed video and
degrade in the dark.
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

from eval_e4 import held_out_sequences  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402

ROOT = os.environ.get("SEER_ROOT", "/cache/zihaohe/seer_cvpr")
TAPS = (5, 11, 17, 23)


def tokens(bb, clip_u8, dev, amp):
    x = torch.from_numpy(clip_u8.copy()).float().div(255.)
    x = x.permute(3, 0, 1, 2)[None].to(dev)
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp):
        feats = bb(x)["feats"]
    return [f.float() for f in feats]


def across_tubelet(tok, grid):
    """Mean cosine distance between spatially matched tokens of different tubelets."""
    t_v, h_v, w_v = grid
    x = tok.reshape(1, t_v, h_v * w_v, -1)[0]          # (T, HW, D)
    x = torch.nn.functional.normalize(x, dim=-1)
    sims = []
    for i in range(t_v):
        for j in range(i + 1, t_v):
            sims.append((x[i] * x[j]).sum(-1).mean())
    return float(1.0 - torch.stack(sims).mean())


def rel_delta(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="sdsd", choices=["sdsd", "bvi"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
    dev = torch.device(f"{args.device}:0")
    amp = torch.bfloat16
    grid = (8, args.crop // 16, args.crop // 16)

    bb = VJepaBackboneLocal21(
        local_repo=f"{ROOT}/modelzoo/vjepa2-main",
        local_ckpt=f"{ROOT}/weights/vjepa2_1_vitl_dist_vitG_384.pt",
        feature_layers=TAPS, image_size=args.crop, frames_per_clip=16,
        patch_size=16, tubelet_size=2, freeze=True).to(dev).eval()

    rng = np.random.default_rng(0)
    acc = {src: {li: {"across": [], "frozen": [], "shuf": []} for li in TAPS}
           for src in ("lq", "gt")}

    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences(args.domain, args.split, 16, args.crop)):
        if kind != "video" or i >= args.clips:
            if i >= args.clips:
                break
            continue
        for src, clip in (("lq", lq), ("gt", gt)):
            t_in = clip.shape[0]
            perm = rng.permutation(t_in)
            mid = t_in // 2
            base = tokens(bb, clip, dev, amp)
            shuf = tokens(bb, clip[perm], dev, amp)
            froz = tokens(bb, np.repeat(clip[mid:mid + 1], t_in, axis=0), dev, amp)
            for k, li in enumerate(TAPS):
                acc[src][li]["across"].append(across_tubelet(base[k], grid))
                acc[src][li]["frozen"].append(across_tubelet(froz[k], grid))
                acc[src][li]["shuf"].append(rel_delta(base[k], shuf[k]))
        print(f"  {i+1}/{args.clips} {name}", flush=True)

    print(f"\n{args.domain} {args.split}: token structure per tapped block")
    print("%-5s %-6s %14s %14s %14s" % (
        "input", "block", "across-tubelet", "same, frozen", "shuffle-delta"))
    out = {}
    for src in ("lq", "gt"):
        for li in TAPS:
            a = float(np.mean(acc[src][li]["across"]))
            f = float(np.mean(acc[src][li]["frozen"]))
            s = float(np.mean(acc[src][li]["shuf"]))
            out[f"{src}/{li}"] = {"across": a, "frozen": f, "shuffle": s}
            print("%-5s %-6d %14.4f %14.4f %14.4f" % (src, li, a, f, s))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"domain": args.domain, "split": args.split,
                   "clips": args.clips, "stats": out}, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
