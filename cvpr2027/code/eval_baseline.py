"""Score a published baseline through the same evaluator our model goes through.

The archived baseline numbers cannot be compared against the new ones: they were
produced by an evaluator whose SDSD-indoor iterator paired LQ and GT frames
incorrectly, which depressed the L0 column, and that bug is deliberately not
inherited by tools/eval_e4.py. Rather than fix the old scripts, this reuses
eval_e4's sequence iteration and metric code verbatim and only swaps in the
baseline's forward pass, so a baseline row and our row differ in the model and
nothing else.

Baselines are per-frame image models; they see each frame independently, which is
how they were evaluated before and how their papers evaluate them.

  python tools/eval_baseline.py --method hvi_cidnet \
      --ckpt .../weights/LOLv1/w_perc.pth --domain sdsd --split val --out x.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = os.path.join(os.path.dirname(CODE), "baselines")
EXT = os.path.join(os.path.dirname(CODE), "ext_methods")
for p in (CODE, os.path.join(CODE, "tools"), BASELINES):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_e4 import (gt_mean_rescale, held_out_sequences, lpips_of,  # noqa: E402
                     psnr_y_uint8)
from common.eval_lib import rgb_psnr_uint8  # noqa: E402
from temporal_metrics import safe_gt_flows, temporal_metrics  # noqa: E402

ROOT = os.environ.get("SEER_ROOT", "/cache/zihaohe/seer_cvpr")


def load_hvi_cidnet(ckpt, dev):
    sys.path.insert(0, os.path.join(BASELINES, "HVI-CIDNet"))
    from net.CIDNet import CIDNet
    m = CIDNet()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("params", sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd, strict=True)
    return m.to(dev).eval(), 16


def load_darkir(ckpt, dev):
    sys.path.insert(0, os.path.join(BASELINES, "DarkIR"))
    from archs.DarkIR import DarkIR
    m = DarkIR(img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2,
               enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
               dilations=[1, 4, 9], extra_depth_wise=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("params", ck), strict=True)
    return m.to(dev).eval(), 8


def load_retinexformer(ckpt, dev):
    sys.path.insert(0, os.path.join(EXT, "Retinexformer", "basicsr", "models", "archs"))
    from RetinexFormer_arch import RetinexFormer
    m = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                      num_blocks=[1, 2, 2])
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        sd = ck.get("params", ck.get("params_ema", ck))
    else:
        sd = ck
    m.load_state_dict(sd, strict=True)
    return m.to(dev).eval(), 4


LOADERS = {"hvi_cidnet": load_hvi_cidnet, "darkir": load_darkir,
           "retinexformer": load_retinexformer}


@torch.no_grad()
def run_frames(model, mult, lq_u8, dev, batch=4):
    """lq_u8: (T,H,W,3) uint8 -> (T,H,W,3) uint8, one frame at a time."""
    t, h, w = lq_u8.shape[:3]
    out = np.empty_like(lq_u8)
    for s in range(0, t, batch):
        x = torch.from_numpy(lq_u8[s:s + batch]).float().div(255.)
        x = x.permute(0, 3, 1, 2).to(dev)
        ph, pw = (-h) % mult, (-w) % mult
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y[:, :, :h, :w].clamp(0, 1)
        out[s:s + batch] = (y.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(LOADERS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain", required=True, choices=["sdsd", "bvi", "lol"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = torch.device(f"{args.device}:0")
    model, mult = LOADERS[args.method](args.ckpt, dev)
    print(f"[eval_baseline] {args.method} ckpt={os.path.basename(args.ckpt)} "
          f"domain={args.domain} split={args.split}", flush=True)

    rows = []
    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences(args.domain, args.split, args.n_frames, args.crop)):
        if args.limit is not None and i >= args.limit:
            break
        pred = run_frames(model, mult, lq, dev)
        n = min(pred.shape[0], gt.shape[0])
        pred, gt_s, lq_s = pred[:n], gt[:n], lq[:n]
        gm = gt_mean_rescale(pred, gt_s)
        to_t = lambda a: torch.from_numpy(a.copy()).float().div(255.).permute(0, 3, 1, 2)
        row = {
            "name": name, "kind": kind, "frames": int(n),
            "psnr_rgb": float(np.mean([rgb_psnr_uint8(pred[j], gt_s[j]) for j in range(n)])),
            "psnr_y": float(np.mean([psnr_y_uint8(pred[j], gt_s[j]) for j in range(n)])),
            "psnr_y_gtmean": float(np.mean([psnr_y_uint8(gm[j], gt_s[j]) for j in range(n)])),
            "lpips": lpips_of(to_t(pred), to_t(gt_s), dev),
            "input_psnr_y": float(np.mean([psnr_y_uint8(lq_s[j], gt_s[j]) for j in range(n)])),
        }
        if kind == "video" and n >= 2:
            pt, gtt = to_t(pred).to(dev), to_t(gt_s).to(dev)
            flows = safe_gt_flows(gtt)
            lp = lambda a, b: lpips_of(a, b, dev)
            for k, v in temporal_metrics(pt, lp, flows).items():
                row[k] = v
            for k, v in temporal_metrics(gtt, lp, flows).items():
                row["gt_" + k] = v
        rows.append(row)
        print(f"  {i:3d} {name:<34} PSNR-RGB {row['psnr_rgb']:6.2f} "
              f"LPIPS {row['lpips']:.4f}", flush=True)

    keys = ["psnr_rgb", "psnr_y", "psnr_y_gtmean", "lpips", "input_psnr_y"]
    keys += [k for k in ("tlp", "mabd", "flicker", "warp_err",
                         "gt_tlp", "gt_mabd", "gt_flicker", "gt_warp_err")
             if k in rows[0]]
    agg = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    out = {"method": args.method, "ckpt": args.ckpt, "domain": args.domain,
           "split": args.split, "crop": args.crop, "n_frames": args.n_frames,
           "n_sequences": len(rows), "aggregate": agg, "per_sequence": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n{len(rows)} sequences | PSNR-RGB {agg['psnr_rgb']:.3f} | "
          f"PSNR-Y {agg['psnr_y']:.3f} | GT-mean {agg['psnr_y_gtmean']:.3f} | "
          f"LPIPS {agg['lpips']:.4f}\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
