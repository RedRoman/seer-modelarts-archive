"""STCD (IJCAI'25 LLVE-STCD, official SDSD-indoor checkpoint) under the unified protocol.

Same data iterators + metric functions as the baselines and Seer:
  - STCD-12_SDSD-indoor : eval_lib.iter_sdsd_indoor_stcd  (STCD's 12-seq test split)
  - L0_SDSD-indoor      : seer_eval_unified.iter_sdsd_indoor_seer (Seer's 15-seq test split)
  - L1_SDSD-outdoor     : eval_lib.iter_sdsd_outdoor (25 seqs)
  - L2_BVI-RLV          : eval_lib.iter_bvi (16 seqs)
Video = center T=16 frames, 512x512 center crop (done by the iterators). Scoring via
seer_eval_unified.score_frames: per-frame uint8-RGB PSNR (rgb_psnr_uint8), LPIPS
(lpips_batch), DISTS (dists_batch), CLIPIQA (clipiqa_batch); mean over frames per
sequence, then mean over sequences.

How STCD is run (network code reused from the official repo, not reimplemented):
  - low_light_transformer42(nf=64, nframes=5, groups=8, front_RBs=1, back_RBs=1,
    predeblur=True, HR_in=True, w_TSA=True), weights indoor_G.pth loaded strict.
  - STCD is a 2-frame model: output frame t = netG(x_t, x_{t+1})[2] (fake_H), exactly as
    in the official test loader (idx1 = min(idx+1, max_idx-1)). Neighbour frames are taken
    from inside the same 16-frame 512x512 clip; at the clip end (t=15) the neighbour is
    replicate-padded (x_16 := x_15), matching the official loader's clamp.
  - Channel order: the official loader reads the SDSD .npy frames (which are stored RGB)
    and swaps [2,1,0], i.e. the network was trained/tested on BGR tensors. We therefore
    feed clip[:, [2,1,0]] and swap the output back to RGB before scoring. All iterators
    here yield true RGB (npy raw order = RGB; PNG via imageio = RGB).
  - fp32, native resolution (no 960x512 resize; the unified protocol crops 512x512 at
    native resolution).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

STCD_ROOT = "/temp/zihaohe/cache/stcd_repro/LLVE_STCD"
STCD_CKPT = os.path.join(STCD_ROOT, "pretrained_model/SDSD_indoor_model/indoor_G.pth")
VJ = "/temp/zihaohe/workspace/vjepa_llve"
BASE = "/temp/zihaohe/workspace/baselines"
CLEAN6 = ["pair50", "pair52", "pair58", "pair60", "pair68", "pair70"]
DEFAULT_OUT = "/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals/STCD_unified_eval.json"


def _import_stcd_arch():
    """Import STCD's arch module without clobbering vjepa_llve's `models` package."""
    is_models = lambda k: k == "models" or k.startswith("models.")  # noqa: E731
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items() if is_models(k)}
    for k in saved_mods:
        del sys.modules[k]
    sys.path.insert(0, STCD_ROOT)
    try:
        mod = importlib.import_module("models.archs.low_light_transformer42")
        cls = mod.low_light_transformer
    finally:
        for k in [k for k in sys.modules if is_models(k)]:
            del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path
    return cls


# STCD arch must be imported before vjepa_llve's `models` package is loaded.
_STCD_CLS = _import_stcd_arch()

for p in (VJ, os.path.join(VJ, "tools"), BASE):
    if p not in sys.path:
        sys.path.insert(0, p)
from common.eval_lib import (  # noqa: E402
    STCD_INDOOR_12, iter_sdsd_indoor_stcd, iter_sdsd_outdoor, iter_bvi,
)
from seer_eval_unified import iter_sdsd_indoor_seer, score_frames  # noqa: E402

SPLITS = {
    "STCD-12_SDSD-indoor": lambda: iter_sdsd_indoor_stcd(n_frames=16, crop=512),
    "L0_SDSD-indoor": lambda: iter_sdsd_indoor_seer(n_frames=16, crop=512),
    "L1_SDSD-outdoor": lambda: iter_sdsd_outdoor(n_frames=16, crop=512),
    "L2_BVI-RLV": lambda: iter_bvi(n_frames=16, crop=512),
}
METRICS = ("psnr_rgb", "lpips", "dists", "clipiqa")


class STCD:
    def __init__(self, device, ckpt=STCD_CKPT):
        net = _STCD_CLS(nf=64, nframes=5, groups=8, front_RBs=1, back_RBs=1, center=None,
                        predeblur=True, HR_in=True, w_TSA=True)
        sd = torch.load(ckpt, map_location="cpu")
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        print(f"[STCD] loaded {ckpt} strict=True ({len(sd)} tensors, "
              f"{sum(v.numel() for v in sd.values()) / 1e6:.2f}M params)", flush=True)
        self.net = net.to(device).eval()

    @torch.no_grad()
    def enhance_clip(self, clip_rgb, chunk=8, swap_bgr=True):
        """clip_rgb: (T,3,H,W) float [0,1] RGB on GPU -> (T,3,H,W) RGB [0,1].
        Output t uses (x_t, x_{min(t+1,T-1)}), i.e. replicate pad at the clip end."""
        x = clip_rgb[:, [2, 1, 0]] if swap_bgr else clip_rgb
        T = x.shape[0]
        nxt = torch.clamp(torch.arange(T, device=x.device) + 1, max=T - 1)
        x2 = x[nxt]
        outs = [self.net(x[i:i + chunk], x2[i:i + chunk])[2] for i in range(0, T, chunk)]
        y = torch.cat(outs, 0).float()
        if swap_bgr:
            y = y[:, [2, 1, 0]]
        return y.clamp(0, 1)


def to_dev(lq_u8, gt_u8, dev):
    lq = torch.from_numpy(np.ascontiguousarray(lq_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
    gt = torch.from_numpy(np.ascontiguousarray(gt_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
    return lq, gt


def mean_over(d, names):
    names = [n for n in names if n in d]
    out = {k: float(np.mean([d[n][k] for n in names])) for k in METRICS} if names else {}
    out["n"] = len(names)
    out["seqs"] = names
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    ap.add_argument("--only_seqs", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    model = STCD(dev)
    per_seq, summary, frame_ids = {}, {}, {}
    t_all = time.time()
    for split in args.splits:
        per_seq[split] = {}
        t0 = time.time()
        for i, (name, lq_u8, gt_u8) in enumerate(SPLITS[split]()):
            if args.limit is not None and i >= args.limit:
                break
            if args.only_seqs and name not in args.only_seqs:
                continue
            lq, gt = to_dev(lq_u8, gt_u8, dev)
            pred = model.enhance_clip(lq)
            r = score_frames(pred, gt)
            r["HxW"] = f"{lq.shape[-2]}x{lq.shape[-1]}"
            per_seq[split][name] = r
            print(f"  {split:20s} {name:28s} PSNR={r['psnr_rgb']:.3f} LPIPS={r['lpips']:.4f} "
                  f"DISTS={r['dists']:.4f} CLIPIQA={r['clipiqa']:.4f} T={r['n_frames']} {r['HxW']}",
                  flush=True)
        d = per_seq[split]
        summary[split] = mean_over(d, list(d.keys()))
        summary[split].pop("seqs")
        if split in ("STCD-12_SDSD-indoor", "L0_SDSD-indoor"):
            summary[split]["clean6"] = mean_over(d, CLEAN6)
        s = summary[split]
        if s["n"]:
            print(f"== {split:20s} PSNR={s['psnr_rgb']:.3f} LPIPS={s['lpips']:.4f} DISTS={s['dists']:.4f} "
                  f"CLIPIQA={s['clipiqa']:.4f} n={s['n']} ({time.time() - t0:.0f}s)", flush=True)

    # split membership (for joining with Seer's JSON)
    l0 = list(per_seq.get("L0_SDSD-indoor", {}).keys())
    membership = {
        "seer_test_split_L0": l0,
        "stcd12": STCD_INDOOR_12,
        "held_out_for_both": [s for s in l0 if s in STCD_INDOOR_12],
        "stcd12_in_seer_train": [s for s in STCD_INDOOR_12 if l0 and s not in l0],
        "seer_test_in_stcd_train": [s for s in l0 if s not in STCD_INDOOR_12],
    }
    # Derived: STCD-12 with index-paired GT for the 6 offset-named seqs (taken from L0,
    # whose iterator pairs LQ/GT by sorted index) + the 6 same-named seqs from STCD-12.
    if "STCD-12_SDSD-indoor" in per_seq and "L0_SDSD-indoor" in per_seq:
        a, b = per_seq["STCD-12_SDSD-indoor"], per_seq["L0_SDSD-indoor"]
        mix = {}
        for s in STCD_INDOOR_12:
            src = b if s in CLEAN6 else a
            if s in src:
                mix[s] = src[s]
        der = mean_over(mix, list(mix.keys()))
        der["note"] = ("DERIVED, not an official split: pair13/15/21/23/31/33 from STCD-12 "
                       "(basename == index pairing for these), pair50..70 from L0 (index pairing)")
        summary["STCD-12_indexpaired_derived"] = der

    out = {
        "method": "STCD (LLVE-STCD, IJCAI'25), official SDSD-indoor ckpt indoor_G.pth",
        "ckpt": STCD_CKPT,
        "protocol": ("unified: video center T=16, 512x512 center crop (eval_lib / seer_eval_unified "
                     "iterators); uint8 RGB PSNR, LPIPS/DISTS/CLIPIQA pyiqa via score_frames; "
                     "per-frame -> per-seq mean -> mean over seqs"),
        "stcd_inference": ("2-frame model: out_t = netG(x_t, x_{t+1})[2]; neighbour from the same "
                           "16-frame 512x512 clip, replicate pad at clip end (t=15 uses x_15); "
                           "input swapped RGB->BGR (official loader swaps the RGB .npy), output "
                           "swapped back; fp32; native resolution, no 960x512 resize"),
        "clean6_definition": CLEAN6,
        "membership": membership,
        "per_seq": per_seq,
        "summary": summary,
        "wall_s": round(time.time() - t_all, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[SAVE] {args.out}", flush=True)
    for k, s in summary.items():
        if s.get("n"):
            print(f"{k:30s} PSNR={s['psnr_rgb']:.3f} LPIPS={s['lpips']:.4f} DISTS={s['dists']:.4f} "
                  f"CLIPIQA={s['clipiqa']:.4f} n={s['n']}")
        if "clean6" in s:
            c = s["clean6"]
            print(f"{'  clean6 of ' + k:30s} PSNR={c['psnr_rgb']:.3f} LPIPS={c['lpips']:.4f} "
                  f"DISTS={c['dists']:.4f} CLIPIQA={c['clipiqa']:.4f} n={c['n']}")


if __name__ == "__main__":
    main()
