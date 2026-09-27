"""Sanity 3: (a) crop-size sweep on first-16 frames to locate the old comprehensive protocol;
(b) pair58/pair68 diagnostics under index pairing (window position, brightness)."""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/temp/zihaohe/workspace/baselines")
import run_stcd_unified as R
from common.eval_lib import rgb_psnr_uint8, center_crop

ROOT = "/cache/zihaohe/datasets/SDSD_indoor/indoor_static_np"
dev = torch.device("cuda")
M = R.STCD(dev)


def names(seq, pairing):
    lq = sorted(os.listdir(f"{ROOT}/input/{seq}")); gt = sorted(os.listdir(f"{ROOT}/GT/{seq}"))
    if pairing == "basename":
        c = sorted(set(lq) & set(gt)); return c, c
    m = min(len(lq), len(gt)); return lq[:m], gt[:m]


def run(seq, pairing, start, n=16, crop=None):
    lq, gt = names(seq, pairing)
    if start == "center":
        start = max(0, (len(lq) - n) // 2)
    lq, gt = lq[start:start + n], gt[start:start + n]
    rd = lambda d, f: (center_crop(np.load(f"{ROOT}/{d}/{seq}/{f}"), crop) if crop else np.load(f"{ROOT}/{d}/{seq}/{f}"))
    a = np.stack([rd("input", f) for f in lq]); b = np.stack([rd("GT", f) for f in gt])
    x, _ = R.to_dev(a, b, dev)
    y = M.enhance_clip(x, chunk=2 if crop is None or crop > 800 else 8)
    p = (y.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    ps = [rgb_psnr_uint8(p[t], b[t]) for t in range(len(p))]
    return float(np.mean(ps)), float(p.mean()), float(b.mean()), float(a.mean()), lq[0], gt[0]


print("== (a) crop sweep, first 16 frames ==")
for seq in ["pair13", "pair50"]:
    for pairing in ["basename", "index"]:
        for crop in [512, 640, 768, 896, 1024, None]:
            r = run(seq, pairing, 0, crop=crop)
            print(seq, pairing, "crop", crop, f"PSNR={r[0]:.3f}", flush=True)

print("== (b) window diagnostics, index pairing, crop 512 ==")
for seq in ["pair58", "pair68", "pair52"]:
    L = len(names(seq, "index")[0])
    for start in [0, 16, 32, "center", L - 16]:
        r = run(seq, "index", start, crop=512)
        print(seq, "start", start, f"PSNR={r[0]:.3f} pred_mean={r[1]:.1f} gt_mean={r[2]:.1f} lq_mean={r[3]:.1f}",
              r[4], r[5], flush=True)
