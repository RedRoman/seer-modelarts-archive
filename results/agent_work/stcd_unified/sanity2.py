"""Sanity 2: try to reproduce the old comprehensive_eval STCD numbers (pair13 22.111, pair50 29.315)
with a full-frame tiled protocol (tile 512 / overlap 128 cosine taper), first 16 frames."""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/temp/zihaohe/workspace/baselines")
import run_stcd_unified as R
from common.eval_lib import rgb_psnr_uint8

ROOT = "/cache/zihaohe/datasets/SDSD_indoor/indoor_static_np"
dev = torch.device("cuda")
M = R.STCD(dev)


def load(seq, pairing, first=True, n=16, crop=None):
    lq = sorted(os.listdir(f"{ROOT}/input/{seq}")); gt = sorted(os.listdir(f"{ROOT}/GT/{seq}"))
    if pairing == "basename":
        c = sorted(set(lq) & set(gt)); lq, gt = c, c
    else:
        m = min(len(lq), len(gt)); lq, gt = lq[:m], gt[:m]
    lq, gt = lq[:n], gt[:n]
    return (np.stack([np.load(f"{ROOT}/input/{seq}/{f}") for f in lq]),
            np.stack([np.load(f"{ROOT}/GT/{seq}/{f}") for f in gt]), None)


def taper(h, w, ov, t, b, l, r):
    wh = torch.ones(h, device=dev); ww = torch.ones(w, device=dev)
    ramp = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, ov, device=dev)))
    if t: wh[:ov] = ramp
    if b: wh[-ov:] = ramp.flip(0)
    if l: ww[:ov] = ramp
    if r: ww[-ov:] = ramp.flip(0)
    return wh[:, None] * ww[None, :]


@torch.no_grad()
def tiled(x, swap=True, tile=512, ov=128):
    T, C, H, W = x.shape
    acc = torch.zeros_like(x); wacc = torch.zeros(1, 1, H, W, device=dev)
    st = tile - ov
    hs = list(range(0, H - tile + 1, st)); hs += [H - tile] if hs[-1] != H - tile else []
    ws = list(range(0, W - tile + 1, st)); ws += [W - tile] if ws[-1] != W - tile else []
    for h0 in hs:
        for w0 in ws:
            p = M.enhance_clip(x[..., h0:h0 + tile, w0:w0 + tile], swap_bgr=swap)
            tw = taper(tile, tile, ov, h0 > 0, h0 + tile < H, w0 > 0, w0 + tile < W)
            acc[..., h0:h0 + tile, w0:w0 + tile] += p * tw
            wacc[..., h0:h0 + tile, w0:w0 + tile] += tw
    return (acc / wacc).clamp(0, 1)


for seq in ["pair13", "pair50"]:
    for pairing in ["basename", "index"]:
        a, b, ids = load(seq, pairing, first=True, n=16, crop=None)
        lq, _ = R.to_dev(a, b, dev)
        for swap in [True, False]:
            p = (tiled(lq, swap).permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            ps = float(np.mean([rgb_psnr_uint8(p[t], b[t]) for t in range(len(p))]))
            print(seq, f"tiled512_first16_{pairing}_swap{swap}", f"{ps:.3f}", flush=True)
