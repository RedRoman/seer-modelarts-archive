"""STCD (indoor_G.pth) on the LOL image splits, appended to STCD_unified_eval.json.

Single image -> netG(x, x)[2] (same image as both frames), BGR in/out exactly as in
run_stcd_unified.STCD, fp32, full resolution. H, W reflect-padded to a multiple of 4 (two stride-2
convs + two x2 pixel-shuffles with skip concats) and cropped back. Metrics via
seer_eval_unified.score_frames (uint8-RGB PSNR, LPIPS, DISTS, CLIPIQA), per image, mean over images.
Iterators: eval_lib.iter_lol_v1 / iter_lol_v2("Syn") / iter_lol_v2("Real"); per_seq keyed by the
image names the iterators yield. A backup of the previous JSON is written first.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/temp/zihaohe/workspace/baselines")
import run_stcd_unified as RS  # noqa: E402
from common.eval_lib import iter_lol_v1, iter_lol_v2  # noqa: E402
from seer_eval_unified import score_frames  # noqa: E402

JSON = RS.DEFAULT_OUT
BACKUP = JSON.replace(".json", ".bak_before_LOL.json")
MULT = 4
SPLITS = {
    "L3_LOL-v1": iter_lol_v1,
    "L4_LOLv2-Syn": lambda: iter_lol_v2("Syn"),
    "L5_LOLv2-Real": lambda: iter_lol_v2("Real"),
}
KEYS = ("psnr_rgb", "lpips", "dists", "clipiqa")


@torch.no_grad()
def enhance_image(net, x_rgb):
    """x_rgb: (1,3,H,W) RGB [0,1] -> (1,3,H,W) RGB [0,1]; netG(x, x)[2] with BGR in/out."""
    H, W = x_rgb.shape[-2:]
    ph, pw = (-H) % MULT, (-W) % MULT
    x = x_rgb[:, [2, 1, 0]]
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    y = net(x, x)[2].float()[..., :H, :W]
    return y[:, [2, 1, 0]].clamp(0, 1)


def main():
    if not os.path.exists(BACKUP):
        shutil.copy2(JSON, BACKUP)
        print(f"[BACKUP] {JSON} -> {BACKUP}", flush=True)
    d = json.load(open(JSON))
    dev = torch.device("cuda")
    model = RS.STCD(dev)
    for split, it_fn in SPLITS.items():
        t0 = time.time()
        ps = {}
        n_pad = 0
        for name, lq_u8, gt_u8 in it_fn():
            lq = torch.from_numpy(np.ascontiguousarray(lq_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            gt = torch.from_numpy(np.ascontiguousarray(gt_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            n_pad += int(lq.shape[-2] % MULT != 0 or lq.shape[-1] % MULT != 0)
            pred = enhance_image(model.net, lq)
            r = score_frames(pred, gt)
            r["HxW"] = f"{lq.shape[-2]}x{lq.shape[-1]}"
            ps[name] = r
        s = {k: float(np.mean([v[k] for v in ps.values()])) for k in KEYS}
        s["n"] = len(ps)
        s["n_padded"] = n_pad
        d["per_seq"][split] = ps
        d["summary"][split] = s
        print(f"== {split:14s} PSNR={s['psnr_rgb']:.3f} LPIPS={s['lpips']:.4f} DISTS={s['dists']:.4f} "
              f"CLIPIQA={s['clipiqa']:.4f} n={s['n']} padded={n_pad} ({time.time() - t0:.0f}s)", flush=True)
    d["lol_inference"] = ("L3-L5 (LOL images): netG(x, x)[2] with the single image as both frames, BGR in/out, "
                          "fp32, full resolution, reflect pad to multiple of 4 then crop (all LOL images are "
                          "already multiples of 4 -> no padding applied); indoor_G.pth cross-domain")
    with open(JSON, "w") as f:
        json.dump(d, f, indent=1)
    print(f"[SAVE] {JSON}", flush=True)


if __name__ == "__main__":
    main()
