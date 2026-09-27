"""Zero-DCE inference on SDSD-indoor held-out 11 sequences.

Protocol (matches wanvj eval):
  - Center 16 frames, center-crop to 256x256.
  - Frame-by-frame.
  - PSNR/SSIM/LPIPS computed; MP4 viz saved.

Zero-DCE is unsupervised — trained on LOL with no GT.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import numpy as np
import torch

ZDCE = "/cache/zihaohe/ext_methods/Zero-DCE/Zero-DCE_code"
VJEPA = "/home/ma-user/workspace/zihaohe/vjepa_llve"
sys.path.insert(0, ZDCE)
sys.path.insert(0, VJEPA)

import model as zdce_model  # type: ignore  # noqa: E402
from tools.iqa_metrics import compute_iqa, save_compare_video  # type: ignore  # noqa: E402


SDSD_INPUT = "/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor/indoor_static_np/input"
SDSD_GT = "/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor/indoor_static_np/GT"

SEQS = [
    "pair13", "pair15", "pair21", "pair23", "pair31",
    "pair33", "pair50", "pair52", "pair58", "pair60", "pair70",
]


def load_seq_center(seq, n_frames=16, crop=256):
    in_dir = os.path.join(SDSD_INPUT, seq)
    gt_dir = os.path.join(SDSD_GT, seq)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".npy"))
    start = (len(files) - n_frames) // 2
    sel = files[start:start + n_frames]
    lq_arr = np.stack([np.load(os.path.join(in_dir, f)) for f in sel], axis=0)
    gt_arr = np.stack([np.load(os.path.join(gt_dir, f)) for f in sel], axis=0)
    H, W = lq_arr.shape[1], lq_arr.shape[2]
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    lq_arr = lq_arr[:, y0:y0 + crop, x0:x0 + crop, :]
    gt_arr = gt_arr[:, y0:y0 + crop, x0:x0 + crop, :]
    lq = torch.from_numpy(lq_arr).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    gt = torch.from_numpy(gt_arr).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    return lq, gt


def psnr_u8(a, b):
    a8 = (a.clamp(0, 1) * 255).round().to(torch.uint8).float()
    b8 = (b.clamp(0, 1) * 255).round().to(torch.uint8).float()
    mse = ((a8 - b8) ** 2).mean()
    if mse < 1e-10:
        return 99.0
    return float(20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/cache/zihaohe/visualizations/_baselines")
    args = ap.parse_args()
    device = torch.device("cuda")
    out_dir = os.path.join(args.out_root, "zero_dce")
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(ZDCE, "snapshots", "Epoch99.pth")
    print(f"[Zero-DCE] loading {weights_path}")
    net = zdce_model.enhance_net_nopool().to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()

    rows = []
    t_total0 = time.time()
    for seq in SEQS:
        try:
            lq, gt = load_seq_center(seq)
        except Exception as e:
            print(f"[{seq}] LOAD FAIL: {e}")
            continue
        lq_d = lq.to(device)
        gt_d = gt.to(device)
        t0 = time.time()
        T = lq_d.shape[2]
        preds = []
        with torch.no_grad():
            for t in range(T):
                frame = lq_d[:, :, t]  # (1,3,H,W)
                _e1, enh, _r = net(frame)
                preds.append(enh)
        pred = torch.stack(preds, dim=2)
        infer_t = time.time() - t0

        psnr = psnr_u8(pred, gt_d)
        try:
            iqa = compute_iqa(pred, gt_d, metrics=("ssim", "lpips"))
            ssim = iqa["ssim"]
            lpips = iqa["lpips"]
        except Exception as e:
            print(f"  iqa fail: {e}")
            ssim, lpips = float("nan"), float("nan")
        psnr_lq = psnr_u8(lq_d, gt_d)
        rows.append({"seq": seq, "psnr": psnr, "ssim": ssim, "lpips": lpips,
                     "psnr_lq": psnr_lq, "time_s": round(infer_t, 2)})
        print(f"  [{seq}] LQ={psnr_lq:.2f}  ZDCE={psnr:.3f}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}  ({infer_t:.1f}s)")

        mp4 = os.path.join(out_dir, f"{seq}.mp4")
        try:
            save_compare_video(mp4, lq_d.cpu(), pred.cpu(), gt_d.cpu(), fps=10)
        except Exception as e:
            print(f"  viz fail: {e}")

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "psnr", "ssim", "lpips", "psnr_lq", "time_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    mean_psnr = float(np.mean([r["psnr"] for r in rows]))
    mean_lq = float(np.mean([r["psnr_lq"] for r in rows]))
    mean_ssim = float(np.mean([r["ssim"] for r in rows]))
    mean_lpips = float(np.mean([r["lpips"] for r in rows]))
    mean_time = float(np.mean([r["time_s"] for r in rows]))
    print(f"\n=== Zero-DCE mean over {len(rows)} seqs ===")
    print(f"  PSNR  = {mean_psnr:.3f} (LQ baseline {mean_lq:.3f})")
    print(f"  SSIM  = {mean_ssim:.4f}")
    print(f"  LPIPS = {mean_lpips:.4f}")
    print(f"  time  = {mean_time:.2f}s/seq, total={time.time()-t_total0:.1f}s")
    with open(csv_path, "a", newline="") as f:
        f.write(f"MEAN,{mean_psnr:.3f},{mean_ssim:.4f},{mean_lpips:.4f},{mean_lq:.3f},{mean_time:.2f}\n")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
