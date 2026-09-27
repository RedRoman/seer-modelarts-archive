"""Day 5-9 + extra experiments while BVI download is paused.

D. SDSD-outdoor audit — v9.0b_mix tile=512 on 25 outdoor test seqs.
C. pair68 deep dive — per-frame PSNR + visual panels.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import torch
import imageio.v2 as imageio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import build_v9_model, rgb_psnr_uint8, load_seq_full
from tools.theta_llvestcd_eval import tiled_inference, to_u8, mean_psnr, load_seq

CKPT = "/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt"
CFG  = "/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml"


def run_outdoor():
    print("\n========== D. SDSD-OUTDOOR AUDIT ==========")
    out_dir = "/cache/zihaohe/checkpoint/audit_outdoor"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")
    ds = SDSDDataset(root="/home/ma-user/workspace/zihaohe/datasets/SDSD_outdoor",
                     subset="direct", split="test", clip_len=16, crop_size=256, is_train=False)
    seqs = list(ds.seqs)
    print(f"outdoor test: {len(seqs)} seqs")
    model = build_v9_model(CFG, CKPT, device)

    rows = []
    for seq_name, lq_paths, gt_paths in seqs:
        t0 = time.time()
        lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=16)
        H, W = lq.shape[-2:]
        lq_dev, gt_dev = lq.to(device), gt.to(device)
        try:
            pred = tiled_inference(model, lq_dev, tile=512, overlap=128)
        except Exception as e:
            print(f"  [{seq_name}] FAILED: {type(e).__name__}: {e}")
            continue
        pred_u8 = to_u8(pred); gt_u8 = to_u8(gt_dev); lq_u8 = to_u8(lq_dev)
        psnr = mean_psnr(pred_u8, gt_u8)
        psnr_lq = mean_psnr(lq_u8, gt_u8)
        rows.append({"seq": seq_name, "T": lq.shape[2], "HxW": f"{H}x{W}",
                     "lq_psnr": psnr_lq, "v9_0b_mix_psnr": psnr,
                     "delta_vs_lq": psnr - psnr_lq, "time_s": round(time.time()-t0,1)})
        print(f"  [{seq_name}] LQ={psnr_lq:.2f} v9.0b_mix={psnr:.3f} ({time.time()-t0:.1f}s)")

    csv_path = os.path.join(out_dir, "outdoor_audit.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}")

    n = len(rows)
    mean_lq = float(np.mean([r["lq_psnr"] for r in rows]))
    mean_v90b = float(np.mean([r["v9_0b_mix_psnr"] for r in rows]))
    median_v90b = float(np.median([r["v9_0b_mix_psnr"] for r in rows]))
    md_path = os.path.join(out_dir, "outdoor_audit_summary.md")
    with open(md_path, "w") as f:
        f.write("# SDSD-outdoor audit (v9.0b_mix tile=512)\n\n")
        f.write(f"n={n}, LQ mean={mean_lq:.3f}, v9.0b_mix mean={mean_v90b:.3f}, median={median_v90b:.3f}\n\n")
        f.write("| seq | T | HxW | LQ | v9.0b_mix | Δ vs LQ |\n|---|---:|---|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['seq']} | {r['T']} | {r['HxW']} | {r['lq_psnr']:.2f} | "
                    f"{r['v9_0b_mix_psnr']:.3f} | {r['delta_vs_lq']:+.2f} |\n")
    print(f"wrote {md_path}")
    print(f"\n=== OUTDOOR RESULTS ===")
    print(f"  LQ mean:        {mean_lq:.3f}")
    print(f"  v9.0b_mix mean: {mean_v90b:.3f}")
    print(f"  v9.0b_mix median: {median_v90b:.3f}")
    return mean_v90b, median_v90b


def run_pair68_deep_dive():
    print("\n========== C. PAIR68 DEEP DIVE ==========")
    out_dir = "/cache/zihaohe/checkpoint/pair68_deep_dive"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")

    in_d = "/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor/indoor_static_np/input/pair68"
    gt_d = "/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor/indoor_static_np/GT/pair68"

    # Load full 16-frame seq
    lq, gt, T = load_seq(in_d, gt_d, n_frames=16)
    H, W = lq.shape[-2:]
    print(f"pair68: shape=(1,3,{T},{H},{W})")
    print(f"  LQ mean={lq.mean():.4f}  GT mean={gt.mean():.4f}")

    model = build_v9_model(CFG, CKPT, device)
    lq_dev = lq.to(device); gt_dev = gt.to(device)

    # Run tile=512
    pred = tiled_inference(model, lq_dev, tile=512, overlap=128)
    pred_u8 = to_u8(pred); gt_u8 = to_u8(gt_dev); lq_u8 = to_u8(lq_dev)

    # Per-frame PSNR
    print(f"\n  per-frame PSNR (LQ → v9.0b_mix → GT):")
    per_frame = []
    for t in range(T):
        psnr_lq = rgb_psnr_uint8(lq_u8[t], gt_u8[t])
        psnr_pred = rgb_psnr_uint8(pred_u8[t], gt_u8[t])
        per_frame.append({"frame": t, "lq_psnr": psnr_lq, "pred_psnr": psnr_pred})
        print(f"    frame {t}: LQ={psnr_lq:.2f}  pred={psnr_pred:.3f}")

    # Save visual panels for first 4 frames
    print(f"\n  saving visual panels...")
    rows = []
    for t in range(min(4, T)):
        err = np.abs(pred_u8[t].astype(np.int16) - gt_u8[t].astype(np.int16)).clip(0, 255).astype(np.uint8)
        err_amp = np.minimum(err * 4, 255).astype(np.uint8)
        # Concat horizontally: LQ | pred | GT | err
        row = np.concatenate([lq_u8[t], pred_u8[t], gt_u8[t], err_amp], axis=1)
        rows.append(row)
    panel = np.concatenate(rows, axis=0)
    panel_path = os.path.join(out_dir, "pair68_panels.png")
    imageio.imwrite(panel_path, panel)
    print(f"  wrote {panel_path} ({panel.shape})")

    # Per-frame CSV
    csv_path = os.path.join(out_dir, "pair68_per_frame.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "lq_psnr", "pred_psnr"])
        w.writeheader()
        for r in per_frame: w.writerow(r)
    print(f"  wrote {csv_path}")

    # Spatial PSNR map: 256x256 patches
    print(f"\n  computing spatial PSNR map (per-tile)...")
    tile_psnr = []
    tile = 256
    for t in range(min(4, T)):
        tps = []
        for h0 in range(0, H, tile):
            for w0 in range(0, W, tile):
                h1 = min(h0+tile, H); w1 = min(w0+tile, W)
                p = pred_u8[t, h0:h1, w0:w1]
                g = gt_u8[t, h0:h1, w0:w1]
                ps = rgb_psnr_uint8(p, g)
                tps.append((h0, w0, ps))
        tile_psnr.append(tps)
        print(f"    frame {t} spatial PSNR (256x256 tiles): "
              f"min={min(p[2] for p in tps):.1f}, max={max(p[2] for p in tps):.1f}, "
              f"mean={np.mean([p[2] for p in tps]):.2f}")

    # Summary
    md_path = os.path.join(out_dir, "pair68_summary.md")
    with open(md_path, "w") as f:
        f.write("# pair68 deep dive — worst case in θ Step 1 (BX1: 13.40 dB)\n\n")
        f.write(f"## Per-frame PSNR\n\n| frame | LQ | pred | gain |\n|---:|---:|---:|---:|\n")
        for r in per_frame:
            f.write(f"| {r['frame']} | {r['lq_psnr']:.2f} | {r['pred_psnr']:.3f} | "
                    f"{r['pred_psnr']-r['lq_psnr']:+.2f} |\n")
        f.write(f"\nFrame mean: LQ={np.mean([r['lq_psnr'] for r in per_frame]):.3f}, "
                f"pred={np.mean([r['pred_psnr'] for r in per_frame]):.3f}\n")
        f.write(f"\n## Stats\n\n  LQ mean (raw): {lq.mean():.4f}  GT mean (raw): {gt.mean():.4f}\n")
        f.write(f"  LQ→GT scaling factor: {gt.mean()/max(lq.mean(),1e-8):.2f}× (very dark scene)\n\n")
        f.write(f"## Visual panels: see pair68_panels.png\n")
    print(f"  wrote {md_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("D", "outdoor", "both"):
        run_outdoor()
    if mode in ("C", "pair68", "both"):
        run_pair68_deep_dive()
