"""Render actual attention heatmaps (full soft distribution, not just argmax)
for a handful of query points at enc_s4 (coarsest, most concentrated scale)
and dec_s3, overlaid conceptually via saving raw 16x16 heatmap arrays plus
the corresponding LQ/GT crop, so we can *look* at whether there's real
correlation with image content (bright/dark regions, structure) before
committing to a paper figure.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

save_dir = "/home/ma-user/workspace/zihaohe/visualizations/fig_cross_attention"
lq = np.load(f"{save_dir}/lq_clip.npy")  # (1,3,T,H,W)
pred = np.load(f"{save_dir}/pred_clip.npy")
gt = np.load(f"{save_dir}/gt_clip.npy")

frame_idx = 8
lq_img = lq[0, :, frame_idx].transpose(1, 2, 0)  # HxWx3
pred_img = pred[0, :, frame_idx].transpose(1, 2, 0)
gt_img = gt[0, :, frame_idx].transpose(1, 2, 0)

for tag, gridname in [("enc_s4", "16x16"), ("dec_s3", "32x32"), ("enc_s3", "32x32")]:
    attn = torch.load(f"{save_dir}/attn_{tag}.pt")  # [BT, h, HW_q, 256]
    a = attn.mean(dim=1)[frame_idx]  # [HW_q, 256]
    Hq = Wq = int(a.shape[0] ** 0.5)

    # pick query points: brightest and darkest patch (by LQ mean intensity in that query cell)
    lq_gray = lq_img.mean(axis=-1)
    Himg, Wimg = lq_gray.shape
    cellH, cellW = Himg / Hq, Wimg / Wq
    means = np.zeros((Hq, Wq))
    for r in range(Hq):
        for c in range(Wq):
            means[r, c] = lq_gray[int(r*cellH):int((r+1)*cellH), int(c*cellW):int((c+1)*cellW)].mean()
    dark_idx = np.unravel_index(np.argmin(means), means.shape)
    bright_idx = np.unravel_index(np.argmax(means), means.shape)
    mid_idx = (Hq // 2, Wq // 2)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(lq_img); axes[0].set_title(f"LQ frame (grid {gridname})")
    for (r, c), color in [(dark_idx, "red"), (bright_idx, "cyan"), (mid_idx, "yellow")]:
        rect = plt.Rectangle((c*cellW, r*cellH), cellW, cellH, fill=False, edgecolor=color, linewidth=2)
        axes[0].add_patch(rect)

    for ax, (r, c), name in zip(axes[1:], [dark_idx, bright_idx, mid_idx], ["dark query", "bright query", "center query"]):
        q_flat = r * Wq + c
        heat = a[q_flat].reshape(16, 16).numpy()
        im = ax.imshow(heat, cmap="viridis")
        ax.set_title(f"{name}\nattn over 16x16 vjepa tokens\nmax={heat.max():.4f} (unif={1/256:.4f})")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out_path = f"{save_dir}/debug_{tag}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("saved", out_path)
