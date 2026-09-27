"""Measure the scale of the Wan latent bridge so sigma_0 can be chosen from data,
not guessed.

Reports, over N sampled training clips:
  - per-channel std of z_lq, z_hq  (should be ~1 after Wan per-channel normalization)
  - std / RMS of the drift d = z_hq - z_lq   <- this is what v_target regresses to
  - the ratio  RMS(d) / RMS(z)               <- tells us what sigma_0 is "large"

Run from the vjepa_llve repo root.
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from tools.v11_oracle_delta_probe import load_clip, build_split_seqs
from data.bvi_rlv_dataset import BVIRLVDataset

WAN_PATH = "/cache/zihaohe/checkpoint/wan21_t2v_1_3b"
BVI_RLV_PATH = "/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos"
N_CLIPS = 12
CROP = 256
device = "cuda"


def wan_encode(vae, x_rgb, mean, std):
    z = vae.encode(x_rgb * 2 - 1).latent_dist.mode()
    return (z - mean) * std


def main():
    print("loading Wan VAE ...")
    vae = AutoencoderKLWan.from_pretrained(
        WAN_PATH, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, 16, 1, 1, 1).float()
    std = (1.0 / torch.tensor(vae.config.latents_std, device=device).view(1, 16, 1, 1, 1).float())

    print("latents_mean :", np.round(np.array(vae.config.latents_mean), 3))
    print("latents_std  :", np.round(np.array(vae.config.latents_std), 3))

    sdsd_train, _ = build_split_seqs()
    bvi = BVIRLVDataset(BVI_RLV_PATH, split="train", clip_len=16, crop_size=CROP)
    print(f"data: BVI={len(bvi)} SDSD-train={len(sdsd_train)}")

    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("z_lq", "z_hq", "drift")}
    per_ch_drift = []

    for i in range(N_CLIPS):
        if i % 2 == 0:
            s = bvi[int(rng.integers(0, len(bvi)))]
            lq, gt = s["lq"].unsqueeze(0).to(device), s["hq"].unsqueeze(0).to(device)
            src = "bvi"
        else:
            _, lq_p, gt_p = sdsd_train[int(rng.integers(0, len(sdsd_train)))]
            lq_t, gt_t = load_clip(lq_p, gt_p, crop_size=CROP)
            lq, gt = lq_t.to(device), gt_t.to(device)
            src = "sdsd"

        with torch.no_grad():
            z_lq = wan_encode(vae, lq, mean, std)
            z_hq = wan_encode(vae, gt, mean, std)
        d = z_hq - z_lq

        acc["z_lq"].append(z_lq.std().item())
        acc["z_hq"].append(z_hq.std().item())
        acc["drift"].append(d.std().item())
        per_ch_drift.append(d.squeeze(0).flatten(1).std(dim=1).cpu().numpy())

        print(f"  [{i:2d}] {src:4s} shape={tuple(z_lq.shape)} "
              f"std(z_lq)={z_lq.std():.3f} std(z_hq)={z_hq.std():.3f} "
              f"std(drift)={d.std():.3f} RMS(drift)={d.pow(2).mean().sqrt():.3f}")

    print("\n===== SUMMARY over %d clips =====" % N_CLIPS)
    for k, v in acc.items():
        print(f"  {k:6s} std: mean={np.mean(v):.4f}  min={np.min(v):.4f}  max={np.max(v):.4f}")
    ratio = np.mean(acc["drift"]) / np.mean(acc["z_hq"])
    print(f"\n  RMS(drift) / RMS(z_hq) = {ratio:.4f}")
    pcd = np.stack(per_ch_drift).mean(0)
    print(f"  per-channel drift std: min={pcd.min():.3f} max={pcd.max():.3f} "
          f"mean={pcd.mean():.3f}")
    print(f"  per-channel: {np.round(pcd, 2)}")

    print("\n  --- sigma_0 guidance (sigma(t)=sigma_0*sin(pi t), peak at t=0.5) ---")
    for s0 in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        print(f"    sigma_0={s0:<5} peak noise / drift RMS = "
              f"{s0 / np.mean(acc['drift']):.3f}   peak noise / z RMS = "
              f"{s0 / np.mean(acc['z_hq']):.3f}")


if __name__ == "__main__":
    main()
