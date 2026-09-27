# SOTA LLVE/LLIE Baselines on SDSD-indoor Held-Out (LLVE-STCD split)

**Date**: 2026-05-11
**Test set**: 11 SDSD-indoor sequences (pair13/15/21/23/31/33/50/52/58/60/70).
**Protocol**: center 16 frames per sequence, center-crop to 256x256, frame-by-frame inference.
**Metrics**: PSNR (uint8), SSIM and LPIPS via `pyiqa`.
**Hardware**: A100 80GB (shared with `wanvj_v32` training, ~30GB occupied).

---

## Models successfully run

| # | Model | Training data | Year/Venue | Source of weights |
|---|---|---|---|---|
| 1 | SCI (easy) | LIME / unsupervised | CVPR 2022 | github.com/vis-opt-group/SCI (baked in repo) |
| 2 | SCI (medium) | LOL / unsupervised | CVPR 2022 | github.com/vis-opt-group/SCI |
| 3 | SCI (difficult) | DARKFACE / unsupervised | CVPR 2022 | github.com/vis-opt-group/SCI |
| 4 | Zero-DCE | SICE Part 1 / unsupervised | CVPR 2020 | github.com/Li-Chongyi/Zero-DCE |
| 5 | MIRNet (pretrained) | LoL Dataset (supervised) | ECCV 2020 | hf.co/dblasko/mirnet-low-light-img-enhancement |
| 6 | MIRNet (finetuned) | LoL + Event-club photos | ECCV 2020 | same repo, finetuned variant |

All models run frame-by-frame (no temporal modelling); they are LLIE methods applied per-frame as proxies for LLVE.

---

## Per-sequence PSNR (dB)

| seq | LQ | wanvj P1 | NAFNet++ | SCI-med | SCI-diff | SCI-easy | Zero-DCE | MIRNet-pre | MIRNet-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pair13 | 19.14 | (n/a) | (n/a) | 24.13 | 23.99 | 20.50 | 24.75 | 9.61 | 11.43 |
| pair15 | 8.01 | (n/a) | (n/a) | 21.53 | 17.56 | 11.98 | 15.36 | 18.48 | 17.96 |
| pair21 | 9.79 | (n/a) | (n/a) | 24.47 | 21.63 | 12.63 | 22.34 | 16.86 | 21.25 |
| pair23 | 6.53 | (n/a) | (n/a) | 20.65 | 16.06 |  9.68 | 15.18 | 24.55 | 19.57 |
| pair31 | 7.38 | (n/a) | (n/a) | 16.56 | 30.44 | 13.98 | 20.87 | 18.61 | 16.33 |
| pair33 | 5.94 | (n/a) | (n/a) | 21.13 | 16.38 |  8.73 | 16.60 | 21.45 | 17.16 |
| pair50 | 7.44 | (n/a) | (n/a) | 19.93 | 17.80 | 10.12 | 19.05 | 20.87 | 21.62 |
| pair52 | 11.29 | (n/a) | (n/a) | 13.80 | 14.38 | 13.23 | 14.47 | 10.99 | 12.25 |
| pair58 | 9.94 | (n/a) | (n/a) | 13.78 | 13.96 | 11.79 | 14.23 | 10.57 | 12.18 |
| pair60 | 13.58 | (n/a) | (n/a) | 19.40 | 19.22 | 15.20 | 20.38 |  9.70 | 11.92 |
| pair70 | 8.74 | (n/a) | (n/a) | 17.22 | 16.54 | 11.71 | 16.05 | 16.73 | 17.78 |
| **MEAN** | **9.80** | **19.64** | **~21+** | **19.33** | **18.91** | **12.69** | **18.12** | **16.22** | **16.31** |

`wanvj P1` and `NAFNet++` rows: per-seq PSNR not reported in this run; only the published mean values are shown for context. Re-run their existing eval scripts to fill in.

---

## Mean SSIM / LPIPS / runtime

| Model | PSNR | SSIM | LPIPS | mean infer time / 16-frame clip (s) |
|---|---:|---:|---:|---:|
| LQ baseline | 9.799 | -- | -- | -- |
| SCI medium | **19.328** | 0.6388 | 0.4620 | 0.02 |
| SCI difficult | 18.905 | 0.6234 | 0.4750 | 0.01 |
| SCI easy | 12.686 | 0.5473 | 0.4584 | 0.01 |
| Zero-DCE | 18.117 | 0.6216 | 0.4510 | 0.04 |
| MIRNet pretrained | 16.220 | 0.5728 | 0.4736 | 1.24 |
| MIRNet finetuned | 16.313 | 0.5887 | 0.4924 | 1.05 |
| **wanvj v3.2 Phase 1 (ref)** | **19.64** | -- | -- | -- |
| **NAFNet++ from-scratch (ref)** | **~21+** | -- | -- | -- |

---

## Observations

1. **None of the off-the-shelf SOTA LLIE baselines beat NAFNet++ (~21+) or even reliably exceed wanvj Phase 1 (19.64).** The strongest is SCI-medium at 19.33, essentially tied with wanvj P1.
2. **SCI is shockingly fast** (~10ms/16-frame clip total) — the model is trivially small. It also has the best mean PSNR among external baselines.
3. **MIRNet underperforms on SDSD-indoor** despite being a supervised LoL-trained model (16.2 PSNR). The finetuned variant (event/club domain) is similarly weak, suggesting domain mismatch with SDSD's static indoor scenes.
4. **Per-seq variance is large.** SCI-difficult achieves 30.44 dB on pair31 (very dark seq, high SSIM=0.92), but only 13.78 on pair58 (pre-illuminated low-contrast). This is consistent with the unsupervised Retinex-style methods doing well in *truly* dark scenes and poorly otherwise.
5. **Zero-DCE has best LPIPS** (0.4510) despite mid-tier PSNR — it produces perceptually pleasing output even when not pixel-accurate.

---

## Setup pain points

- **Network access on host 30138 is severely restricted.** Only `github.com` (intermittent) and `hf-mirror.com` are reachable. Direct `drive.google.com` and `huggingface.co` are blocked. `ghfast.top` worked as a github proxy.
- **Retinexformer** (CVPR 2023, the obvious first choice with SDSD pretrained weights) **could not be run** — official weights are on Google Drive / Baidu Pan only, neither of which is reachable. There is no HF-hosted port. Adding it later requires either fetching the .pth manually onto the box, or an outbound HTTP proxy.
- **LLVE-STCD** (IJCAI 2025, the only direct video LLVE comparison) **also blocked** — pretrained weights are on Google Drive only, and additionally requires DKM dependency setup.
- **MIRNet weights via HF** worked smoothly via `huggingface_hub.hf_hub_download` once `HF_ENDPOINT=https://hf-mirror.com` was set. But the only HF-hosted MIRNet (dblasko's) is fine-tuned on a non-SDSD domain (event photography), so PSNR underperforms expectations.
- **SCI and Zero-DCE weights are baked into their github repos.** This is the only reason they were trivially deployable here.

## Recommendations for follow-up

- If a path to download files from Google Drive / Baidu opens up (VPN, proxy on host, or manual scp): grab `RetinexFormer_SDSD_indoor.pth` from the gdrive folder linked in the README of `caiyuanhao1998/Retinexformer`. The yml + script in `/cache/zihaohe/ext_methods/Retinexformer/Options/RetinexFormer_SDSD_indoor.yml` is ready to use.
- LLVE-STCD weights from gdrive `1sUzhwKWxJOjITFL8ypb3UQeofz0KI7E4` (~LLVE-STCD SDSD-indoor pretrained). DKMv3_indoor.pth already accessible via github releases (Parskatt/storage).
- MGLD-VSR / RVRT / BasicVSR++ — all on github and have weights, would be good additions if more time.

---

## File layout

```
/cache/zihaohe/ext_methods/
  SCI/                       # CVPR 2022 SCI
    CVPR/weights/{easy,medium,difficult}.pt
    run_sci.py               # inference script
  Zero-DCE/                  # CVPR 2020
    Zero-DCE_code/snapshots/Epoch99.pth
    run_zerodce.py
  MIRNet/                    # ECCV 2020
    model/MIRNet/*.py        # model defs (downloaded from HF)
    run_mirnet.py
  LLVE_STCD/                 # IJCAI 2025 (cloned, no weights -> not run)
  Retinexformer/             # ICCV 2023 (cloned, no weights -> not run)

/cache/zihaohe/visualizations/_baselines/
  sci_easy/      # 11 .mp4 + results.csv
  sci_medium/    # 11 .mp4 + results.csv  (BEST external baseline)
  sci_difficult/ # 11 .mp4 + results.csv
  zero_dce/      # 11 .mp4 + results.csv
  mirnet_pretrained/  # 11 .mp4 + results.csv
  mirnet_finetuned/   # 11 .mp4 + results.csv
  wanvj_25k_NOadain/  # (pre-existing, our model)
  nafnet_pp_30k/      # (pre-existing, our model)
```

CSV schema: `seq, psnr, ssim, lpips, psnr_lq, time_s` (last row is `MEAN`).
MP4 layout: `LQ | enhanced | GT` side-by-side, 10 fps, x264.
