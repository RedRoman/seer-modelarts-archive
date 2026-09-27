# LLVE-STCD-Style Audit Summary

Protocol: RGB PSNR/SSIM on uint8 full frames, N_frames=5 center-frame evaluation, no Y conversion, no crop.

## Split Report

- Sequence mode: `auto`
- Directly comparable to LLVE-STCD 12 scenes: `False`
- Fallback used: `False`
- Intersection: `pair13, pair15, pair21, pair23, pair31, pair33`
- Selected: `pair13, pair15, pair21, pair23, pair31, pair33`

| Model | RGB-full PSNR seq | RGB-full SSIM seq | RGB-full PSNR frame | RGB-full SSIM frame | Seq | Frames |
|---|---:|---:|---:|---:|---:|---:|
| LQ_baseline | 8.064 | 0.3578 | 8.064 | 0.3578 | 6 | 6 |
