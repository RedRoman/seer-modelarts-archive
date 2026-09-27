# SOTA baseline eval v2 (2026-07-14) — HVI-CIDNet, DarkIR, URCSA

Follow-up to `REPORT.md` (Retinexformer + 3 blocked baselines). Adds three more
methods, evaluated with the same protocol.

## Protocol (unchanged from v1)

- Videos (SDSD-indoor L0, SDSD-outdoor L1, BVI-RLV L2): first 16 frames per
  seq, center-crop 512x512, per-frame metrics, mean over T then over
  sequences.
- Images (LOL-v1 L3, LOLv2-Syn L4, LOLv2-Real L5): full image, per-image
  metrics, mean over images.
- PSNR: uint8 [0,255] RGB, `20*log10(255/sqrt(MSE))` (matches Seer's
  LLVE-STCD protocol).
- LPIPS: `pyiqa.create_metric("lpips")` (AlexNet, ImageNet-normalized, [0,1]).
- No GT-mean matching (HVI-CIDNet's paper numbers use `gated=True` /
  `gated2` with a learned alpha; we keep those flags off to remain
  faithful to Retinexformer's no-GT-mean setting).

## Results

| Method        | L0 SDSD-indoor    | L1 SDSD-outdoor   | L2 BVI-RLV        | L3 LOL-v1         | L4 LOLv2-Syn      | L5 LOLv2-Real     |
|---------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| Retinexformer | 20.18 / 0.3120    | 22.00 / 0.2219    | 17.89 / 0.2179    | 17.23 / 0.2406    | 15.23 / 0.3034    | 18.81 / 0.2428    |
| HVI-CIDNet    | 16.47 / 0.4913    | 18.51 / 0.3062    | 18.93 / 0.1902    | 23.21 / 0.0999    | 25.13 / 0.0449    | 20.60 / 0.1274    |
| DarkIR        | 16.45 / 0.4018    | 18.75 / 0.3117    | 20.46 / 0.1500    | 17.09 / 0.2109    | 18.77 / 0.1909    | 19.92 / 0.2316    |
| URCSA         | 19.98 / 0.3339    | 19.60 / 0.2585    | 18.38 / 0.2212    | 21.88 / 0.1104    | 16.09 / 0.2302    | 24.84 / 0.0809    |

Format: `PSNR-RGB (uint8) / LPIPS (pyiqa AlexNet)`.

### Status

| Method     | Status              | Notes                                                                                 |
|------------|---------------------|---------------------------------------------------------------------------------------|
| HVI-CIDNet | OK                  | LOL splits matched (HF `Fediory/HVI-CIDNet-*`); SDSD/BVI use `Generalization.bin` cross-domain. |
| DarkIR     | partial-cross-domain| Only `DarkIR_384.pt` (LOL-Blur trained, DarkIR-m width=32) available on HF; LOL-multi-task `DarkIR_1k_cr_mt.pt` is OneDrive-only + auth-gated (403). All 6 datasets run cross-domain with the LOL-Blur weight. |
| URCSA      | OK                  | SDSD indoor/outdoor matched (`RCSAindoor.pth`, `RCSAoutdoor.pth`); BVI cross-domain via indoor; all 3 LOL splits use the single published `URCSA/lol/checkpoint.pth`. **Required a source-code patch** — see below. |

## Method-specific notes

### HVI-CIDNet (CVPR 2025)

- Repo: https://github.com/Fediory/HVI-CIDNet (cloned to `baselines/HVI-CIDNet/`).
- Weights pulled from HuggingFace via `HF_ENDPOINT=https://hf-mirror.com`:
  - `Fediory/HVI-CIDNet-LOLv1-wperc` -> LOL-v1 matched
  - `Fediory/HVI-CIDNet-LOLv2-real-bestPSNR` -> LOLv2-Real matched
  - `Fediory/HVI-CIDNet-LOLv2-syn-wperc` -> LOLv2-Syn matched
  - `Fediory/HVI-CIDNet-Generalization` -> SDSD_indoor, SDSD_outdoor, BVI_RLV cross-domain
    (random-gamma-trained LOLv2-Syn checkpoint recommended by the author for
    cross-dataset use)
- `model.trans.gated=False`, `model.trans.gated2=False` (no GT-mean matching).
- LOL-v1 matched PSNR = 23.21 dB (paper w/o GT-mean = 23.5 dB); LOLv2-Syn
  matched = 25.13 dB (paper 25.1); LOLv2-Real matched = 20.60 dB (paper
  23.4, but that uses GT-mean w/ perc loss).

### DarkIR (CVPR 2025, 3x NTIRE 2025)

- Repo: https://github.com/cidautai/DarkIR (cloned to `baselines/DarkIR/`).
- Only `DarkIR_384.pt` (LOL-Blur trained, DarkIR-m width=32) is public on
  HuggingFace (`Cidaut/DarkIR`). The multi-task `DarkIR_1k_cr_mt.pt` is
  hosted on `cidautes-my.sharepoint.com` (OneDrive-for-Business) and
  returns 403 without SSO auth. No mirror on ModelScope / Zenodo /
  HuggingFace search.
- All 6 datasets evaluated with the LOL-Blur weight cross-domain. Numbers
  are therefore lower-bounded by domain mismatch (LOL-Blur has both blur
  and noise; SDSD/BVI have neither, LOL is denoised-only).
- Padding: reflect-pad to multiple of 8 before forward.

### URCSA (Computer Graphics Forum 2024)

- Repo: https://github.com/cq-dong/URCSA (cloned to `baselines/URCSA/`).
- **Pretrained ckpts are shipped in the repo**: `sdsd/RCSAindoor.pth`,
  `sdsd/RCSAoutdoor.pth`, `lol/checkpoint.pth`. So this method **is not**
  train-only.
- **Bug in released source**: the three checkpoints reference two
  `deline` fusion modules on the `low_block`:
  - `block.ende_quarres_up_deline`   : 1x1 conv 256->128 (2*dim -> 2*dim)
  - `block.ende_quarres_up_up_deline`: 1x1 conv 128->64  (2*dim -> dim)
  The released `sdsd_basemodel.py` / `lol/base_model.py` don't define these;
  they use pure residual additions at the up-fusion stages. Loading with
  `strict=False` silently drops the two modules and the network
  over-brightens by ~2x (SDSD-indoor PSNR collapsed to 8.4 dB before the
  fix, 20.0 dB after). We monkey-patch `low_block.__init__` /
  `low_block.forward` in `run_urcsa.py` to (a) create the two conv blocks
  and (b) use `concat -> 1x1 conv` at the 4x-up and 2x-up-of-up stages.
  After the patch the state_dict loads strictly.
- SDSD variant's `low_block` returns a `(out, out)` tuple; LOL variant's
  returns a single tensor. We keep both signatures in the patch.

### Retinexformer (v1, unchanged)

- Reported here for completeness; from `REPORT.md`.

## Where things live on host 30138

- Runners: `/home/ma-user/workspace/zihaohe/baselines/run_hvi_cidnet.py`,
  `run_darkir.py`, `run_urcsa.py`.
- Qualitative-crop dumper: `/home/ma-user/workspace/zihaohe/baselines/rerun_matched_v2.py`.
- Weights: `/cache/zihaohe/baseline_ckpts/{HVI-CIDNet,DarkIR}/`;
  URCSA weights already live inside the cloned repo at
  `/home/ma-user/workspace/zihaohe/baselines/URCSA/{sdsd,lol}/`.
- Per-seq eval JSONs (per-frame PSNR/LPIPS for every scene):
  - `/cache/zihaohe/baseline_ckpts/HVI-CIDNet_eval.json`
  - `/cache/zihaohe/baseline_ckpts/DarkIR_eval.json`
  - `/cache/zihaohe/baseline_ckpts/URCSA_eval.json`
  Same three files also copied to `/cache/zihaohe/checkpoint/runs/`.

## Qualitative-crop outputs (494x512 each)

`/tmp/qual_v2/` layout:
- `HVI_CIDNet/bvi_Books5_t08.png`, `HVI_CIDNet/bvi_Mario3_t08.png`, `HVI_CIDNet/outdoor_MVI_1009_t08.png`
- `DarkIR/bvi_Books5_t08.png`, `DarkIR/bvi_Mario3_t08.png`, `DarkIR/outdoor_MVI_1009_t08.png`
- `URCSA/bvi_Books5_t08.png`, `URCSA/bvi_Mario3_t08.png`, `URCSA/outdoor_MVI_1009_t08.png`

All 9 outputs are exactly 494x512 (verified via imageio load; padding of 2
rows to 496 before forward, then cropped back).

## Time budget

Full run wall-time (start-to-finish including debugging URCSA weight bug):
~40 minutes. GPU was solely occupied by these runs; no interference on
GPU 0 during any run.
