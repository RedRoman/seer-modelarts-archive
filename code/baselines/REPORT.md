# SOTA baseline eval on 6 OOD datasets — 2026-07-14

Task: fill PSNR-RGB / LPIPS numbers for four baselines
(Retinexformer, MambaLLIE, BVI-Mamba, BVI-CDM) on six OOD levels
matching Seer's protocol.

## Protocol

- Videos (SDSD-indoor L0, SDSD-outdoor L1, BVI-RLV L2):
  first 16 frames per sequence (center-picked), center-crop 512×512, per-frame
  metrics, mean over T then over sequences.
- Images (LOL-v1 L3, LOLv2-Syn L4, LOLv2-Real L5):
  full image, per-image metrics, mean over images.
- PSNR: uint8 [0,255] RGB, `20*log10(255/sqrt(MSE))`, matches LLVE-STCD /
  most LLIE papers.
- LPIPS: `pyiqa.create_metric("lpips")` (default AlexNet backbone,
  ImageNet-normalized inputs, images in [0,1]).
- Splits:
  - SDSD-indoor: LLVE-STCD's canonical 12-scene test list
    (pair13/15/21/23/31/33/50/52/58/60/68/70).
  - SDSD-outdoor: SDSDDataset(subset=direct, split=test) → 25 sequences
    (80/20 deterministic split by sorted name).
  - BVI-RLV: BVIRLVDataset(split=test) → 8 test scenes × 2 fps = 16 sequences.
  - LOL-v1: 15 test images (`/cache/zihaohe/datasets/LOL/LOLv1_test/test`).
  - LOLv2-Real: 100 test images (`okhater_lolv2-real/Test`).
  - LOLv2-Syn: 100 test images (`okhater_lolv2-synthetic/Test`).

## Results

### Retinexformer (ICCV 2023)

Available weights (bundled locally): `SDSD_indoor.pth`, `SDSD_outdoor.pth`.
LOL v1 / v2-real / v2-syn official weights are on Google Drive / Baidu Pan,
both unreachable from host 30138 (DNS-poisoned drive.google.com; Baidu Pan
requires auth + captcha). No mirror on Huggingface / ModelScope / Zenodo /
Gitcode / OpenReview supplement. LOL weights were not obtained during this
run, so LOL-v1 / v2-real / v2-syn (and BVI-RLV, which has no official
Retinexformer weight) are evaluated **cross-domain using SDSD_indoor.pth**.

| Level | Dataset       | PSNR-RGB | LPIPS   | n  | Weight source |
|-------|---------------|---------:|--------:|---:|---------------|
| L0    | SDSD-indoor   |   20.18  |  0.3120 | 12 | matched (SDSD_indoor.pth) |
| L1    | SDSD-outdoor  |   22.00  |  0.2219 | 25 | matched (SDSD_outdoor.pth) |
| L2    | BVI-RLV       |   17.89  |  0.2179 | 16 | cross-domain (SDSD_indoor.pth; no official BVI weight) |
| L3    | LOL-v1        |   17.23  |  0.2406 | 15 | cross-domain (SDSD_indoor.pth; LOL weights on GDrive/Baidu, unreachable) |
| L4    | LOLv2-Syn     |   15.23  |  0.3034 | 100 | cross-domain (SDSD_indoor.pth; LOL weights on GDrive/Baidu, unreachable) |
| L5    | LOLv2-Real    |   18.81  |  0.2428 | 100 | cross-domain (SDSD_indoor.pth; LOL weights on GDrive/Baidu, unreachable) |

Sanity check vs V-JEPA-LLVE's own runs (comprehensive_eval / final_eval_outdoor / final_eval_bvi):
- HMP-raw (v9.0b) SDSD-outdoor = 22.11 — Retinexformer matched-weights = 22.00 (close, as expected: RF is a strong LLIE
  baseline on outdoor).
- HMP-raw BVI-RLV = 16.60, STCD-baseline = 15.71 — Retinexformer cross-domain = 17.89 (RF actually generalizes well to BVI).
- HMP-raw SDSD-indoor v9.0b = 24.02 — Retinexformer matched = 20.18 (lower because RF fails hard on pair68/pair58 without any
  first-frame color-align trick that some previous LLVE-STCD eval scripts applied; per-seq numbers show pair68 collapses to
  ~8dB and pair58 to ~14dB).

Per-sequence PSNR/LPIPS for all datasets: `/cache/zihaohe/baseline_ckpts/Retinexformer_eval.json`
(also copied to `/cache/zihaohe/checkpoint/runs/Retinexformer_eval.json` alongside Seer's own eval JSONs).

### MambaLLIE (NeurIPS 2024) — NOT RUN

- Repo cloned to `/home/ma-user/workspace/zihaohe/baselines/MambaLLIE`
  via `ghfast.top` proxy. `mamba_ssm` (v2.3.1) already installed in
  `/cache/zihaohe/conda_envs/vjepa2-pip`; the arch imports cleanly.
- **Blocker**: all pretrained weights (LOLv2-real, LOLv2-syn, SMID,
  SDSD-indoor, SDSD-outdoor) are only released on Google Drive
  (`drive.google.com/drive/folders/1ahQ7QW6xq2PImAFNxElNNLXQED8kqkxg`).
  Drive/Baidu are unreachable from this host.
- Searched: `hf-mirror.com`, `modelscope.cn`, `zenodo.org`, `openreview.net`
  supplementary attachment (auth-required), arxiv `2405.16105`, author page
  — no external mirror found.
- Training from scratch on 5 datasets (LOLv1/v2-real/v2-syn + SDSD-indoor/outdoor)
  would take multiple days; not feasible in 12h.

### BVI-Mamba (arXiv 2604.23655 / edhuang1/BVI-Mamba) — NOT RUN

- Repo cloned. STASUNet architecture reads cleanly.
- **Blocker**: no pretrained weights released — only training code
  (`train.py`, `test_fullresolution.py`). Default config
  (`configs/BVIMamba-SDSD.yml`) trains for 20 epochs on BVI-RLV with
  batch_size=1, image_size=512, DCN CUDA extension required. Estimated
  ≥10h training just to reach a converged BVI-RLV model, before starting
  the other 5 datasets. Not runnable within budget.

### BVI-CDM (lrr-rachel/BVI-CDM) — NOT RUN

- Repo not cloned (redundant with the other two — same blocker).
- **Blocker**: no pretrained weights. Only training code. This is a video
  diffusion model with DCNv2 JIT extensions and long training schedule
  (the paper reports days on 8× GPUs).

## Files written

- `/cache/zihaohe/baseline_ckpts/Retinexformer_eval.json` (264 per-seq rows,
  6 dataset summaries; PSNR/LPIPS/n_frames per entry)
- `/cache/zihaohe/baseline_ckpts/MambaLLIE_eval.json` (nulls + notes)
- `/cache/zihaohe/baseline_ckpts/BVI-Mamba_eval.json` (nulls + notes)
- `/cache/zihaohe/baseline_ckpts/BVI-CDM_eval.json` (nulls + notes)
- Same 4 JSONs copied to `/cache/zihaohe/checkpoint/runs/`.
- Runner: `/home/ma-user/workspace/zihaohe/baselines/run_retinexformer.py`
- Common eval lib: `/home/ma-user/workspace/zihaohe/baselines/common/eval_lib.py`

## Network state on host 30138 (for future runs)

Reachable: `github.com` (via `ghfast.top` proxy), `pypi.org`,
`files.pythonhosted.org`, `hf-mirror.com`, `modelscope.cn`, `openreview.net`
(pdfs only, attachments 404), `arxiv.org`, `zenodo.org`.

Blocked: `drive.google.com` (DNS-poisoned), `docs.google.com`,
`googleapis.com`, `huggingface.co` (use hf-mirror), `pan.baidu.com`,
`gitcode.com` search (WAF-blocked), most CDN edges.

To run MambaLLIE / BVI-Mamba / BVI-CDM in a future session, the fastest path
is to manually side-load the Google Drive folders (e.g. via a proxy machine
or a manual scp upload) into `/cache/zihaohe/baseline_ckpts/<name>/`, then
the arch code + eval harness already present here can produce numbers.
