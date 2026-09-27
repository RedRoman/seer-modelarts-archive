# V-JEPA-LLVE Project Handoff Document
> For new reviewer joining mid-project, 2026-05-04.
> Current state: **v11 JDAP architecture, predicted_delta works strongly but killer test fails. Awaiting next-step decision.**

---

## 1. Project Goal

CVPR 2027 paper on **Low-Light Video Enhancement (LLVE)** using **V-JEPA 2.1** (ViT-L) as the architectural centerpiece.

Hard constraint: V-JEPA must be **load-bearing**. If the architecture would work identically with a generic backbone (DINOv2 / MAE / VideoMAE), there is no contribution.

**Target benchmark splits** (locked, RGB-PSNR uint8 tile=512/overlap=128):

| split | n | LQ mean | v9.0b_mix baseline | LLVE-STCD reported |
|---|---:|---:|---:|---:|
| LLVE-STCD-12 (primary indoor hard) | 12 | 8.82 | **20.868** | 28.93 |
| outdoor-25 | 25 | 13.07 | **21.779** | — |
| original indoor-14 (legacy guard) | 14 | 8.82 | 25.738 | — |
| BZ indoor-15 (ablation only) | 15 | — | 27.016 | — |

Same-pipeline gap to LLVE-STCD's 28.93 = 8.06 dB. Realistic v11 target: close ~1-3 dB of that gap with a justified V-JEPA-specific contribution.

---

## 2. Project History (Abridged)

| version | what was done | why abandoned |
|---|---|---|
| v6/v7/v8 | Retinex compose + LPIPS + temporal smoothness (loss engineering) | not architectural |
| v9.0/v9.0b | RAFT-supervised reflectance / PSNR finetune — **current best** baseline | not architectural |
| v9.4 series | tile=256 protocol, deeper pixel branch ablations | found protocol bug; tile=512 became canonical |
| v10 | M1 3D positional embedding + M5 mean-color loss + M2 crop curriculum + M7 fixed-protocol | **ABORTED by author** (CY): all engineering, no V-JEPA-specific architectural claim |
| v11 (current) | JDAP = JEPA-Delta Action Prior | active; partial success |

**v10 was killed** because every change was protocol/loss/positional engineering. None of v10 used V-JEPA in a way DINOv2/MAE could not replace. CVPR cannot be written from that.

---

## 3. v11 Architecture: JDAP (JEPA-Delta Action Prior)

### Hypothesis (single, falsifiable)
> V-JEPA is useful for LLVE only if we learn a target-conditioned latent restoration action `delta = z_H - z_L`, **not** if we merely attend to low-light tokens.

### Architecture
```
LQ video                                       HQ video (training only)
   |                                                |
   v                                                v
V-JEPA 2.1 ViT-L (frozen, 304M)            V-JEPA 2.1 ViT-L (frozen)
   |                                                |
   v                                                v
z_L_layers (4 layers, B,N=2048,1024)        z_H_layers (4 layers)
   |                                                |
   |  delta_oracle = z_H - z_L  (training-only teacher signal)
   v
JEPADeltaAdapter (compact, 14.7M trainable)
  delta_pred = A(z_L)
   |
   v
DeltaModulatedDecoder (compact, 33.9M trainable)
  passive z_L → cross-attn at deepest 2 scales (KV)
  delta → AdaLN modulation at every scale (zero-init)
  3D U-Net residual to LQ → pred RGB
```

**Key implementation details:**
- Decoder stem downsamples 2× spatially (avoids 256² 3D-conv OOM)
- Cross-attn only at scales 3,4 (small spatial, manageable memory)
- AdaLN zero-init → identity-at-init when delta=0
- 5-scale U-Net: dims (32,48,96,192,256), blocks (1,2,2,3,4)

### Spec files
- `V11_JDAP_SPEC.md` — original design (now stale per DP)
- `V11_PROBE_C_SPEC.md` — Probe C (oracle decoder upper bound)

---

## 4. Probe Results So Far

### Probe A — Does delta exist? **PASS**

`tools/v11_delta_stats.py` computed cos(z_L, z_H) and ||z_H-z_L||/||z_L|| across 33 clips.

| layer_idx | cos_mean | rel_delta_mean | verdict |
|---:|---:|---:|---|
| 0 (V-JEPA layer 5) | 0.957 | 0.287 | PASS |
| 1 (V-JEPA layer 11) | 0.955 | 0.291 | PASS |
| 2 (V-JEPA layer 17) | 0.954 | 0.287 | PASS |
| 3 (V-JEPA layer 23) | **0.809** | **0.595** | PASS (strongest) |

V-JEPA is not illumination-invariant; deepest layer carries strong signal.

Artifact: `/cache/zihaohe/checkpoint/v11_probes/delta_stats/`

### Probe B' — Adapter beats trivial baselines in latent space? **PASS**

`tools/v11_train_delta_adapter.py` trained 14.7M `JEPADeltaAdapter` with cos+L2 loss (alpha=1.0), 2000 iter, sequence-disjoint LLVE-STCD-12 held-out.

Final eval table (all-4-layer Gate A strict):
| layer | cos_id | cos_adapter | L2_id | L2_adapter | verdict |
|---:|---:|---:|---:|---:|---|
| 0 | 0.948 | **0.970** | 0.320 | **0.240** | PASS |
| 1 | 0.946 | 0.961 | 0.327 | 0.276 | PASS |
| 2 | 0.947 | 0.961 | 0.325 | 0.277 | PASS |
| 3 | 0.770 | **0.813** | 1.272 | **1.122** | PASS |

Adapter beats identity / mean-delta / affine baselines on every layer in cos AND L2. Recovery of true delta direction: cos(delta_pred, delta_true) = 0.45–0.66.

Artifact: `/cache/zihaohe/checkpoint/v11_probes/probe_b_prime/delta_adapter.pt`

### Probe C long-crop — Oracle works hugely; predicted hurts. **PARTIAL PASS**

`tools/v11_oracle_delta_probe.py`. Trained DeltaModulatedDecoder under 7 conditions, same init (sha256[:16]=c278657cff204fc4), 500 iter each, fixed center-crop 256 eval on LLVE-STCD-12.

| condition | mean PSNR | Δ vs identity_delta |
|---|---:|---:|
| no_prior | 17.899 | -0.566 |
| passive_zL | 17.723 | -0.742 |
| identity_delta | 18.465 | 0.000 |
| shuffled_delta | 17.961 | -0.504 |
| oracle_zH | 18.106 | -0.359 |
| **oracle_delta** | **22.230** | **+3.765** |
| **predicted_delta** | **17.530** | **-0.935** |

**Findings:**
- Oracle delta works massively (+3.77 above identity).
- Wrong delta is harmful (shuffled −0.50 below identity, killer pass).
- **Adapter from Probe B' fails in RGB**: predicted_delta is WORSE than zero delta. The cos+L2-trained adapter is RGB-useless.

Conclusion: decoder-side delta interface is correct; adapter training objective is wrong (latent metrics ≠ RGB metrics).

Artifact: `/cache/zihaohe/checkpoint/v11_probes/probe_c_long_crop256/`

### v11 jdap_rgbdistill_500 — RGB-aligned distill. **MIXED**

`tools/v11_jdap_rgbdistill.py`. Trained adapter+decoder jointly under RGB Charbonnier loss + scheduled action forcing (m: 1→0) + ranking loss (margin 0.005, weight 0.1) + scale-aware adapter wrapper.

Same crop-256 eval, 4 conditions:

| condition | mean PSNR | Δ vs identity | Notes |
|---|---:|---:|---|
| identity_delta | 11.702 | 0.000 | retrained baseline |
| oracle_delta | 9.446 | -2.256 | distribution-shift artifact (m→0 made decoder forget oracle) |
| **predicted_delta** | **19.054** | **+7.352** | huge improvement vs Probe C |
| shuffled_predicted_delta | 14.898 | +3.196 | killer FAIL |

**Recovery vs Probe C oracle**: (19.054 − 11.702) / (22.230 − 11.702) = **70%**. Far above DP6 30% gate.

**3 issues:**
1. predicted_delta works strongly (+7.35) — first real RGB-level v11 signal ✓
2. oracle_delta dropped — scheduled-sampling distribution shift, expected, not a JDAP failure
3. shuffled_predicted_delta = +3.2 above identity — adapter learned generic restoration prior that helps even with wrong content. Killer test FAIL.

Artifact: `/cache/zihaohe/checkpoint/v11_probes/v11_jdap_rgbdistill_500/`

---

## 5. Current Open Question (REVIEW SECTION DR)

The author posted **DR** in `VJEPA_LLVE_CODE_REVIEW.md` (line ~23728+) summarizing the jdap_rgbdistill_500 result and asking the reviewer to decide:

1. Is predicted_delta = +7.35 / 70% recovery enough to call v11 first signal? Or is killer fail blocking?
2. Which of these to run next:
    - **A**: anneal m only to 0.3 (preserve oracle path) — ~1.5 hr
    - **B**: ranking_weight=1.0 (10× stronger killer enforcement) — ~1.5 hr
    - **C**: A+B+1000 iter — ~3 hr
    - **D**: accept and write up
3. Should BVI-RLV smoke test be started in parallel?
4. Update V11_JDAP_SPEC.md to reflect current architecture?

The author's recommendation: **B then C** if B helps killer; otherwise reconsider whether adapter learned a generic prior versus a content-specific action.

---

## 6. File Map

### Core code
```
models/vjepa_backbone.py            V-JEPA 2.1 wrapper (frozen)
models/v11/jepa_delta_adapter.py    14.7M adapter (Probe B' main)
models/v11/scale_aware_adapter.py   wrapper adding per-layer gain+confidence (DP5)
models/v11/delta_modulated_decoder.py  33.9M U-Net + AdaLN modulation
losses/v9.py                        v9 losses (used as RGB Charbonnier source)

tools/v11_delta_stats.py            Probe A: latent statistics
tools/v11_train_delta_adapter.py    Probe B': latent adapter training
tools/v11_oracle_delta_probe.py     Probe C: 7-condition decoder probe
tools/v11_jdap_rgbdistill.py        DP6 main: RGB-aligned student training
tools/v10_audit.py                  4-split eval (tile=512), reused for v11

data/sdsd_dataset.py                SDSD indoor + outdoor
data/bvi_rlv_dataset.py             BVI-RLV (just written, not smoke-tested)
```

### Datasets
```
/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor/   (used)
/home/ma-user/workspace/zihaohe/datasets/SDSD_outdoor/  (used)
/cache/zihaohe/datasets/BVI_RLV/extracted/              (just unzipped, 65972 PNG)
```

### Checkpoints
```
/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt      v9.0b_mix anchor (current paper-grade baseline source)
/cache/zihaohe/checkpoint/v11_probes/probe_b_prime/delta_adapter.pt
/cache/zihaohe/checkpoint/v11_probes/probe_c_long_crop256/oracle_delta/ckpt.pt
/cache/zihaohe/checkpoint/v11_probes/v11_jdap_rgbdistill_500/ckpt.pt
```

### V-JEPA backbone weights
```
/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt
```

### Audits / baselines
```
/cache/zihaohe/checkpoint/v10_audit/v9_0b_mix_sentinel_baseline/sentinel_baseline.json
  (v9.0b_mix on 5-seq sentinel panel at tile=512: pair68=13.40, pair31=17.67,
   MVI_1030=14.92, MVI_1019=13.61, pair6=27.56, mean=17.43)
```

---

## 7. How to Engage with the Review Thread

The full thread is in `VJEPA_LLVE_CODE_REVIEW.md` (~24K lines, do not read sequentially).

**Suggested entry points for new reviewer:**
- **CY** (line ~20201): author self-critique that triggered v11 pivot
- **CZ** (line ~20427): previous reviewer's lock of JDAP hypothesis
- **DH** (line ~21528): Probe B' approval + Probe C design
- **DM** (line ~22601): Probe C smoke approval
- **DN** (line ~22828): Probe C long-crop results (oracle big, predicted dies)
- **DP** (line ~23426): aggressive pivot to RGB distill (the basis for current jdap_rgbdistill_500 run)
- **DR** (line ~23728): current results, awaiting your call

**Section letter convention**: alternating reviewer (single uppercase) and author (next letter). Latest is `DR` (author). Your reply should be `DS`.

---

## 8. Decision Points You Will Need to Address

In rough priority order:

1. **Killer test fail.** shuffled_predicted_delta gives +3.2 dB above identity. JDAP requires content-specific action; this leaks generic prior. How to fix:
    - (a) ranking_weight=1.0 + harder margin
    - (b) detach decoder from adapter early in training (so adapter must produce content-correct delta to get RGB grad)
    - (c) reformulate the adapter objective entirely

2. **Schedule tuning for joint training.** m anneal 1→0 broke oracle path. Should it stop at 0.3? Random per-batch mix?

3. **Per-layer or layer-23-only adapter?** Probe A says layer 23 carries 60% of delta; shallow layers are noisy. Should JDAP narrow to layer 23 only? (DC2 noted any narrowing requires explicit spec change before judging.)

4. **Eval protocol.** All probes used crop=256. v9.0b baselines are tile=512 full-frame. Final v11 paper number requires tile=512 eval. When to switch?

5. **BVI-RLV cross-dataset.** Data ready, loader written, not smoked. When to start using it (CJ5 originally said only after first v11 result was known).

6. **Spec update.** V11_JDAP_SPEC.md is stale — describes original latent-only adapter. After architecture locks, needs full rewrite.

---

## 9. Compute / Environment

- 1× A100 80GB on host port 30138 (HuggingFace ModelArts)
- conda env: `/cache/zihaohe/conda_envs/vjepa2-pip` (torch 2.3.1+cu121, V-JEPA 2.1 from local clone)
- **launch script**: `scripts/train_v10.sh` — bakes in `LD_LIBRARY_PATH=$CUDNN:$LD_LIBRARY_PATH` for cuDNN issue (CN-F)
- Disk: 9.3 TB / cache, 6.9 TB free

To run any v11 probe:
```bash
CUDNN=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$CUDNN:$LD_LIBRARY_PATH
source /home/ma-user/miniconda3/bin/activate /cache/zihaohe/conda_envs/vjepa2-pip
cd /home/ma-user/workspace/zihaohe/vjepa_llve
CUDA_VISIBLE_DEVICES=0 python tools/v11_jdap_rgbdistill.py --help
```

---

## 10. Open Risks

1. **JDAP claim still unproven for RGB.** Predicted-delta works in jdap_rgbdistill_500 but killer fails. If killer cannot be made to pass, JDAP loses its V-JEPA-specific story.
2. **Crop-256 eval is not the production protocol.** Numbers above are not directly comparable to v9.0b's tile=512 = 20.87 on LLVE-STCD-12.
3. **Same-pipeline gap to SOTA (LLVE-STCD's 28.93) is 8 dB.** Even strong v11 will likely close 1-3 dB of this. Need to discuss whether that is a publishable contribution.
4. **Author has been compressing scope to fit ~1 month timeline.** Major scope changes risk missing CVPR window.

---

*End of handoff. Welcome to the project. Read CY → CZ → DH → DN → DP → DR in sequence to fully orient. Any questions, reply to DR with section DS.*
