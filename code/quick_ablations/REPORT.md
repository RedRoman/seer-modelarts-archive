# Quick Ablations Report

**Host:** 30138 (A100-SXM4-80GB, ma-user)
**Working dir:** `/home/ma-user/workspace/zihaohe/quick_ablations/`

---

## TL;DR

- **Task A (M1/M3 leave-one-out): BLOCKED.** The v35 Seer model source
  (`VJepaPixelBranch` with `dec_igab`, `illu_estimator`, `illu_proj`,
  `router_logits`, `cf_refiner` submodules) has been deleted from
  `/home/ma-user/workspace/zihaohe/vjepa_llve/models/`. The checkpoint
  `/cache/zihaohe/checkpoint/runs/v35_S4_indoor_ft/best_y512.pt` (1.5 GB,
  962 pixel_branch + 185 cf_refiner keys) exists, but there is no Python
  class definition to load it into. Full search of `/home`, `/cache`,
  `/root`, `/tmp` and site-packages found no source or importable `.pyc`
  matching keys like `dec_igab`, `router_logits`, `cf_refiner`, or
  `illu_estimator`. Delivered instead: diagnostic dump of the *trained*
  M1 gate value and M3 router probabilities — the closest read of what LOO
  would remove. See "Task A – diagnostics".
- **Task B (Complexity): DONE for 6 baselines.** Seer partial — parameter
  counts and V-JEPA-encoder-only cost measured; full-pipeline GFLOPs /
  peak-mem / wall-clock require the missing v35 source, marked NaN.

---

## Task A – checkpoint-derived diagnostics for M1 and M3

Source: `/home/ma-user/workspace/zihaohe/quick_ablations/m1m3_checkpoint_diagnostics.json`.

### M1 (counterfactual-predictor gate `g = σ(w)`)

Checkpoint tensor: `cf_refiner.gate` shape (1,).

| quantity | value |
|---|---|
| raw scalar `w` | 4.881e-05 |
| `σ(w)` (gate at inference) | **0.500012** |

Interpretation: `w` never moved from its zero initialisation, so the gate
sits at exactly 0.5. This means the M1 counterfactual predictor contributes
at half-strength. Forcing `σ(w) = 0` (i.e. `no_M1`) would be a materially
large intervention.

### M3 (cross-layer router `π_s^{(l)}`)

Checkpoint tensor: `pixel_branch.router_logits` shape (5, 4) — 5 scales × 4
candidate V-JEPA layers.

Learned logits (rounded):

```
scale 0:  [ 4.000,  0.000,  0.000,  0.000]   top-1 = layer 0  (94.79%)
scale 1:  [ 4.001, -0.000, -0.000, -0.002]   top-1 = layer 0  (94.80%)
scale 2:  [-0.001,  4.002, -0.001, -0.001]   top-1 = layer 1  (94.81%)
scale 3:  [-0.001, -0.001,  4.000,  0.000]   top-1 = layer 2  (94.79%)
scale 4:  [-0.003, -0.003, -0.003,  4.003]   top-1 = layer 3  (94.82%)
```

The router has collapsed to a near one-hot per scale (~94.8% top-1). `no_M3`
would replace 94.8 %→100 % on the top layer — a ~5 pp shift. Router picks
exactly the same mapping as v9's hand-picked `vjepa_layer_per_scale =
(0, 0, 1, 2, 3)`.
**Prediction:** `no_M3` PSNR delta ≤ 0.05 dB. The M3 story is "the router
converges to the right layer selection", not "the router materially
changes PSNR at inference".

### Table 1 — Full Seer reference numbers (from existing eval JSONs; LOO rows blocked)

Sources: `/cache/zihaohe/checkpoint/runs/comprehensive_eval.json`
(v35-S4 row), `final_eval_outdoor.json`, `final_eval_bvi.json`.

| Variant | L0 SDSD-in (PSNR_RGB / Y) | L1 SDSD-out (PSNR_RGB / Y) | L2 BVI (PSNR_RGB / Y) |
|---|---|---|---|
| Full Seer (v35-S4, raw)          | 23.738 / 25.062 | 22.109 / 23.336 | 16.650 / 18.378 |
| Full Seer (v35-S4, +CP post)     | —               | 21.945 / 23.319 | 17.835 / 18.376 |
| no_M1 (σ(w) := 0)                | **N/A — source deleted** | N/A | N/A |
| no_M3 (router := one-hot)        | **N/A — source deleted** | N/A | N/A |

---

## Task B – Model complexity comparison

Source: `/home/ma-user/workspace/zihaohe/quick_ablations/model_complexity.json`
Runner: `/home/ma-user/workspace/zihaohe/quick_ablations/model_complexity.py`

Hardware: A100-SXM4-80GB. Timing = mean of middle 60% of 20 runs after 5
warmups, GPU-synced. Memory = `torch.cuda.max_memory_allocated` around one
fresh forward. FLOPs = `fvcore.nn.FlopCountAnalysis.total()`, falling back
to `thop.profile × 2` for V-JEPA where fvcore's jit tracer fails.

### Table 2 — Complexity

| Method | Params (M) | GFLOPs | GPU mem (MB) | Time (ms) | Input |
|---|---:|---:|---:|---:|---|
| Retinexformer         |   1.61 |   66.86 |   626 |  73.9 | 1x3x512x512 (single image) |
| HVI-CIDNet            |   1.98 |   31.65 |   386 |  80.6 | 1x3x512x512 (single image) |
| DarkIR                |   3.32 |   28.37 |   442 |  16.6 | 1x3x512x512 (single image) |
| URCSA                 |   3.11 |  436.92 |  1354 |  32.2 | 1x3x512x512 (single-image variant) |
| MambaLLIE             |   2.29 |   89.56 |  2448 | 131.3 | 1x3x512x512 (no weights on host) |
| BVI-Mamba (STASUNet)  |  10.00 |  188.24 |   965 | 182.3 | 1x5x3x512x512 (5-frame native) |
| **Seer — V-JEPA encoder only**  | 303.89 | 1244.33 (thop) | 1415 | 116.9 | 1x3x16x512x512 |
| **Seer — trainable branches**   | 392.77 |   NaN  |  NaN | NaN | (source deleted) |
| **Seer — TOTAL**                | **696.66** (304.68 frozen + 392.77 trainable) | NaN | NaN | NaN | 1x3x16x512x512 |

Seer sub-part params (from checkpoint state_dict):

- `pixel_branch` — 369.75 M trainable (Restormer-style UNet + Retinexformer
  IGAB + illumination modules + 5×4 router).
- `cf_refiner` — 23.02 M trainable (V-JEPA counterfactual predictor with
  LoRA-adapted qkv, 12 predictor blocks, mask_tokens).
- **V-JEPA-2.1 ViT-L encoder** — 304.68 M frozen (measured from
  `/cache/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt` `encoder`
  state_dict; matches thop-measured 303.89 M when instantiated).

### Failure / caveat notes

1. **Seer full-pipeline complexity is unmeasurable here.** v35 model classes
   are gone. V-JEPA encoder alone is 1244 GFLOPs / 1.4 GB / 117 ms. The pixel
   branch (369.7 M params, 5 scales up to 1024 channels, IGAB blocks) on a
   16-frame clip probably dominates. Rough order-of-magnitude estimate:
   **3–5 × 10³ GFLOPs total, wall-clock ≥ 300 ms per 16-frame clip.**
   *Do not cite this estimate; only the encoder-only number is a real
   measurement.*
2. **URCSA needs a wrapper.** Native forward is `(lq, lq, mode="test")`;
   fvcore/thop only pass one input. Fix: wrap with a single-input
   `nn.Module` shim (done in `model_complexity.py`).
3. **fvcore fails on V-JEPA-2.1** with a jit-tracer data-dependence error.
   Fell back to `thop` (`GFLOPs ≈ 2 × MACs`).
4. **Retinexformer / HVI-CIDNet / DarkIR / URCSA / MambaLLIE are
   single-image models.** To compare to Seer's per-clip cost, multiply by
   16 frames (e.g. Retinexformer per-clip ≈ 1070 GFLOPs / 1183 ms).
5. **BVI-Mamba** instantiated with the paper's `BVIMamba.yml` config
   (`num_feat=embed_dim=16, depths=[8,8,8,8], num_heads=[8,8,8,8],
   num_frames=5, hr_in=True, window_size=8`). Requires 5-frame clips (PCD
   alignment with `center_frame_idx=2`).
6. **MambaLLIE and BVI-Mamba have no pretrained weights on this host.**
   Arch complexity is valid; PSNR/LPIPS not available in any eval JSON.
7. **STCD not installed** under `/home/ma-user/workspace/zihaohe/baselines/`;
   omitted.
8. **einops × torch._dynamo circular-import bug** in this pytorch 2.3.1 /
   einops build trips fvcore's rearrange. Workaround: force
   `import torch._dynamo` before any einops call. Applied at the top of
   `model_complexity.py`.

---

## Files produced

- `/home/ma-user/workspace/zihaohe/quick_ablations/model_complexity.py` — measurement script
- `/home/ma-user/workspace/zihaohe/quick_ablations/model_complexity.json` — raw numbers
- `/home/ma-user/workspace/zihaohe/quick_ablations/m1m3_checkpoint_diagnostics.json` — M1 gate / M3 router diagnostics
- `/home/ma-user/workspace/zihaohe/quick_ablations/complexity.log` — runtime log
- `/home/ma-user/workspace/zihaohe/quick_ablations/REPORT.md` — this file

No files under `/home/ma-user/workspace/zihaohe/vjepa_llve/` were modified.
