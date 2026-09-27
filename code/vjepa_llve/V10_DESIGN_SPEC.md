# V10_DESIGN_SPEC

V-JEPA-LLVE v10 architecture, training, and evaluation specification.

```
Author:       project author
Reviewer:     project reviewer (review thread VJEPA_LLVE_CODE_REVIEW.md sections CA-CH)
Date:         2026-05-03 (v2 patch per CH)
Status:       LOCKED for implementation after CH-edits applied
Successor of: v9.0b_mix_best
```

> **v2 changes vs v1**:
> - Removed M4a from MVP (D1b verifies content-tied not position-tied). v10 MVP = 4 mandatory.
> - Fixed-grid V-JEPA (8,16,16) statement — wrapper resizes to 256 regardless of crop (CH1).
> - Identity-at-init via gamma_pos=1 + P_*=0 (CH3 + CL1 fix; gamma=0 dead-gradient was caught in CL1 review).
> - v9.0b loss config preserved (lpips_weight=0) — color loss is the only addition (CH4).
> - Numeric corrections in §7 (CH5).
> - Sentinel validation panel (CH6).

---

## 1. Fixed benchmark ledger (CB0)

All v10 evaluation MUST use these 4 splits with the locked tile=512 / overlap=128 RGB-PSNR uint8 full-frame tiled protocol. Numbers are v9.0b_mix_best baselines.

| split                      | n  | LQ mean | v9.0b_mix mean | role                            |
|----------------------------|---:|--------:|---------------:|---------------------------------|
| **LLVE-STCD-12**           | 12 |    8.82 |     **20.868** | primary hard indoor (SOTA anchor)|
| **outdoor-25**             | 25 |   13.07 |     **21.779** | primary outdoor                 |
| original indoor-14         | 14 |    8.82 |     25.738     | legacy sanity (guard against regression) |
| BZ 15-seq                  | 15 |       — |     27.016     | ablation-only (not for headline)|

Also tracked but not headline:
- pair68 (worst case in LLVE-STCD-12): 13.40 dB. NOTE: D1b (CH2 verification) shows this is content/scene-time tied, not output-position tied. v10 MVP does NOT promise to fix pair68 directly; M4 mechanism options are deferred to v10.5.
- MVI_1030 / MVI_1030_2 (over-processing failure cases): currently negative-delta vs LQ; v10 must not regress.

Reference: LLVE-STCD reported (their model on their full LLVE-STCD-12 split): **28.93 dB**. Same-pipeline gap = 8.06 dB.

---

## 2. Failure evidence summary

| failure mode | source | v10 component addressing |
|---|---|---|
| Tile-context inconsistency (per-tile V-JEPA recompute → seam) | v9.4c BI3 | RESOLVED: tile=512 protocol + larger context per tile |
| Cross-attn permutation invariance on V-JEPA K/V (no spatial-temporal alignment) | v9.3 / BC1 | M1 grid-aware V-JEPA conditioning (spatial only at fixed grid) |
| Global photometric / color cast (oracle Δ +3.62 outdoor, +0.33 indoor) | D3 | M5 mean-color L1 loss |
| Cross-split degradation (-4.87 dB intra-pipeline) | BX3 + outdoor BY1 | M2 multi-scale pixel-branch context + M5 + M7 |
| Border modeling | originally hypothesized BV4-B1 | NOT a problem (D2 reversed) — DROPPED |
| Per-frame degradation on pair68 (-0.187 dB/frame) | BY3 + D1 | DEFERRED to v10.5 — D1b shows content-tied (CH2), not position-tied |

---

## 3. v10 MVP scope (4 mandatory components)

### M1 — grid-aware V-JEPA conditioning (spatial pos-emb at fixed grid)

**Mechanism**: Add learned spatial positional embeddings P_y, P_x to V-JEPA tokens *before* the existing cross-attention K/V projection. Use a learnable scalar `gamma_pos` initialized to 1 with P_*=0 so that v10 = v9.0b_mix exactly at iter 0 (gamma * 0 == 0) AND P gradients are nonzero at step 0. (CL1 fix; gamma=0 init was dead-grad.)

```python
# v10 V-JEPA token augmentation, applied per cross-attn block
# V-JEPA wrapper produces fixed token grid (T'=8, H'=16, W'=16); see CH1 note.

class VJepaPosEmb(nn.Module):
    def __init__(self, T_v=8, H_v=16, W_v=16, dim=1024):
        super().__init__()
        self.P_t = nn.Embedding(T_v, dim)  # temporal pos-emb (kept; not load-bearing per D1b)
        self.P_y = nn.Embedding(H_v, dim)
        self.P_x = nn.Embedding(W_v, dim)
        # init pos embeddings to zero so iter-0 contribution is zero regardless of gamma
        nn.init.zeros_(self.P_t.weight)
        nn.init.zeros_(self.P_y.weight)
        nn.init.zeros_(self.P_x.weight)
        self.gamma_pos = nn.Parameter(torch.ones(1))  # CL1 fix: identity via P_*=0, nonzero P grads (CH3)

    def forward(self, vjepa_tokens, T_v, H_v, W_v):
        # vjepa_tokens: (B, T_v*H_v*W_v, D)
        B, N, D = vjepa_tokens.shape
        device = vjepa_tokens.device
        t_idx = torch.arange(T_v, device=device).view(T_v, 1, 1).expand(T_v, H_v, W_v).reshape(-1)
        y_idx = torch.arange(H_v, device=device).view(1, H_v, 1).expand(T_v, H_v, W_v).reshape(-1)
        x_idx = torch.arange(W_v, device=device).view(1, 1, W_v).expand(T_v, H_v, W_v).reshape(-1)
        pos = self.P_t(t_idx) + self.P_y(y_idx) + self.P_x(x_idx)  # (N, D)
        return vjepa_tokens + self.gamma_pos * pos.unsqueeze(0)
```

**Param add**: 3 × max_grid × dim per cross-attn block. For dim=1024 and grid (8,16,16): (8+16+16) × 1024 = 40K + 1 (gamma) per block × 7 blocks ≈ **280K total params (negligible vs 361M trainable)**.

**Identity-at-init guarantee** (CH3): with `gamma_pos=1` (P_*=0 gives identity at iter 0; CL1 fixed dead-grad of `gamma_pos=0`) AND `P_*` weights zero-initialized, the cross-attn input is byte-identical to v9.0b_mix at iter 0. Required zero-iter test (see §7).

**M4a status**: P_t embedding is INCLUDED but is not the primary mechanism. D1b (CH2 follow-up) shows pair68 degradation is content/scene-time tied not output-index tied. P_t is kept as a small inductive bias but its evidence-justified contribution is small. If P_t hurts at training, ablate it out (see §8).

**Target gain** (M1 spatial only): +0.4-0.9 dB on indoor (where M5 won't help much).

### M2 — train/eval scale alignment via pixel-branch crop curriculum

**IMPORTANT (CH1)**: V-JEPA wrapper hardcodes image_size=256. ALL crops (256/384/512) get resized to 256 before V-JEPA encode. So M2 changes ONLY the pixel-branch context window and the V-JEPA's view of the wider scene (V-JEPA sees a downsampled 256×256 of a wider 512×512 area). V-JEPA grid stays (8,16,16) throughout.

**Mechanism**: multi-scale crop curriculum during pixel-branch training.

```text
Schedule (5000 iter total budget):
  iter 0-1500:    crop=256  (warm-up; V-JEPA sees actual 256 area)
  iter 1500-3500: crop=384  (transition; V-JEPA sees larger context downsampled)
  iter 3500-5000: crop=512  (final; matches eval tile size)

Or simpler 2-stage:
  iter 0-3500:    crop=384
  iter 3500-5000: crop=512  (finetune)
```

**Stability mitigation** (R4): warm restart at each transition (lr × 0.5 for first 100 iter post-transition).

**Fallback**: fixed crop=384 for full 5000 iter if multi-scale destabilizes.

**Target gain**: +0.5-1.2 dB on tile=512 evaluation (less than v1 estimate because V-JEPA grid is unchanged).

### M5 — color/brightness alignment loss

**Mechanism**: per-frame per-channel mean-color L1 penalty.

```python
def color_mean_loss(pred, gt):
    # pred, gt: (B, 3, T, H, W) in [0, 1]
    return (pred.mean(dim=(-2, -1)) - gt.mean(dim=(-2, -1))).abs().mean()
```

**Loss weight**: 0.1 relative to L_pixel (CE3, CH7-Q2). Do NOT raise to 0.5.

**Target gain**: outdoor +1.5-3.5 dB (D3 oracle ceiling +3.62); indoor +0.1-0.3 dB.

### M7 — fixed benchmark protocol

**Mechanism**: NO MODEL CHANGE. Hard-coded eval scripts use the 4 fixed splits in §1, no random 80/20 shuffles.

**File**: `tools/v10_audit.py` to be written, replicating tile=512/overlap=128 RGB-PSNR audit on all 4 anchored splits.

---

## 4. Architecture diagram / dataflow

```
 LQ video (B, 3, T=16, H, W)   [H, W = train crop size: 256/384/512]
        │
        ├──→ V-JEPA backbone (frozen, 304M)
        │       Internal: _resize_bcthw → 256x256
        │       Output: feats[L1..L4], token_grid (T'=8, H'=16, W'=16)  ← FIXED
        │
        ▼
   ── NEW v10 (M1) ─────────────────────────────────────
     vjepa_tokens_pos = vjepa_tokens + gamma_pos * (P_t[t] + P_y[y] + P_x[x])
     Initially gamma_pos=1, P_*=0 → v10 ≡ v9.0b_mix at iter 0 (gamma*0=0; CL1 fix)
   ──────────────────────────────────────────────────────
        │
        ▼
   pixel_branch (361M trainable, UNCHANGED architecture)
        │
        ├──→ enc_blocks (248M)
        ├──→ enc_xattn  ←── kv = Linear(vjepa_tokens_pos)  (per scale 1-4)
        ├──→ ... [all other components per v9.0b_mix]
        └──→ pred_pixels
        │
        ▼
   pred (B, 3, T, H, W)

ALL other v9.0b_mix components UNCHANGED.
```

---

## 5. Losses + training schedule

### Losses (CH4: keep v9.0b_mix loss config)

```python
# v9.0b_mix existing losses (lpips_weight=0)
L_pixel = charbonnier(pred, gt)
# (other v9.0b losses kept as-is per train_v9_0b.yaml; no LPIPS active)

# NEW v10 loss
L_color = color_mean_loss(pred, gt)

L_total = L_pixel + ... (other v9.0b weights unchanged) + 0.1 * L_color
# Color weight 0.1 (CE3, CH7-Q2). Do NOT start at 0.5.
```

### Optimizer + schedule

```text
init: v9.0b_mix_best.pt (load all weights; gamma_pos=1 + P_*=0 ensures equivalence at iter 0 AND P grad nonzero; CL1 fix)
optimizer: AdamW (existing v9.0b_mix config)
lr: 2e-5 base, scaled with cosine to 1e-6 (CU3 reconcile: matches configs/train_v10.yaml)
warmup: 0 (continues from v9.0b_mix_best.pt anchor; warmup unnecessary)

iter budget: 5000 (CE1; longer iter only after positive slope shown)
batch: 1 (memory-limited at crop=512)
gradient accumulation: 4 (effective batch 4; scheduler converts iter→optim-step per CO6/CP6)
```

### Crop curriculum (M2)

```text
Try first:
  iter 0-1500:    crop=256
  iter 1500-3500: crop=384  + lr×0.5 for 100 iter post-transition
  iter 3500-5000: crop=512  + lr×0.5 for 100 iter post-transition

Fallback (if multi-scale unstable):
  fixed crop=384 for 4000 iter, then crop=512 for 1000 iter
```

### Validation cadence (CH6 sentinel panel)

```text
Every 250 iter, eval at tile=512 on:
  hard indoor (LLVE-STCD-12 sub):  pair68, pair31
  outdoor:                          MVI_1030, MVI_1019
  legacy guard:                     pair6    (was pair3_2; pair3_2 dropped:
                                              only 14 basename-matched frames < T=16
                                              after CN-A alignment fix)
                                    -----
                                    5 sentinel seqs

Full 4-split audit every 1000 iter and at end.
```

---

## 6. Memory / param / throughput (BU4 measured)

### Param count

| component | params (v9.0b_mix) | params (v10) | Δ      |
|---|---:|---:|---:|
| backbone (V-JEPA frozen) | 303,893,504 | 303,893,504 | 0 |
| pixel_branch (trainable) | 361,117,589 | ~361,400,000 | +280K (M1 pos-emb + gamma) |
| **total** | **665,011,093** | **~665,300,000** | **+280K (~0.04%)** |

v10 essentially adds zero params.

### Memory profile (bf16, batch=1, T=16, single A100-80GB)

(unchanged from v9.0b since param count is essentially identical)

Inference:
| crop | vram peak | fwd time |
|---:|---:|---:|
|  256 |    6.0 GB |  1170 ms |
|  384 |    9.5 GB |  1339 ms |
|  512 |   14.3 GB |  2125 ms |

Training (fwd+bwd, no optimizer state):
| crop | vram peak | fwd+bwd time | iter/sec |
|---:|---:|---:|---:|
|  256 |   17.2 GB |       3580 ms | 0.28 |
|  384 |   35.3 GB |       6727 ms | 0.15 |
|  512 |   60.5 GB |       9678 ms | 0.10 |

### Wall-clock estimates

| schedule | h pure fwd+bwd | h real (×1.7) |
|---|---:|---:|
| Multi-scale 5000 iter (256/384/512 split as §5)  | ~6 h | ~10 h |
| Fallback fixed crop=384 + 1000 finetune crop=512 | ~10 h | ~17 h |

---

## 7. Acceptance criteria

### Technical target (CE5)

```text
PRIMARY: v10 must beat v9.0b_mix on
  - LLVE-STCD-12: ≥ +1.0 dB (i.e. ≥ 21.87)
  - outdoor-25:   ≥ +1.0 dB (i.e. ≥ 22.78)        ← CH5 corrected
  AND not regress on
  - original-14:  ≥ -0.2 dB (i.e. ≥ 25.54)         ← CH5 standardized
```

### Strong target (CH5 corrected)

```text
+2.0 dB on at least one hard split AND +1.0 dB on the other.
  e.g. LLVE-STCD-12 ≥ 22.87 AND outdoor-25 ≥ 22.78
  or   LLVE-STCD-12 ≥ 21.87 AND outdoor-25 ≥ 23.78
```

### Identity-at-init test (CH3 mandatory)

```text
Before any v10 training step:
  Load v9.0b_mix weights into v10 model (with gamma_pos=1, P_*=0; CL1 fix).
  Forward identical input through v9.0b_mix and v10.
  Verify:
    mean abs(pred_v10 - pred_v9) < 1e-5
    PSNR difference < 0.01 dB
  If fail: implementation bug; do not train.
```

### Failure-mode targets

```text
MVI_1030 + MVI_1030_2: ≥ -1.0 dB above LQ (i.e. v10 pred > 17.0)
                       — M5 should fix the over-processing
                       
pair68: NOT a hard target for v10 (D1b: content-tied, M4 deferred)
        Track but accept degradation if other splits improve.
```

### Stop conditions during training

```text
After iter 1500 (256→384 transition): val ≥ +0.3 dB over v9.0b_mix on sentinel panel
After iter 3500 (384→512 transition): val ≥ +0.7 dB
At iter 5000:                          val ≥ +1.0 dB

If any milestone fails by ≥0.3 dB: pause training, rerun ablation.
```

---

## 8. Ablation plan

After v10 main run lands, ablate:

| ablation | what's removed | hypothesis if drop |
|---|---|---|
| M1 off (gamma_pos frozen at 0, P_*=0 frozen) | pos-emb contribution | M1 contribution isolated |
| M5 off | L_color removed | M5 contribution to indoor/outdoor split |
| M2 fixed crop (256-only train) | no curriculum | scale-alignment gain |
| P_t off (only P_y/P_x) | temporal pos-emb removed | confirms D1b — P_t marginal |
| TTA on/off | 4-flip averaging | reportable boost |

Each ablation = 1 short retrain (~3000 iter, ~3-6 h). Total ablation budget ~5 retrain × ~5 h = 25 h compute.

---

## 9. Open risks

```text
R1. V-JEPA wrapper resize behavior under crop=384/512:
  All crops are resized internally to 256. M2's gain comes from pixel-branch
  context, NOT from V-JEPA seeing higher-res tokens. CH1 confirmed; spec now
  reflects this. Variable V-JEPA grid is v10.5 scope.

R2. M1 may not learn enough in 5000 iter (CL1-fixed init):
  With identity-at-init via gamma_pos=1 + P_*=0, P embeddings learn first and gamma_pos adjusts as P norms grow.
  P_* embeddings learn. Mitigation: lr scaling or unfreeze P_* embeddings
  Monitor: ||P_y||/||P_x|| > 0 by step 100 (smoke confirmed grad nonzero from step 0); gamma_pos value should drift > 0.95 (it starts at 1.0) by
  iter 1000 to indicate M1 is being used.

R3. Color loss vs pixel loss conflict:
  L_color may pull mean toward GT mean at expense of structural fidelity.
  Mitigation: low initial weight 0.1; raise only if color converges slowly.

R4. Multi-scale curriculum instability:
  Crop transitions may cause loss spikes. Mitigation: warm restart at each
  transition (lr × 0.5 for first 100 iter post-transition).

R5. M1 P_t embedding is included but D1b shows it may not help meaningfully:
  If P_t weights stay near zero in training, that's fine. If P_t actively
  harms (gamma_pos drifts toward 0 or grad of P_t stays trivial), ablate it.

R6. BVI-RLV download still in progress (~7 GB / 142 GB):
  v10 first run will be SDSD-only. BVI integration is v10.5 scope if first
  v10 result looks promising.
```

---

## File / config artifacts to be written

```text
configs/train_v10.yaml              # extends train_v9_0b.yaml: + color_mean_weight, + crop curriculum config
models/v10/
  __init__.py
  vjepa_pos_emb.py                  # M1 module (~30 lines)
  pixel_branch_v10.py               # tiny diff vs pixel_branch_v9 — wires VJepaPosEmb
  vjepa_llve_v10.py                 # main wrapper
losses/color_mean.py                # M5 (~10 lines)
tools/v10_audit.py                  # M7 fixed-split eval script
tools/v10_zero_iter_test.py         # CH3 identity-at-init verification
scripts/train_v10.sh                # launch script
scripts/v10_ablation.sh             # 5-ablation sweep launcher
```

Implementation effort estimate:
- Code: ~2 days (M1 module is small, mostly training pipeline + curriculum)
- Zero-iter test: 0.5 hour (mandatory)
- First train run: ~10-17 h depending on schedule
- Eval + initial review: 0.5 day
- Total: ~3-4 days to first v10 results

---

## Cross-references

- Locked anchored splits: `VJEPA_LLVE_CODE_REVIEW.md` §CB0
- D1-D3 evidence: `VJEPA_LLVE_CODE_REVIEW.md` §CD1-CD3
- D1b verification (M4 demoted): §CH2 + author CI follow-up
- v10 component lock: §CE + §CH approval
- v9.0b_mix architecture and measurements: §BZ
- BVI-RLV viability: §BW

This v2 supersedes v1 of V10_DESIGN_SPEC.md.
