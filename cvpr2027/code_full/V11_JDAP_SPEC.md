# V11 — JEPA-Delta Action Prior (JDAP) Design Spec
> v0 (2026-05-03) — proposed after v10 abort. Written under CZ proposal.
> Source threads: VJEPA_LLVE_CODE_REVIEW.md §CY (self-critique), §CZ (proposal).

---

## 1. Hypothesis (single, falsifiable)

**V-JEPA is useful for LLVE only if we learn a target-conditioned latent
restoration action `delta = z_H - z_L`, not if we merely attend to low-light tokens.**

If true:
  - Adapter A(z_L) -> z_hat_H must beat identity baseline on held-out clips.
  - An RGB decoder conditioned on (delta_hat, z_L) must beat decoder(LQ, z_L cross-attn).
  - Replacing predicted delta with random / shuffled delta must KILL the gain.

If the "shuffled delta" ablation still works, JDAP is a sham — the decoder
doesn't actually use V-JEPA-specific information.

## 2. Architecture

### 2.1 Components

```
LQ video                                        HQ video (training only)
   |                                                |
   v                                                v
V-JEPA encoder (frozen, 304M)                V-JEPA encoder (frozen, 304M)
   |                                                |
   v                                                v
z_L_layers (4 layers, B,N,1024)            z_H_layers (4 layers, B,N,1024)
   |                                                |
   |  +--- L_delta = ||z_hat_H - z_H|| training only|
   v  v                                              |
JEPA Delta Adapter (compact, 14.7M)                  |
  delta_hat_layers = A(z_L_layers)                   |
  z_hat_H_layers = z_L_layers + delta_hat_layers     |
   |                                                 |
   v                                                 |
Delta-Modulated Decoder (compact, target ≤ 200M)     |
  pred = D(LQ, z_L, delta_hat, z_hat_H)              |
   |                                                 |
   v                                                 v
pred_RGB                                  HQ supervision: L_pix, etc.
```

### 2.2 JEPA Delta Adapter

Compact bottleneck per V-JEPA layer:

```
z_L (B, N=2048, D=1024)
  -> Linear(D, bot_dim=256)          # down
  -> 4 × Transformer block (dim=256, heads=4, ffn=4×)
  -> LN + Linear(bot_dim, D)         # up (zero-init -> identity at iter 0)
  -> delta_hat
z_hat_H = z_L + delta_hat
```

4 V-JEPA layers × ~3.7M = **14.7M params**.

### 2.3 Delta-Modulated Decoder (proposed)

NOT cross-attn-only. Use modulation interfaces:

```
At each U-Net scale s, given (z_L_s, delta_hat_s) projected to spatial token map:
  - AdaLN gates from delta_hat: per-token scale/shift
  - Optional: dynamic 1x1 conv whose weights come from a small MLP(delta_hat)
  - Cross-attn KEPT as one secondary path (baseline-comparable)
```

Decoder param budget: ≤ 200M (v9 pixel_branch was 361M). Aim 100-150M.

### 2.4 Identity-at-init guarantee

Adapter `out_proj` zero-initialized → delta_hat = 0 at iter 0 →
z_hat_H = z_L → AdaLN scale = 0 → decoder sees only z_L (cross-attn baseline).
Equivalent to v9.0b at iter 0; no harm gate failure.

## 3. Probes (CZ4)

### Probe A — Delta statistics (DONE)

Per V-JEPA layer cos(z_L, z_H) and ||z_H - z_L|| stats on real LQ/HQ pairs.
Already executed (`tools/v11_delta_stats.py`):
- All 4 layers PASS: rel_delta 0.29–0.60, cos 0.81–0.96
- Layer 23 strongest signal (cos 0.81, rel_delta 0.60)
- LLVE-STCD-12 (hardest) shows largest delta — task severity correlates
  with delta magnitude (good news).

### Probe B — Adapter beats identity (RUNNING)

Train compact `JEPADeltaAdapter` (14.7M) on 134 SDSD train clips, 2000 iter.
Evaluate on LLVE-STCD-12 (12 held-out hard clips).

Pass criterion: per-layer adapter-cos > identity-cos AND adapter-L2 < identity-L2.

### Probe C — Oracle decoder upper bound (TODO)

Train one decoder under 4 input conditions:
1. `decoder(LQ, z_L cross-attn only)`  — baseline (~v9.0b style)
2. `decoder(LQ, z_L, oracle z_H)`       — upper bound with target known
3. `decoder(LQ, z_L, oracle delta = z_H - z_L modulation)`  — upper bound for delta interface
4. `decoder(LQ, z_L, predicted delta from A)`  — actual v11 path

Pass criterion (CZ4):
- Oracle delta condition (3) must improve RGB tile=512 by ≥ +0.7 dB
- Predicted delta (4) must recover ≥ 30-40% of oracle gain

If (3) ≤ +0.7 dB → decoder interface is wrong (not delta-modulation-friendly).
If (4) recovers <30% → adapter is too weak; expand or change.

## 4. Killer Ablations (CZ5)

Required in any v11 paper:

| ablation | description | expected v11 result |
|---|---|---|
| no_prior | decoder(LQ) only | weakest |
| passive_prior | decoder(LQ, z_L cross-attn) | ≈ v9.0b baseline |
| **shuffled_delta** | delta from a different sequence's z_L,z_H | **MUST drop to passive baseline** |
| identity_delta | delta=0 | should equal passive |
| oracle_delta | delta=z_H-z_L | upper bound |
| predicted_delta | delta=A(z_L) | v11 main result |
| backbone_swap (optional) | DINOv2 / VideoMAE in place of V-JEPA | should weaken substantially |

The shuffled-delta test isolates whether v11 actually uses content-correct delta.

## 5. Training Protocol (TBD; for v11.0 main run)

Locked from v10 hygiene:
- tile=512 / overlap=128 RGB-PSNR uint8 evaluation
- 4 fixed splits (LLVE-STCD-12, outdoor-25, original-14, BZ-15)
- per-seq sentinel guard (≥ v9_baseline - 0.5 dB)
- save_ckpt saves FULL state (adapter + decoder)
- LD_LIBRARY_PATH cuDNN fix in scripts/train_v11.sh

To be set after Probe B/C results inform decoder size:
- decoder size, depth, modulation type (AdaLN vs FiLM vs dynamic conv)
- crop curriculum (likely 256 → 384 → 512 as in v10)
- loss recipe (delta supervision + RGB Charbonnier; NO new aux losses without ablation)

## 6. Rejected (kept as design record)

- ❌ v10's positional embedding M1 (still useful as a side fix; not a contribution)
- ❌ Latent V-JEPA diffusion (v6/v6D explored variants; needs new oracle proof)
- ❌ Reference-based retrieval (different task)
- ❌ V-JEPA LoRA finetune (weakens claim; do only if Probe C succeeds)
- ❌ Deeper pixel branch (already shown low return; user explicit guidance)

## 7. Decision Gates

```
gate G1: Probe A pass               -> proceed to Probe B  [DONE 2026-05-03]
gate G2: Probe B per-layer pass     -> proceed to Probe C
gate G3: Probe C oracle gain ≥+0.7  -> design v11.0 main model
gate G4: Probe C predicted ≥30%     -> launch v11.0 train
gate G5: shuffled-delta ablation     -> publish v11
```

If any gate fails, stop and re-design rather than patch.

## 8. Risks

```
R1. V-JEPA encoder output is flat (cos≈1 everywhere)
    -> Probe A invalidates JDAP. Mitigation: switch to V-JEPA finetune track.
    Status: Probe A passed; risk closed.

R2. Adapter overfits 134 train clips
    -> early stop on held-out per-layer cos; small adapter; weight decay.

R3. Oracle delta does not help decoder
    -> means latent-modulation interface is wrong; try cross-attn replacement
       with attention over (z_L || delta_hat || z_hat_H) tokens.

R4. Shuffled delta still works
    -> JDAP claim dies. Pivot to either V-JEPA finetune or different prior task.
```

## 9. Source review thread

Author / Reviewer letters in `VJEPA_LLVE_CODE_REVIEW.md`:
- §CY: author self-critique, six loose ideas
- §CZ: reviewer locks single hypothesis (this spec)
