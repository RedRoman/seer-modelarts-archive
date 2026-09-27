# V11 Probe C — Oracle-Delta Decoder Upper Bound

> v0 (2026-05-04). Approved in `VJEPA_LLVE_CODE_REVIEW.md` §DH4.
> Probe A and Probe B' are PASS. This probe answers the next question:
> **Does a pixel decoder actually use the JEPA delta to improve RGB?**

## 1. Hypothesis

> Conditioning a small RGB decoder on the V-JEPA target latent action
> `delta = z_H - z_L` lets it produce strictly better RGB than:
> - no V-JEPA prior at all
> - passive `z_L` cross-attention
> - random ("shuffled") delta from a different sequence

If oracle delta does not help RGB, the JDAP architecture loses its premise even
though Probe B' showed delta is learnable in latent space.

## 2. Conditions (DH5)

Same decoder, same params, same data split, same RGB loss / schedule.

| # | name | input | role |
|---:|---|---|---|
| 0 | no_prior | D(LQ) | weakest baseline |
| 1 | passive_zL | D(LQ, z_L) | v9/v10-like baseline |
| 2 | identity_delta | D(LQ, z_L, delta=0) | control: modulation path with zero |
| 3 | shuffled_delta | D(LQ, z_L, delta from wrong seq) | KILLER ablation |
| 4 | oracle_zH | D(LQ, z_H) | target-latent upper bound |
| 5 | oracle_delta | D(LQ, z_L, delta=z_H - z_L) | JDAP upper bound |
| 6 | predicted_delta | D(LQ, z_L, A(z_L)) | v11 main result (A frozen) |

Minimum subset if budget tight: 1, 2, 3, 5, 6. `oracle_zH` recommended.

## 3. Decoder constraints (DH6)

Hard:
- ≤ v9 pixel_branch param count (361M); target **≤ 80M trainable**.
- Same architecture across all conditions.
- Same data split as Probe B' (sequence-disjoint LLVE-STCD-12 held-out).
- Same RGB loss + schedule across conditions.
- For `predicted_delta`: load `probe_b_prime/delta_adapter.pt`, **frozen**.
- Delta interface: AdaLN / FiLM / gates (NOT plain cross-attn).

Architecture sketch:
```
LQ video (B,3,T=16,H=256,W=256)
  → 5-scale 3D U-Net encoder (dims: 32,64,128,256,512; blocks: 2,4,4,6,8)
  → at each scale s:
        z_L scale-specific projection → spatial map (cross-attn KEYS+VALUES)
        delta scale-specific projection → AdaLN scale,shift modulator
  → 5-scale decoder
  → conv_out → pred RGB
```

Rough param count target:
```
backbone U-Net (3D conv stack): ~25-40M
per-scale cross-attn (z_L → KV): ~5-10M
per-scale AdaLN heads (delta): ~3-5M
total trainable: ~35-55M
```

(Frozen V-JEPA backbone 304M is shared and not counted as trainable.)

## 4. Training protocol

- Load V-JEPA frozen for all conditions (same encoder pass).
- Per-iter on each condition:
  - Forward LQ → get z_L (and z_H for oracle conditions).
  - Compute delta per condition (random other-sequence z_H - z_L for shuffled).
  - Decoder forward → pred RGB.
  - Charbonnier loss on pred vs HQ (no aux losses for this probe).
- Optimizer: AdamW lr=2e-4 → 2e-5 cosine, 1500 iter / condition.
- Batch=1, accum=2, crop=256 (LQ resized to 256 by V-JEPA wrapper anyway).
- Sequence-disjoint LLVE-STCD-12 held-out for eval (same as B').

## 5. Eval

After each condition completes:
- Run inference at tile=512 / overlap=128 on LLVE-STCD-12 (12 clips).
- Compute RGB-PSNR vs HQ.
- Persist condition_<name>.pt + per_seq.csv.

Output:
```
/cache/zihaohe/checkpoint/v11_probes/probe_c/
    no_prior/{ckpt.pt, per_seq.csv, summary.md}
    passive_zL/...
    identity_delta/...
    shuffled_delta/...
    oracle_zH/...
    oracle_delta/...
    predicted_delta/...
    PROBE_C_SUMMARY.md   # cross-condition table + verdict
```

## 6. Pass / fail gates (DH7)

```
Primary: oracle_delta ≥ passive_zL + 0.7 dB on LLVE-STCD-12 mean PSNR
Secondary: predicted_delta recovers ≥ 30-40% of (oracle_delta - passive_zL)
Killer: shuffled_delta within ~0.1 dB of passive_zL/identity_delta
Bonus: oracle_zH ≥ oracle_delta (target-latent upper bound)
```

If primary fails: latent-modulation interface is wrong — not the prior itself.
If killer fails (shuffled delta still helps): JDAP claim DIES.

## 7. Compute budget

```
6 conditions × 1500 iter × ~3 s/iter ≈ 7-8 hr
+ 6 condition evals × 12 seq × ~35 s/seq ≈ 40 min
+ 1 baseline-baseline (passive_zL) eval already done as v9.0b sentinel ref
total: ~9 hr GPU
```

If budget tight, run minimum subset (1,2,3,5,6) ≈ 6 hr.

## 8. What this proves / does not prove

PROVES:
- Whether a pixel decoder can use latent JEPA delta at all.
- Whether the upper-bound (oracle delta) is even useful.
- Whether the learned adapter delta closes meaningful fraction of upper bound.
- Killer test: that V-JEPA-specific content matters (not just any conditioning).

DOES NOT PROVE:
- Final RGB-PSNR vs SOTA (v9, v10, V-DUN, LLVE-STCD).
- That a deeper / larger v11 trained end-to-end will beat v10.
- Generalization to BVI-RLV / outdoor.

Those require full v11 training, which only comes after Probe C passes.
