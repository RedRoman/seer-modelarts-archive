---

# Author Response — CM. CL Acknowledged; All 5 Blockers Will Be Fixed Before Re-Launch (2026-05-03)

## CM0. CL accepted in full

Five blockers + one medium-severity issue stand. v10_first run is halted; the
second launch attempt also failed (cuDNN backward error on the SECOND iter,
log-tailed below) which independently confirms the implementation is not yet
production-ready.

```text
v10_first status:
  attempt 1: aborted at sanity gate (stale 26.843 target — own bug, not v10 bug)
  attempt 2: aborted at backward step (cuDNN engine miss on first train batch)
```

The cuDNN error is likely co-caused by CL3 (workers see stale dataset) +
mixed eval/train state. We will not debug it before fixing CL1-CL5.

## CM1. CL1 — M1 dead-gradient. Fix: gamma_pos = 1.0

Adopting the preferred fix from CL1:

```python
# vjepa_pos_emb.py
self.P_t.weight = 0
self.P_y.weight = 0
self.P_x.weight = 0
self.gamma_pos  = 1.0      # was 0
```

Identity-at-init still holds exactly: `tokens + 1.0 * 0 == tokens`.
But now `dL/dP ∝ gamma_pos = 1` so M1 actually learns.

Updated zero-iter test expectation:

```text
gamma_pos prints 1.0 (not 0)
P_t/P_y/P_x sum prints 0
forward equivalence diff: still must be 0.0
```

## CM2. CL2 — checkpoint save/load. Fix: dedicated save_ckpt_v10 + audit hardening

```python
# train_v10.py
def save_ckpt_v10(path, model, optim, sched, ema, step, best, val_history):
    state = {
        "step": step, "best_metric": best, "val_history": list(val_history),
        "model": model.state_dict(),     # FULL — includes vjepa_pos_emb.*
        "optim": optim.state_dict(),
        "scheduler": sched.state_dict(),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)

# tools/v10_audit.py (--model v10 branch)
- if "model" in state: load it (preferred path for v10 trained ckpts)
- elif "state_dict" in state: load it
- elif "pixel_branch" in state:
    print warning that this is a v9-style ckpt with no vjepa_pos_emb
    only load pixel_branch into model.pixel_branch
    require an explicit --allow-v9-ckpt flag, otherwise abort
- else: abort
- always: print first 10 missing/unexpected keys; abort if model_keys_loaded < 80%
```

## CM3. CL3 — DataLoader transition. Fix: persistent_workers=False + rebuild + assert

```python
# train_v10.py
DataLoader(... persistent_workers=False ...)

def _make_loader(train_set, ...):
    return DataLoader(...)

# at each transition:
    n_upd = _set_crop_size(train_set, new_crop)
    train_loader = _make_loader(train_set, ...)   # full rebuild
    data_iter = iter(train_loader)
    # one-batch shape assertion
    probe_batch = next(data_iter)
    H, W = probe_batch["lq"].shape[-2:]
    assert (H, W) == (new_crop, new_crop), \
        f"crop transition failed: got {(H,W)}, expected {(new_crop, new_crop)}"
    logger.info(f"[curriculum] verified crop {(H,W)} after rebuild")
```

We pay one rebuild per transition (only twice in 5000 iter), trivial cost.

## CM4. CL4 — train-time validation. Fix: keep Y-512 as warning metric, add sentinel sub-eval

We will NOT promote v10 best.pt on Y-512 alone. Plan:

```text
during training:
  every val_every (250):  cheap RGB-256 quick_eval (fast continuity)
  every audit_every (1000): tile=512 sentinel-5 eval (the locked CJ3 panel)
                            using the same v10_audit infrastructure
  best.pt promotion gate:   sentinel-5 mean > current_best AND no individual
                            seq drops > 0.5 dB vs prior

after training:
  full 4-split v10_audit at end + best.pt
  this is the ONLY headline metric
```

Y-512 audit_y512() stays as logged warning (catches collapse) but does not
gate best.pt.

## CM5. CL5 — sanity_y512_target=28.621. Fix: drop the gate

Removing the Y-512 hard gate from train_v10.py. Identity-at-init is enforced by
the standalone `tools/v10_zero_iter_test.py` (already PASSED with 0.0 diff).
Y-512 stays as a logged baseline only:

```text
log a one-shot Y-512 audit at iter 0, NO assertion
log the v9.0b sentinel baseline (already in /cache/.../v10_audit/v9_0b_mix_sentinel_baseline/)
```

This removes the magic number 28.621 from config.

## CM6. CL6 — LR decay overridden by LambdaLR. Fix: in-scheduler factor

```python
# train_v10.py — replace manual param_group lr edit with a transition factor
# applied inside the lambda:
def make_lambda(...):
    def lr_lambda(s):
        base = cosine_or_warmup(s)  # current logic
        # transition decay: ×0.5 for transition_lr_iters after each curriculum step
        for t_iter in transition_iters:
            if t_iter <= s < t_iter + transition_lr_iters:
                return base * transition_lr_decay
        return base
    return lr_lambda
```

Plus log lr every step for the first 5 steps after each transition.

## CM7. CL8 — pre-launch order

Adopting CL8 strict order:

```text
1. M1 init fix (CM1)                                    [todo]
2. ckpt save/load (CM2)                                 [todo]
3. crop curriculum + shape assertion (CM3)              [todo]
4. drop sanity_y512 (CM5)                               [todo]
5. fix transition LR (CM6)                              [todo]
6. zero-iter equivalence with new init                  [todo]
7. v10_audit v9 sentinel baseline                       [DONE — see CM_appendix_A]
8. 2-iter smoke train: gamma_pos grad >0, ckpt reload   [todo]
9. ONLY THEN launch v10_first
```

## CM_appendix_A. v9.0b sentinel baseline (CJ3, already computed)

Recorded at `/cache/zihaohe/checkpoint/v10_audit/v9_0b_mix_sentinel_baseline/`:

```text
protocol: tile=512, overlap=128, RGB-PSNR uint8 full-frame tiled
ckpt: /cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt

| kind    | seq      | LQ    | v9.0b  | Δ vs LQ |
|---------|----------|-------|--------|---------|
| indoor  | pair68   |  8.54 | 29.566 | +21.029 |
| indoor  | pair31   |  7.84 | 17.665 |  +9.822 |
| outdoor | MVI_1030 | 18.95 | 14.917 |  -4.032 |
| outdoor | MVI_1019 |  5.95 | 13.608 |  +7.655 |
| indoor  | pair3_2  | 10.87 | 26.994 | +16.122 |

sentinel-5 mean: 20.550 dB
```

## CM8. ETA

```text
CM1-CM6 fixes:           ~2 hr
zero-iter re-verify:     ~3 min
2-iter smoke:            ~10 min
post results to CN:      same evening
```

If CN looks clean, await your "approved, launch v10_first" before re-running.

(CM posted; v10_first kept halted.)
