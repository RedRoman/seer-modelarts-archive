"""train_v10.py — V-JEPA-LLVE v10 trainer.

Diffs vs train_v9_0b:
  - Model: VJepaLLVE_v10 (v9.0b pixel_branch + VJepaPosEmb identity-at-init)
  - Loads v9.0b_mix weights via model.load_v9_weights() (CK0)
  - Crop curriculum (M2): 256→384→512 with lr×0.5 for 100 iter post-transition
  - color_mean_loss (M5) added with weight 0.1
  - Y-512 audit at iter 0 is INFORMATIONAL only (CL5/CN6 fix).
    Identity-at-init is enforced by tools/v10_zero_iter_test.py exclusively.

Spec: V10_DESIGN_SPEC.md §3 / §5 / §7 / CJ4.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data import build_dataset, build_sampler
from data.sdsd_dataset import SDSDDataset
from losses.v9 import LLVELoss_v9
from losses.color_mean import color_mean_loss
from models.v10 import VJepaLLVE_v10
from utils import (
    AverageMeter, ParameterEMA, dump_config, ensure_dir, load_config,
    set_seed, setup_logger,
)

# Reuse v9.0b's helpers verbatim — they don't depend on the model class.
from train_v9_0b import (
    amp_dtype, amp_enabled,
    list_audit_seqs, audit_y512, quick_eval,
    build_optimizer, build_scheduler,
)
from tools.v10_audit import (
    INDOOR_ROOT, OUTDOOR_ROOT, SENTINEL_5,
    index_seqs as _v10_index_seqs,
    eval_one_seq as _v10_eval_one_seq,
)
import json as _json


# ---- sentinel-5 eval (CL4 fix) --------------------------------------
def sentinel_eval(model, indoor_idx, outdoor_idx, device):
    """Run the 5-seq sentinel panel at tile=512 inside the trainer.
    Returns (mean_psnr, per_seq_dict). per_seq_dict maps seq_name -> psnr.
    Used as the promotion gate for best.pt (CL4)."""
    model.eval()
    out = {}
    for kind, s in SENTINEL_5:
        idx = indoor_idx if kind == "indoor" else outdoor_idx
        if s not in idx:
            continue
        lq_p, gt_p = idx[s]
        try:
            psnr, _, _, _, _ = _v10_eval_one_seq(model, lq_p, gt_p, device)
        except Exception as e:
            print(f"[sentinel_eval] {kind}/{s} FAILED: {type(e).__name__}: {e}")
            continue
        out[s] = psnr
    model.train()
    if not out:
        return float("nan"), {}
    mean = float(sum(out.values()) / len(out))
    return mean, out


# ---- v10 checkpoint save (CL2 fix) ----------------------------------
def save_ckpt_v10(path, model, optim, sched, ema, step, best, val_history):
    """Save FULL model state (incl. vjepa_pos_emb.*); v9.0b's save_ckpt only
    saved pixel_branch and would silently lose new v10 params (CL2 blocker)."""
    ensure_dir(os.path.dirname(path))
    state = {
        "step": step, "best_metric": best,
        "val_history": list(val_history),
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": sched.state_dict(),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


# ---- crop curriculum --------------------------------------------------
def _set_crop_size(dataset, crop_size: int) -> int:
    """Recursively set crop_size on SDSDDataset instances inside ConcatDataset.
    Returns number of leaf datasets updated."""
    if isinstance(dataset, ConcatDataset):
        return sum(_set_crop_size(d, crop_size) for d in dataset.datasets)
    if hasattr(dataset, "crop_size"):
        dataset.crop_size = int(crop_size)
        return 1
    return 0


def _scale_optim_lr(optim, factor: float) -> None:
    for g in optim.param_groups:
        g["lr"] = g["lr"] * factor


def build_scheduler_v10(optim, opt_cfg, schedule, transition_lr_decay, transition_lr_iters,
                        accum_steps: int = 1):
    """LambdaLR with cosine + warmup + transition-window decay (CL6 + CO6/CP6 fix).
    sched.step() is called once per optim.step() (every accum_steps raw iters),
    so the lambda's 's' is OPTIM-STEP count, not raw-iter count. We convert all
    raw-iter quantities to optim-step units here so the cosine + transitions land
    at the right wall-clock iteration."""
    accum = max(1, int(accum_steps))
    total_raw = int(opt_cfg["total_iters"])
    warmup_raw = int(opt_cfg.get("warmup_iters", 0))
    total = max(1, math.ceil(total_raw / accum))     # optim steps total
    warmup = math.ceil(warmup_raw / accum)
    base = float(opt_cfg["lr"])
    floor = float(opt_cfg.get("min_lr", 0.0))
    min_mul = max(0.0, min(1.0, floor / base)) if base > 0 else 0.0
    # Crop transitions are scheduled in raw iter; convert to optim steps.
    transition_steps = [math.ceil(it / accum) for it, _ in schedule[1:]]
    window = max(1, math.ceil(int(transition_lr_iters) / accum))
    decay = float(transition_lr_decay)

    def lr_lambda(s: int) -> float:
        # 's' here = optim step (LambdaLR increments by 1 per sched.step()).
        if warmup and s < warmup:
            base_factor = (s + 1) / warmup
        else:
            prog = max(0.0, min(1.0, (s - warmup) / max(total - warmup, 1)))
            base_factor = min_mul + (1 - min_mul) * 0.5 * (1 + math.cos(math.pi * prog))
        for t_step in transition_steps:
            if t_step <= s < t_step + window:
                return base_factor * decay
        return base_factor

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def crop_for_iter(step: int, schedule: List[Tuple[int, int]]) -> int:
    """Return crop_size for given step. schedule = [(min_iter, crop), ...] sorted ascending."""
    cur = schedule[0][1]
    for min_iter, crop in schedule:
        if step >= min_iter:
            cur = crop
        else:
            break
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    out_dir = ensure_dir(cfg["output_dir"])
    ckpt_dir = ensure_dir(os.path.join(out_dir, "ckpts"))
    logger = setup_logger(log_dir=out_dir)
    dump_config(cfg, os.path.join(out_dir, "config.yaml"))

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))

    tcfg = cfg["train"]
    # Crop curriculum schedule: list of (start_iter, crop_size).
    # Default per V10_DESIGN_SPEC.md §5: 0→256, 1500→384, 3500→512.
    schedule_raw = tcfg.get("crop_schedule",
                            [{"iter": 0, "crop": 256},
                             {"iter": 1500, "crop": 384},
                             {"iter": 3500, "crop": 512}])
    schedule: List[Tuple[int, int]] = sorted(
        [(int(s["iter"]), int(s["crop"])) for s in schedule_raw], key=lambda x: x[0])
    initial_crop = schedule[0][1]
    transition_lr_decay = float(tcfg.get("transition_lr_decay", 0.5))
    transition_lr_iters = int(tcfg.get("transition_lr_iters", 100))
    logger.info(f"Crop curriculum: {schedule}; transition_lr_decay={transition_lr_decay}"
                f" for {transition_lr_iters} iter post-transition")

    train_set = build_dataset(
        cfg["data"]["train"],
        clip_len=tcfg.get("clip_len", 16),
        crop_size=initial_crop, is_train=True,
    )
    n_set = _set_crop_size(train_set, initial_crop)
    logger.info(f"  initial crop={initial_crop} applied to {n_set} leaf dataset(s)")
    sampler = build_sampler(cfg["data"]["train"], train_set)
    nw = cfg.get("num_workers", 4)

    def _make_train_loader():
        # CL3 fix: persistent_workers=False so workers always pick up the latest
        # train_set.crop_size after a curriculum transition.
        return DataLoader(
            train_set, batch_size=tcfg["batch_size"], sampler=sampler,
            shuffle=(sampler is None),
            num_workers=nw, pin_memory=True,
            drop_last=True, persistent_workers=False,
        )
    train_loader = _make_train_loader()

    val_set = build_dataset(
        cfg["data"]["val"],
        clip_len=tcfg.get("clip_len", 16),
        crop_size=tcfg.get("val_crop_size", 256), is_train=False,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=max(1, cfg.get("num_workers", 4) // 2),
                            pin_memory=True)
    audit_seqs = list_audit_seqs(cfg["data"]["val"]["root"], split="test")
    # CL4 fix: index sentinel datasets for in-trainer 5-seq eval
    indoor_idx = _v10_index_seqs(INDOOR_ROOT)
    outdoor_idx = _v10_index_seqs(OUTDOOR_ROOT)
    # CO3/CP4 + CO4: load corrected v9.0b sentinel baseline. If missing, sentinel best
    # promotion is disabled (step ckpts only saved).
    baseline_path = tcfg.get(
        "sentinel_baseline_json",
        "/cache/zihaohe/checkpoint/v10_audit/v9_0b_mix_sentinel_baseline/sentinel_baseline.json")
    v9_sent_baseline = None
    if os.path.isfile(baseline_path):
        with open(baseline_path) as f:
            v9_sent_baseline = _json.load(f)
        logger.info(f"Loaded v9 sentinel baseline from {baseline_path}: "
                    f"mean={v9_sent_baseline['sentinel_mean']:.3f} per_seq={v9_sent_baseline['per_seq']}")
    else:
        logger.warning(f"Sentinel baseline JSON missing at {baseline_path}; "
                        "best.pt sentinel promotion DISABLED (step ckpts only).")
    logger.info(f"Train n={len(train_set)}  Val n={len(val_set)}  Audit n={len(audit_seqs)}  "
                f"sentinel_idx: indoor={len(indoor_idx)} outdoor={len(outdoor_idx)}")

    model = VJepaLLVE_v10(cfg["model"]).to(device)
    model.freeze_backbone()
    n_train = sum(p.numel() for p in model.trainable_parameters())
    logger.info(f"Trainable params: {n_train/1e6:.1f}M")

    # Load v9.0b base — pos-emb stays at identity-init (CH3).
    anchor_path = cfg["optim"]["resume_from"]
    if not (anchor_path and os.path.isfile(anchor_path)):
        raise FileNotFoundError(f"Anchor ckpt missing: {anchor_path}")
    state = torch.load(anchor_path, map_location="cpu", weights_only=False)
    if "model" in state: v9_state = state["model"]
    elif "state_dict" in state: v9_state = state["state_dict"]
    elif "pixel_branch" in state:
        # v9.0b ckpt stores pixel_branch separately; load just that.
        v9_state = {f"pixel_branch.{k}": v for k, v in state["pixel_branch"].items()}
    else:
        v9_state = state
    model.load_v9_weights(v9_state)
    logger.info(f"Loaded v9.0b anchor from {anchor_path}")

    # ===== zero-iter audit (CL5 fix: informational only) =====
    # Identity-at-init is enforced by tools/v10_zero_iter_test.py (standalone).
    # Here we just LOG iter-0 baselines for the train log; no hard gate.
    logger.info("=== Zero-iter audit (informational; no hard gate, see CL5) ===")
    t0 = time.time()
    audit0 = audit_y512(model, audit_seqs, device)
    y512_0 = audit0["y512_framemean"]
    logger.info(f"Zero-iter audit ({time.time()-t0:.1f}s): "
                f"RGB-256={audit0['rgb256_framemean']:.3f}  Y-256={audit0['y256_framemean']:.3f}  "
                f"RGB-512={audit0['rgb512_framemean']:.3f}  Y-512={y512_0:.3f}")

    # ===== loss + optim =====
    lcfg = cfg["loss"]
    loss_fn = LLVELoss_v9(
        pix_weight=lcfg.get("pix_weight", 1.5),
        lpips_weight=lcfg.get("lpips_weight", 0.0),
        retinex_compose_weight=lcfg.get("retinex_compose_weight", 0.15),
        L_smooth_spatial_weight=lcfg.get("L_smooth_spatial_weight", 0.02),
        L_smooth_temporal_weight=lcfg.get("L_smooth_temporal_weight", 0.0),
        R_adjacent_weight=lcfg.get("R_adjacent_weight", 0.0),
        R_adjacent_scale=int(lcfg.get("R_adjacent_scale", 2)),
        pred_dod_weight=lcfg.get("pred_dod_weight", 0.0),
    )
    color_weight = float(lcfg.get("color_mean_weight", 0.1))
    logger.info(f"Loss: pix={lcfg.get('pix_weight',1.5)} retinex={lcfg.get('retinex_compose_weight',0.15)}"
                f" Ls={lcfg.get('L_smooth_spatial_weight',0.02)} color_mean={color_weight}")

    optim = build_optimizer(model, cfg["optim"])
    accum_for_sched = max(1, int(cfg["optim"].get("accum_steps", 1)))
    sched = build_scheduler_v10(optim, cfg["optim"], schedule,
                                 transition_lr_decay, transition_lr_iters,
                                 accum_steps=accum_for_sched)

    ema_decay = float(tcfg.get("ema_decay", 0.999))
    ema_start = int(tcfg.get("ema_start", 500))
    ema = ParameterEMA(model.trainable_parameters(), decay=ema_decay).to(device)

    # ===== training loop =====
    prec = cfg.get("precision", "bf16")
    use_amp = amp_enabled(prec); amp_dt = amp_dtype(prec)
    total_iters = int(cfg["optim"]["total_iters"])
    log_every = tcfg.get("log_every", 50)
    val_every = tcfg.get("val_every", 500)
    audit_every = tcfg.get("audit_every", 1000)
    save_every = tcfg.get("save_every", 1000)
    grad_clip = cfg["optim"].get("grad_clip", 1.0)
    accum = max(1, int(cfg["optim"].get("accum_steps", 1)))

    val_window = int(tcfg.get("val_window", 3))
    y512_history = collections.deque(maxlen=val_window); y512_history.append(y512_0)
    rgb256_history = collections.deque(maxlen=val_window)
    # CL4: best.pt is gated by sentinel-5 mean, not Y-512.
    # Start with -inf so first audit always promotes.
    best_sentinel = float("-inf")

    model.train(); step = 0
    loss_meter = AverageMeter()
    pix_m, ret_m, color_m = (AverageMeter() for _ in range(3))
    data_m, step_m = AverageMeter(), AverageMeter()
    nan_count = 0
    cur_crop = initial_crop
    next_transition_idx = 1  # we already applied schedule[0]
    transition_lr_until = -1; transition_log_until = -1

    data_iter = iter(train_loader)
    t_last = time.time()

    while step < total_iters:
        # --- crop curriculum transition (CL3 fix: full DataLoader rebuild + shape assert) ---
        if next_transition_idx < len(schedule) and step >= schedule[next_transition_idx][0]:
            new_crop = schedule[next_transition_idx][1]
            n_upd = _set_crop_size(train_set, new_crop)
            transition_lr_until = step + transition_lr_iters
            train_loader = _make_train_loader()
            data_iter = iter(train_loader)
            # one-batch shape probe so we KNOW workers picked up new crop
            probe = next(data_iter)
            ph, pw = probe["lq"].shape[-2:]
            assert (ph, pw) == (new_crop, new_crop), (
                f"[curriculum] crop transition FAILED: probe batch shape ({ph},{pw}) "
                f"!= expected ({new_crop},{new_crop}). Worker dataset out of sync?"
            )
            logger.info(f"[curriculum] step {step}: crop {new_crop} applied to {n_upd} ds; "
                        f"probe batch shape verified ({ph},{pw}); "
                        f"lr ×{transition_lr_decay} for next {transition_lr_iters} iter")
            cur_crop = new_crop
            next_transition_idx += 1
            transition_log_until = step + 5  # log lr every step for first 5 post-transition steps
            # we already consumed one batch from probe; use it as this iteration's batch
            batch = probe; data_m.update(0.0)
            lq = batch["lq"].to(device, non_blocking=True)
            hq = batch["hq"].to(device, non_blocking=True)
            # skip the regular fetch below by setting a flag-like behavior via continue-and-recompute
            _skip_fetch = True
        else:
            _skip_fetch = False
        # CL6: lr decay is inside the LambdaLR; auto-restored when window ends.
        if transition_lr_until >= 0 and step == transition_lr_until:
            logger.info(f"[curriculum] step {step}: transition lr-decay window ended (lambda auto-restored)")
            transition_lr_until = -1; transition_log_until = -1

        if not _skip_fetch:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader); batch = next(data_iter)
            data_m.update(time.time() - t_last)
            lq = batch["lq"].to(device, non_blocking=True)
            hq = batch["hq"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dt, enabled=use_amp):
            out = model(lq, video_hq=hq)
            losses = loss_fn(pred_pixels=out["pred_pixels"], hq=hq,
                             R_hat=out.get("R_hat"), L_hat=out.get("L_hat"))
            l_color = color_mean_loss(out["pred_pixels"].clamp(0, 1), hq)
            total = (losses["total"] + color_weight * l_color) / accum

        if not bool(torch.isfinite(total).item()):
            nan_count += 1
            optim.zero_grad(set_to_none=True)
            if (step + 1) % log_every == 0:
                logger.warning(f"  [NaN] step {step+1}: skipped (cum {nan_count})")
            step += 1; t_last = time.time(); continue

        total.backward()

        # CN-C smoke: log vjepa_pos_emb param norms + grad norms in first 5 train steps
        if step < 5 and model.vjepa_pos_emb is not None:
            pe = model.vjepa_pos_emb
            gp = pe.gamma_pos.item()
            n_pt = pe.P_t.weight.norm().item(); n_py = pe.P_y.weight.norm().item(); n_px = pe.P_x.weight.norm().item()
            g_pt = pe.P_t.weight.grad.norm().item() if pe.P_t.weight.grad is not None else float("nan")
            g_py = pe.P_y.weight.grad.norm().item() if pe.P_y.weight.grad is not None else float("nan")
            g_px = pe.P_x.weight.grad.norm().item() if pe.P_x.weight.grad is not None else float("nan")
            g_g  = pe.gamma_pos.grad.norm().item() if pe.gamma_pos.grad is not None else float("nan")
            logger.info(f"[posemb-trace step {step}] gamma={gp:.4e} | "
                        f"||P_t||={n_pt:.3e} ||P_y||={n_py:.3e} ||P_x||={n_px:.3e} | "
                        f"grad ||g_P_t||={g_pt:.3e} ||g_P_y||={g_py:.3e} ||g_P_x||={g_px:.3e} ||g_gamma||={g_g:.3e}")

        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            optim.step(); optim.zero_grad(set_to_none=True); sched.step()
            if (step + 1) >= ema_start:
                ema.update(model.trainable_parameters())

        loss_meter.update(losses["total"].item() + color_weight * l_color.item())
        if "pix" in losses: pix_m.update(losses["pix"].item())
        if "retinex" in losses: ret_m.update(losses["retinex"].item())
        color_m.update(l_color.item())
        step_m.update(time.time() - t_last); t_last = time.time()

        if transition_log_until >= 0 and step <= transition_log_until:
            logger.info(f"[curriculum-trace] step {step} lr={optim.param_groups[0]['lr']:.3e}")
        if (step + 1) % log_every == 0:
            lr = optim.param_groups[0]["lr"]
            logger.info(
                f"it {step+1}/{total_iters} crop={cur_crop} | loss {loss_meter.avg:.4f} | "
                f"pix {pix_m.avg:.4f} | ret {ret_m.avg:.4f} | color {color_m.avg:.4f} | "
                f"lr {lr:.2e} | step {step_m.avg:.2f}s | data {data_m.avg:.2f}s"
                + (f" | nan {nan_count}" if nan_count else "")
            )
            for m in (loss_meter, pix_m, ret_m, color_m): m.reset()

        do_audit = False
        if (step + 1) % val_every == 0:
            qm = quick_eval(model, val_loader, device)
            rgb256_history.append(qm["psnr"])
            sliding = sum(rgb256_history) / len(rgb256_history)
            logger.info(f"[val-rgb256 @ {step+1}] PSNR {qm['psnr']:.3f} SSIM {qm['ssim']:.4f} "
                        f"sliding({len(rgb256_history)}) {sliding:.3f}")
            if len(rgb256_history) >= 2 and qm["psnr"] > rgb256_history[-2]:
                do_audit = True

        if (step + 1) % audit_every == 0 or do_audit:
            # 1) Y-512 audit kept as informational warning metric (CL4: not promotion gate)
            t_a = time.time()
            au = audit_y512(model, audit_seqs, device)
            y512_history.append(au["y512_framemean"])
            sliding_y512 = sum(y512_history) / len(y512_history)
            logger.info(
                f"[audit @ {step+1}] ({time.time()-t_a:.1f}s) "
                f"RGB-256={au['rgb256_framemean']:.3f} Y-256={au['y256_framemean']:.3f} "
                f"RGB-512={au['rgb512_framemean']:.3f} Y-512={au['y512_framemean']:.3f} "
                f"| sliding(3) Y-512 = {sliding_y512:.3f}  [info only, CL4]")
            # 2) sentinel-5 eval: tile=512 on hard indoor + outdoor + legacy guard
            t_s = time.time()
            sent_mean, sent_per = sentinel_eval(model, indoor_idx, outdoor_idx, device)
            seq_breakdown = " ".join([f"{k}={v:.2f}" for k, v in sent_per.items()])
            logger.info(f"[sentinel @ {step+1}] ({time.time()-t_s:.1f}s) mean={sent_mean:.3f} | {seq_breakdown}")
            # 3) best.pt promoted ONLY when (a) sentinel mean improved
            #    AND (b) per-seq gate vs v9 baseline (CO4/CP3): every seq within -0.5 dB.
            #    If baseline is missing, do not promote.
            promote = False
            reason = ""
            if v9_sent_baseline is None:
                reason = "no v9 baseline JSON; sentinel-best promotion disabled"
            elif sent_mean <= best_sentinel:
                reason = f"sentinel mean {sent_mean:.3f} <= best_sentinel {best_sentinel:.3f}"
            else:
                v9_per = v9_sent_baseline["per_seq"]
                regressed = [
                    (s, sent_per[s], v9_per.get(s, float("nan")))
                    for s in sent_per if s in v9_per and sent_per[s] < v9_per[s] - 0.5
                ]
                if regressed:
                    reason = f"per-seq guard tripped: " + ", ".join(
                        [f"{s}={p:.2f}<{b-0.5:.2f}" for s, p, b in regressed])
                else:
                    promote = True
            if promote:
                best_sentinel = sent_mean
                save_ckpt_v10(os.path.join(ckpt_dir, "best.pt"),
                              model, optim, sched, ema, step + 1, best_sentinel, y512_history)
                logger.info(f"        new best (sentinel-5 mean) = {best_sentinel:.3f}")
            else:
                logger.info(f"        not promoted: {reason}")

        if (step + 1) % save_every == 0:
            save_ckpt_v10(os.path.join(ckpt_dir, f"step_{step+1}.pt"),
                      model, optim, sched, ema, step + 1, best_sentinel, y512_history)

        step += 1

    save_ckpt_v10(os.path.join(ckpt_dir, "final.pt"),
              model, optim, sched, ema, step, best_sentinel, y512_history)
    v9_mean_str = f"{v9_sent_baseline['sentinel_mean']:.3f}" if v9_sent_baseline else "n/a"
    logger.info(
        f"v10 done. Best sentinel-5 mean = {best_sentinel:.3f}; "
        f"v9.0b sentinel baseline = {v9_mean_str}; NaN={nan_count}")


if __name__ == "__main__":
    main()
