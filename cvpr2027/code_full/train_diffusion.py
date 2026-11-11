"""train_diffusion.py — V-JEPA-LLVE v5 training loop.

Rectified flow z_LQ → z_HQ in SD-VAE latent space:
    z_LQ = VAE(LQ),  z_HQ = VAE(HQ)
    t ~ U(0, 1)
    z_t = (1-t) * z_LQ + t * z_HQ
    v_target = z_HQ - z_LQ
    v_pred = DiT(z_t, t, ctx)        ctx = V-JEPA features (predictor-refined)
    Loss = w_flow * MSE(v_pred, v_target)
         + w_cf   * L1(refined, E(HQ).detach())

Inference: ODE Euler from z_LQ for `inference_steps` (default 8) steps.

Disk-safe checkpointing:
    - Always overwrites best.pt
    - Keeps only last `keep_n_step_ckpts` step_X.pt files (rolling delete)
    - final.pt written once at end
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data import build_dataset, build_sampler
from losses.core import LLVELossV5
from models.vjepa_diffusion import VJepaDiffusion
from utils import (
    AverageMeter, count_parameters, dump_config, ensure_dir,
    load_config, psnr, set_seed, setup_logger, ssim,
)


# ===========================================================================
# AMP / optim helpers
# ===========================================================================
def amp_dtype(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)


def amp_enabled(precision: str) -> bool:
    return precision in ("fp16", "bf16")


def build_optimizer(model, opt_cfg):
    return torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.99))),
    )


def build_scheduler(optim, opt_cfg):
    total = int(opt_cfg["total_iters"])
    warmup = int(opt_cfg.get("warmup_iters", 0))
    base_lr = float(opt_cfg["lr"])
    min_lr = float(opt_cfg.get("min_lr", 0.0))
    min_mul = max(0.0, min(1.0, min_lr / base_lr)) if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        progress = max(0.0, min(1.0, progress))
        cos_mul = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_mul + (1.0 - min_mul) * cos_mul

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


# ===========================================================================
# Checkpointing with rolling delete (disk-safe)
# ===========================================================================
def save_checkpoint(path, model, optim, scheduler, step, best_metric):
    ensure_dir(os.path.dirname(path))
    state = {
        "step": step,
        "best_metric": best_metric,
        "dit": model.dit.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if model.predictor_enabled:
        state["refiner"] = model.refiner.state_dict()
    torch.save(state, path)


def prune_step_ckpts(ckpt_dir: str, keep_n: int) -> None:
    """Delete old step_*.pt files, keeping only the most recent `keep_n`.
    best.pt and final.pt are never touched."""
    if keep_n <= 0:
        return
    step_files = []
    for path in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
        m = re.match(r"step_(\d+)\.pt", os.path.basename(path))
        if m:
            step_files.append((int(m.group(1)), path))
    step_files.sort()
    to_delete = step_files[:-keep_n] if len(step_files) > keep_n else []
    for _, path in to_delete:
        try:
            os.remove(path)
        except OSError:
            pass


def load_checkpoint(path, model, optim=None, scheduler=None,
                    load_optim=True, load_scheduler=True):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.dit.load_state_dict(state["dit"], strict=False)
    if model.predictor_enabled and "refiner" in state:
        model.refiner.load_state_dict(state["refiner"], strict=False)
    if load_optim and optim is not None and "optim" in state:
        try:
            optim.load_state_dict(state["optim"])
        except Exception as e:
            print(f"[ckpt] optim state load failed ({e}); skipping.")
    if load_scheduler and scheduler is not None and "scheduler" in state:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as e:
            print(f"[ckpt] scheduler state load failed ({e}); skipping.")
    if not load_optim and not load_scheduler:
        return 0, -1.0
    return int(state.get("step", 0)), float(state.get("best_metric", -1.0))


# ===========================================================================
# Quick eval (sample-based)
# ===========================================================================
@torch.no_grad()
def run_quick_eval(model, val_loader, device, num_steps, max_samples=14):
    model.eval()
    psnr_m, ssim_m = AverageMeter(), AverageMeter()
    seen = 0
    for batch in val_loader:
        lq = batch["lq"].to(device, non_blocking=True)
        hq = batch["hq"].to(device, non_blocking=True)
        out = model.sample(lq, num_steps=num_steps)
        psnr_m.update(psnr(out["pred"], hq).item())
        ssim_m.update(ssim(out["pred"], hq).item())
        seen += lq.shape[0]
        if seen >= max_samples:
            break
    model.train()
    return {"psnr": psnr_m.avg, "ssim": ssim_m.avg}


# ===========================================================================
# Disk-safety check before launching
# ===========================================================================
def assert_disk_space(path: str, min_free_gb: float = 30.0) -> None:
    """Refuse to start training if the workspace volume has < min_free_gb free.
    Mandatory rule (memory: feedback_disk_management): training fills disk if
    not bounded; small workspaces (50G ModelArts overlay) get reclaimed."""
    import shutil
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
    except Exception:
        return
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"DISK SAFETY: only {free_gb:.1f} GB free on {path} (< {min_free_gb} GB). "
            f"Clean up before training. df -h, then rm step_*.pt or older runs."
        )


# ===========================================================================
# Main
# ===========================================================================
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

    # Disk-safety check (memory: feedback_disk_management)
    assert_disk_space(out_dir, min_free_gb=30.0)

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))

    tcfg = cfg["train"]
    data_train = cfg["data"]["train"]
    data_val = cfg["data"].get("val")

    train_set = build_dataset(
        data_train, clip_len=tcfg.get("clip_len", 16),
        crop_size=tcfg.get("crop_size", 256), is_train=True,
    )
    sampler = build_sampler(data_train, train_set)
    train_loader = DataLoader(
        train_set, batch_size=tcfg["batch_size"],
        sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
        drop_last=True, persistent_workers=cfg.get("num_workers", 4) > 0,
    )
    logger.info(
        f"Train set: {len(train_set)} samples | "
        f"effective batch={tcfg['batch_size']} * accum{cfg['optim'].get('accum_steps', 1)}"
    )

    val_loader = None
    if data_val is not None:
        val_set = build_dataset(
            data_val, clip_len=tcfg.get("clip_len", 16),
            crop_size=tcfg.get("crop_size", 256), is_train=False,
        )
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                num_workers=max(1, cfg.get("num_workers", 4) // 2),
                                pin_memory=True)
        logger.info(f"Val set:   {len(val_set)} samples")

    # --- model + loss ---
    model = VJepaDiffusion(cfg["model"]).to(device)
    model.freeze_backbone()
    n_dit = sum(p.numel() for p in model.dit.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    logger.info(
        f"DiT: {n_dit/1e6:.1f}M | total trainable: {n_train/1e6:.1f}M"
    )

    lcfg = cfg["loss"]
    loss_fn = LLVELossV5(
        flow_weight = lcfg.get("flow_weight", 1.0),
        counterfactual_weight = lcfg.get("counterfactual_weight", 0.1),
        lpips_weight = lcfg.get("lpips_weight", 0.0),
        temporal_weight = lcfg.get("temporal_weight", 0.0),
    ).to(device)

    optim_state = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optim_state, cfg["optim"])

    start_step, best_metric = 0, -1.0
    resume_from = cfg["optim"].get("resume_from")
    reset_optim = bool(cfg["optim"].get("resume_reset_optim", False))
    if resume_from and os.path.isfile(resume_from):
        start_step, best_metric = load_checkpoint(
            resume_from, model, optim_state, scheduler,
            load_optim=not reset_optim, load_scheduler=not reset_optim,
        )
        logger.info(f"Resumed from {resume_from}: step={start_step}")

    prec = cfg.get("precision", "bf16")
    use_amp = amp_enabled(prec)
    amp_dt = amp_dtype(prec)
    use_scaler = (prec == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    logger.info(f"Precision: {prec}")

    needs_cf = lcfg.get("counterfactual_weight", 0.0) > 0 and model.predictor_enabled
    needs_decoded = (lcfg.get("lpips_weight", 0.0) > 0
                     or lcfg.get("temporal_weight", 0.0) > 0)

    total_iters = int(cfg["optim"]["total_iters"])
    log_every = tcfg.get("log_every", 50)
    save_every = tcfg.get("save_every", 2000)
    val_every = tcfg.get("val_every", 1000)
    grad_clip = cfg["optim"].get("grad_clip", 1.0)
    accum = max(1, int(cfg["optim"].get("accum_steps", 1)))
    keep_n = int(tcfg.get("keep_n_step_ckpts", 2))
    inference_steps = int(cfg["model"].get("inference_steps", 8))

    model.train()
    step = start_step
    loss_meter = AverageMeter()
    flow_meter = AverageMeter()
    cf_meter = AverageMeter()
    data_time_m = AverageMeter()
    step_time_m = AverageMeter()

    nan_step_count = 0
    nan_recent = []
    NAN_WINDOW = 200
    NAN_ABORT = 50

    data_iter = iter(train_loader)
    t_last = time.time()

    while step < total_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        data_time_m.update(time.time() - t_last)

        lq = batch["lq"].to(device, non_blocking=True)
        hq = batch["hq"].to(device, non_blocking=True)
        B = lq.shape[0]

        # --- frozen encoders ---
        z_LQ = model.encode_pixels_to_latent(lq)
        z_HQ = model.encode_pixels_to_latent(hq)
        raw_feats_lq = model.encode_video_features(lq)
        e_hq_target = None
        if needs_cf:
            raw_feats_hq = model.encode_video_features(hq)
            e_hq_target = raw_feats_hq[model.predictor_target_layer]

        # --- sample t, build flow target ---
        t = torch.rand(B, device=device, dtype=torch.float32)
        t_b = t.view(B, 1, 1, 1, 1).to(z_LQ.dtype)
        z_t = (1.0 - t_b) * z_LQ + t_b * z_HQ
        v_target = (z_HQ - z_LQ).to(z_LQ.dtype)

        # --- DiT + refiner forward (under autocast) ---
        with torch.autocast(device_type="cuda", dtype=amp_dt, enabled=use_amp):
            ctx, refined = model.make_ctx_layers(raw_feats_lq, target_dtype=z_LQ.dtype)
            v_pred = model(z_t, t, ctx)

            decoded = None
            if needs_decoded:
                # decode predicted z_HQ approximation: z_LQ + v_pred (single-step)
                # cheap proxy for full ODE; use sparingly
                decoded = model.decode_latent_to_pixels((z_LQ + v_pred).to(model.vae.dtype))

            losses = loss_fn(
                v_pred=v_pred,
                v_target=v_target,
                refined=refined,
                e_hq_target=e_hq_target,
                decoded_pred=decoded,
                hq=hq,
            )
            total = losses["total"] / accum

        bad = not bool(torch.isfinite(total).item())
        nan_recent.append(1 if bad else 0)
        if len(nan_recent) > NAN_WINDOW:
            nan_recent.pop(0)
        if bad:
            nan_step_count += 1
            optim_state.zero_grad(set_to_none=True)
            if (step + 1) % log_every == 0:
                logger.warning(
                    f"  [NaN guard] step {step+1}: skipped (cum {nan_step_count})"
                )
            if sum(nan_recent) > NAN_ABORT:
                logger.error("NaN rate too high, aborting.")
                save_checkpoint(os.path.join(ckpt_dir, "aborted_nan.pt"),
                                model, optim_state, scheduler, step + 1, best_metric)
                return
            step += 1
            t_last = time.time()
            continue

        if use_scaler:
            scaler.scale(total).backward()
        else:
            total.backward()

        if (step + 1) % accum == 0:
            if use_scaler:
                scaler.unscale_(optim_state)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                scaler.step(optim_state)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                optim_state.step()
            optim_state.zero_grad(set_to_none=True)
            scheduler.step()

        loss_meter.update(losses["total"].item())
        flow_meter.update(losses.get("flow", torch.tensor(0.0)).item())
        if "counterfactual" in losses:
            cf_meter.update(losses["counterfactual"].item())
        step_time_m.update(time.time() - t_last)
        t_last = time.time()

        if (step + 1) % log_every == 0:
            lr = optim_state.param_groups[0]["lr"]
            parts = [
                f"it {step+1}/{total_iters}",
                f"loss {loss_meter.avg:.4f}",
                f"flow {flow_meter.avg:.4f}",
            ]
            if cf_meter.count > 0:
                parts.append(f"cf {cf_meter.avg:.4f}")
            parts.extend([
                f"lr {lr:.2e}",
                f"step {step_time_m.avg:.2f}s",
                f"data {data_time_m.avg:.2f}s",
                f"t̄ {t.mean().item():.2f}",
            ])
            if nan_step_count > 0:
                parts.append(f"nan_skips {nan_step_count}")
            logger.info(" | ".join(parts))
            loss_meter.reset(); flow_meter.reset(); cf_meter.reset()

        if val_loader is not None and (step + 1) % val_every == 0:
            m = run_quick_eval(model, val_loader, device, num_steps=inference_steps)
            logger.info(f"[val @ {step+1}] PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}")
            if m["psnr"] > best_metric:
                best_metric = m["psnr"]
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"),
                                model, optim_state, scheduler, step + 1, best_metric)
                logger.info(f"        new best PSNR {best_metric:.3f}")

        if (step + 1) % save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"step_{step+1}.pt"),
                            model, optim_state, scheduler, step + 1, best_metric)
            prune_step_ckpts(ckpt_dir, keep_n)

        step += 1

    save_checkpoint(os.path.join(ckpt_dir, "final.pt"),
                    model, optim_state, scheduler, step, best_metric)
    prune_step_ckpts(ckpt_dir, keep_n)
    logger.info(
        f"Training done. Best PSNR = {best_metric:.3f}  (NaN skips: {nan_step_count})"
    )


if __name__ == "__main__":
    main()
