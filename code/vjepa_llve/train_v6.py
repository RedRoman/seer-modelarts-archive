"""train_v6.py — V-JEPA-LLVE v6 training loop.

Stage A: rectified flow z_LQ → z_HQ in V-JEPA latent (deepest layer).
Stage B: V-JEPA-grounded pixel decoder (LQ pixels + V-JEPA cross-attn → HQ).

Training (per batch):
    1. Encode LQ + HQ → 4 V-JEPA layers each.
    2. Sample t ~ U(0,1); z_t = (1-t)·E(LQ)[-1] + t·E(HQ)[-1]; v_target = E(HQ)[-1] - E(LQ)[-1].
    3. v_pred = LatentDiT(z_t, t).
    4. Decoder takes raw E(LQ) at l5/l11/l17 and TEACHER-FORCED E(HQ) at l23.
    5. Loss = flow_mse(v_pred, v_target) + char(pred_pixels, hq) + LPIPS + temporal.

Inference (val/test):
    1. Encode LQ → 4 V-JEPA layers.
    2. ODE solve: refined = LatentDiT.sample(E(LQ)[-1]).
    3. Decoder uses raw E(LQ) at l5/l11/l17 and `refined` at l23.

Disk-safe ckpts: keep N latest step_*.pt, always keep best.pt + final.pt.
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
from losses.v6 import LLVELoss_v6
from models.vjepa_llve_v6 import VJepaLLVE_v6
from utils import (
    AverageMeter, count_parameters, dump_config, ensure_dir, load_config,
    psnr, set_seed, setup_logger, ssim,
)


def amp_dtype(p: str): return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(p, torch.float32)
def amp_enabled(p: str): return p in ("fp16", "bf16")


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
    base = float(opt_cfg["lr"])
    floor = float(opt_cfg.get("min_lr", 0.0))
    min_mul = max(0.0, min(1.0, floor / base)) if base > 0 else 0.0

    def lr_lambda(s: int) -> float:
        if warmup and s < warmup:
            return (s + 1) / warmup
        prog = max(0.0, min(1.0, (s - warmup) / max(total - warmup, 1)))
        return min_mul + (1 - min_mul) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def save_ckpt(path, model, optim, sched, step, best):
    ensure_dir(os.path.dirname(path))
    state = {
        "step": step, "best_metric": best,
        "latent_dit": model.latent_dit.state_dict(),
        "pixel_decoder": model.pixel_decoder.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": sched.state_dict(),
    }
    torch.save(state, path)


def prune_step_ckpts(ckpt_dir: str, keep_n: int) -> None:
    if keep_n <= 0:
        return
    files = []
    for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
        m = re.match(r"step_(\d+)\.pt", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    for _, p in files[:-keep_n] if len(files) > keep_n else []:
        try:
            os.remove(p)
        except OSError:
            pass


def load_ckpt(path, model, optim=None, sched=None, load_optim=True, load_sched=True):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.latent_dit.load_state_dict(state["latent_dit"], strict=False)
    model.pixel_decoder.load_state_dict(state["pixel_decoder"], strict=False)
    if load_optim and optim is not None and "optim" in state:
        try:
            optim.load_state_dict(state["optim"])
        except Exception as e:
            print(f"[ckpt] optim load failed ({e}); skipping.")
    if load_sched and sched is not None and "scheduler" in state:
        try:
            sched.load_state_dict(state["scheduler"])
        except Exception as e:
            print(f"[ckpt] scheduler load failed ({e}); skipping.")
    if not load_optim and not load_sched:
        return 0, -1.0
    return int(state.get("step", 0)), float(state.get("best_metric", -1.0))


@torch.no_grad()
def run_quick_eval(model, val_loader, device, max_samples=14):
    """Inference-mode val: ODE + decoder. Computes PSNR/SSIM."""
    model.eval()
    psnr_m, ssim_m = AverageMeter(), AverageMeter()
    seen = 0
    for batch in val_loader:
        lq = batch["lq"].to(device, non_blocking=True)
        hq = batch["hq"].to(device, non_blocking=True)
        out = model.sample(lq)
        psnr_m.update(psnr(out["pred_pixels"], hq).item())
        ssim_m.update(ssim(out["pred_pixels"], hq).item())
        seen += lq.shape[0]
        if seen >= max_samples:
            break
    model.train()
    return {"psnr": psnr_m.avg, "ssim": ssim_m.avg}


def assert_disk_space(path: str, min_free_gb: float = 30.0) -> None:
    import shutil
    try:
        free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"DISK SAFETY: only {free_gb:.1f} GB free on {path} (< {min_free_gb} GB). "
            f"Clean up before training."
        )


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
    assert_disk_space(out_dir, min_free_gb=30.0)

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))

    tcfg = cfg["train"]
    train_set = build_dataset(
        cfg["data"]["train"],
        clip_len=tcfg.get("clip_len", 16),
        crop_size=tcfg.get("crop_size", 256), is_train=True,
    )
    sampler = build_sampler(cfg["data"]["train"], train_set)
    train_loader = DataLoader(
        train_set, batch_size=tcfg["batch_size"],
        sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
        drop_last=True, persistent_workers=cfg.get("num_workers", 4) > 0,
    )
    logger.info(f"Train: {len(train_set)} samples, B={tcfg['batch_size']} accum={cfg['optim'].get('accum_steps', 1)}")

    val_loader = None
    if cfg["data"].get("val") is not None:
        val_set = build_dataset(
            cfg["data"]["val"],
            clip_len=tcfg.get("clip_len", 16),
            crop_size=tcfg.get("crop_size", 256), is_train=False,
        )
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                num_workers=max(1, cfg.get("num_workers", 4) // 2),
                                pin_memory=True)
        logger.info(f"Val:   {len(val_set)} samples")

    # --- model + loss ---
    model = VJepaLLVE_v6(cfg["model"]).to(device)
    model.freeze_backbone()
    n_dit = sum(p.numel() for p in model.latent_dit.parameters())
    n_dec = sum(p.numel() for p in model.pixel_decoder.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    logger.info(f"LatentDiT: {n_dit/1e6:.1f}M | PixelDecoder: {n_dec/1e6:.1f}M | Total trainable: {n_train/1e6:.1f}M")

    lcfg = cfg["loss"]
    loss_fn = LLVELoss_v6(
        flow_weight=lcfg.get("flow_weight", 1.0),
        pix_weight=lcfg.get("pix_weight", 1.0),
        lpips_weight=lcfg.get("lpips_weight", 0.1),
        temporal_weight=lcfg.get("temporal_weight", 0.1),
    ).to(device)

    optim = build_optimizer(model, cfg["optim"])
    sched = build_scheduler(optim, cfg["optim"])

    start_step, best = 0, -1.0
    resume = cfg["optim"].get("resume_from")
    reset = bool(cfg["optim"].get("resume_reset_optim", False))
    if resume and os.path.isfile(resume):
        start_step, best = load_ckpt(
            resume, model, optim, sched,
            load_optim=not reset, load_sched=not reset,
        )
        logger.info(f"Resumed from {resume}: step={start_step}")

    prec = cfg.get("precision", "bf16")
    use_amp = amp_enabled(prec)
    amp_dt = amp_dtype(prec)
    use_scaler = (prec == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    logger.info(f"Precision: {prec}")

    total_iters = int(cfg["optim"]["total_iters"])
    log_every = tcfg.get("log_every", 50)
    save_every = tcfg.get("save_every", 2000)
    val_every = tcfg.get("val_every", 1000)
    grad_clip = cfg["optim"].get("grad_clip", 1.0)
    accum = max(1, int(cfg["optim"].get("accum_steps", 1)))
    keep_n = int(tcfg.get("keep_n_step_ckpts", 2))

    model.train()
    step = start_step
    loss_meter = AverageMeter()
    flow_m, pix_m, lpips_m, temp_m = (AverageMeter() for _ in range(4))
    data_m, step_m = AverageMeter(), AverageMeter()

    nan_count = 0
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

        data_m.update(time.time() - t_last)
        lq = batch["lq"].to(device, non_blocking=True)
        hq = batch["hq"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dt, enabled=use_amp):
            out = model(lq, video_hq=hq)
            losses = loss_fn(
                v_pred=out.get("v_pred"),
                v_target=out.get("v_target"),
                pred_pixels=out["pred_pixels"],
                hq=hq,
            )
            total = losses["total"] / accum

        bad = not bool(torch.isfinite(total).item())
        nan_recent.append(1 if bad else 0)
        if len(nan_recent) > NAN_WINDOW:
            nan_recent.pop(0)
        if bad:
            nan_count += 1
            optim.zero_grad(set_to_none=True)
            if (step + 1) % log_every == 0:
                logger.warning(f"  [NaN] step {step+1}: skipped (cum {nan_count})")
            if sum(nan_recent) > NAN_ABORT:
                logger.error("NaN rate too high, aborting.")
                save_ckpt(os.path.join(ckpt_dir, "aborted_nan.pt"),
                          model, optim, sched, step + 1, best)
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
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                optim.step()
            optim.zero_grad(set_to_none=True)
            sched.step()

        loss_meter.update(losses["total"].item())
        if "flow"     in losses: flow_m.update(losses["flow"].item())
        if "pix"      in losses: pix_m.update(losses["pix"].item())
        if "lpips"    in losses: lpips_m.update(losses["lpips"].item())
        if "temporal" in losses: temp_m.update(losses["temporal"].item())
        step_m.update(time.time() - t_last)
        t_last = time.time()

        if (step + 1) % log_every == 0:
            lr = optim.param_groups[0]["lr"]
            parts = [
                f"it {step+1}/{total_iters}",
                f"loss {loss_meter.avg:.4f}",
                f"flow {flow_m.avg:.4f}" if flow_m.count else "",
                f"pix {pix_m.avg:.4f}" if pix_m.count else "",
                f"lpips {lpips_m.avg:.4f}" if lpips_m.count else "",
                f"temp {temp_m.avg:.4f}" if temp_m.count else "",
                f"lr {lr:.2e}",
                f"step {step_m.avg:.2f}s",
                f"data {data_m.avg:.2f}s",
            ]
            if nan_count > 0:
                parts.append(f"nan {nan_count}")
            logger.info(" | ".join(p for p in parts if p))
            loss_meter.reset()
            flow_m.reset(); pix_m.reset(); lpips_m.reset(); temp_m.reset()

        if val_loader is not None and (step + 1) % val_every == 0:
            m = run_quick_eval(model, val_loader, device)
            logger.info(f"[val @ {step+1}] PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}")
            if m["psnr"] > best:
                best = m["psnr"]
                save_ckpt(os.path.join(ckpt_dir, "best.pt"),
                          model, optim, sched, step + 1, best)
                logger.info(f"        new best PSNR {best:.3f}")

        if (step + 1) % save_every == 0:
            save_ckpt(os.path.join(ckpt_dir, f"step_{step+1}.pt"),
                      model, optim, sched, step + 1, best)
            prune_step_ckpts(ckpt_dir, keep_n)

        step += 1

    save_ckpt(os.path.join(ckpt_dir, "final.pt"),
              model, optim, sched, step, best)
    prune_step_ckpts(ckpt_dir, keep_n)
    logger.info(f"Training done. Best PSNR = {best:.3f}  (NaN: {nan_count})")


if __name__ == "__main__":
    main()
