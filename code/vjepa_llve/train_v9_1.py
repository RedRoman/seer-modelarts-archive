"""train_v9_1.py — V-JEPA-LLVE v9.1a training loop.

Differences vs train_v9:
  - Frozen RAFT-small for adjacent-pair flow + occlusion mask (J7).
  - LLVELoss_v9_1 with pred_warp + R_warp + lowered R_adjacent.
  - Extension rule (J1): if last 3 vals still improving at iter ~total_iters,
    extend to extend_to (default 15000).
"""
from __future__ import annotations

import argparse
import collections
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
from losses.v9_1 import LLVELoss_v9_1
from models.raft_flow import build_raft_small, compute_flow_and_mask
from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import (
    AverageMeter, ParameterEMA, dump_config, ensure_dir, load_config,
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


def build_scheduler(optim, opt_cfg, total_iters: int):
    warmup = int(opt_cfg.get("warmup_iters", 0))
    base = float(opt_cfg["lr"])
    floor = float(opt_cfg.get("min_lr", 0.0))
    min_mul = max(0.0, min(1.0, floor / base)) if base > 0 else 0.0

    def lr_lambda(s: int) -> float:
        if warmup and s < warmup:
            return (s + 1) / warmup
        prog = max(0.0, min(1.0, (s - warmup) / max(total_iters - warmup, 1)))
        return min_mul + (1 - min_mul) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def save_ckpt(path, model, optim, sched, ema, step, best, best_ema, val_history):
    ensure_dir(os.path.dirname(path))
    state = {
        "step": step, "best_metric": best, "best_ema_metric": best_ema,
        "val_history": list(val_history),
        "pixel_branch": model.pixel_branch.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": sched.state_dict(),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
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


def load_ckpt(path, model, optim=None, sched=None, ema=None,
              load_optim=True, load_sched=True):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    if load_optim and optim is not None and "optim" in state:
        try: optim.load_state_dict(state["optim"])
        except Exception as e: print(f"[ckpt] optim load failed ({e}); skipping.")
    if load_sched and sched is not None and "scheduler" in state:
        try: sched.load_state_dict(state["scheduler"])
        except Exception as e: print(f"[ckpt] scheduler load failed ({e}); skipping.")
    if ema is not None and "ema" in state:
        try: ema.load_state_dict(state["ema"])
        except Exception as e: print(f"[ckpt] ema load failed ({e}); skipping.")
    if not load_optim and not load_sched:
        return 0, -1.0, -1.0, []
    return (
        int(state.get("step", 0)),
        float(state.get("best_metric", -1.0)),
        float(state.get("best_ema_metric", -1.0)),
        list(state.get("val_history", [])),
    )


@torch.no_grad()
def run_quick_eval(model, val_loader, device, max_samples=14):
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
        raise RuntimeError(f"DISK SAFETY: only {free_gb:.1f} GB free on {path}")


def should_extend(val_history, current_iter: int, base_iters: int) -> bool:
    """J1 extension rule: extend if last 3 vals are still monotonically improving
    when we reach base_iters (or 9500 within base_iters)."""
    if current_iter < base_iters - 500:
        return False
    if len(val_history) < 3:
        return False
    a, b, c = val_history[-3], val_history[-2], val_history[-1]
    return (b > a) and (c > b)


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
        train_set, batch_size=tcfg["batch_size"], sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
        drop_last=True, persistent_workers=cfg.get("num_workers", 4) > 0,
    )
    logger.info(
        f"Train: {len(train_set)} samples, B={tcfg['batch_size']} "
        f"accum={cfg['optim'].get('accum_steps', 1)} "
        f"effective={tcfg['batch_size'] * cfg['optim'].get('accum_steps', 1)}"
    )

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

    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    n_pb = sum(p.numel() for p in model.pixel_branch.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    logger.info(f"PixelBranch v9: {n_pb/1e6:.1f}M | Total trainable: {n_train/1e6:.1f}M")

    raft_cfg = cfg["model"].get("raft", {})
    raft = build_raft_small(device, weights=raft_cfg.get("weights", "C_T_V2"))
    n_raft = sum(p.numel() for p in raft.parameters())
    logger.info(f"RAFT-small (frozen): {n_raft/1e6:.1f}M params")
    raft_threshold = float(raft_cfg.get("occlusion_fb_threshold", 1.0))
    raft_iter = int(raft_cfg.get("n_iter", 12))

    lcfg = cfg["loss"]
    loss_fn = LLVELoss_v9_1(
        pix_weight=lcfg.get("pix_weight", 1.0),
        lpips_weight=lcfg.get("lpips_weight", 0.1),
        retinex_compose_weight=lcfg.get("retinex_compose_weight", 0.5),
        L_smooth_spatial_weight=lcfg.get("L_smooth_spatial_weight", 0.05),
        L_smooth_temporal_weight=lcfg.get("L_smooth_temporal_weight", 0.10),
        R_adjacent_weight=lcfg.get("R_adjacent_weight", 0.01),
        R_adjacent_scale=lcfg.get("R_adjacent_scale", 2),
        pred_dod_weight=lcfg.get("pred_dod_weight", 0.05),
        pred_warp_weight=lcfg.get("pred_warp_weight", 0.05),
        R_warp_weight=lcfg.get("R_warp_weight", 0.10),
    ).to(device)

    base_iters = int(cfg["optim"]["total_iters"])
    extend_to = int(cfg["optim"].get("extend_to", base_iters))
    sched_total = max(extend_to, base_iters)  # cosine planned for max possible run

    optim = build_optimizer(model, cfg["optim"])
    sched = build_scheduler(optim, cfg["optim"], total_iters=sched_total)

    ema_decay = float(tcfg.get("ema_decay", 0.999))
    ema_start = int(tcfg.get("ema_start", 1000))
    ema = ParameterEMA(model.trainable_parameters(), decay=ema_decay).to(device)
    logger.info(f"EMA enabled: decay={ema_decay} start={ema_start}")
    logger.info(f"Schedule: base={base_iters}, may extend to {extend_to}")

    val_window = int(tcfg.get("val_window", 3))
    val_history: collections.deque = collections.deque(maxlen=val_window)
    val_history_ema: collections.deque = collections.deque(maxlen=val_window)
    val_history_full = []  # for extension-rule check (full sliding history)

    start_step, best, best_ema = 0, -1.0, -1.0
    resume = cfg["optim"].get("resume_from")
    reset = bool(cfg["optim"].get("resume_reset_optim", False))
    if resume and os.path.isfile(resume):
        start_step, best, best_ema, vh = load_ckpt(
            resume, model, optim, sched, ema,
            load_optim=not reset, load_sched=not reset,
        )
        ema.to(device)
        for v in vh[-val_window:]:
            val_history.append(v)
        logger.info(f"Resumed from {resume}: step={start_step}")

    prec = cfg.get("precision", "bf16")
    use_amp = amp_enabled(prec)
    amp_dt = amp_dtype(prec)
    use_scaler = (prec == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    logger.info(f"Precision: {prec}")

    log_every = tcfg.get("log_every", 50)
    save_every = tcfg.get("save_every", 2000)
    val_every = tcfg.get("val_every", 500)
    grad_clip = cfg["optim"].get("grad_clip", 1.0)
    accum = max(1, int(cfg["optim"].get("accum_steps", 1)))
    keep_n = int(tcfg.get("keep_n_step_ckpts", 2))

    model.train()
    step = start_step
    loss_meter = AverageMeter()
    pix_m, lpips_m, ret_m, lsm_m, ltm_m, radj_m, dod_m, pw_m, rw_m, mask_m = (
        AverageMeter() for _ in range(10)
    )
    data_m, step_m, flow_m = AverageMeter(), AverageMeter(), AverageMeter()

    nan_count = 0
    nan_recent = []
    NAN_WINDOW = 200
    NAN_ABORT = 50

    data_iter = iter(train_loader)
    t_last = time.time()

    total_iters = base_iters  # will be extended if J1 rule fires
    extended = False

    while step < total_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        data_m.update(time.time() - t_last)
        lq = batch["lq"].to(device, non_blocking=True)
        hq = batch["hq"].to(device, non_blocking=True)

        # RAFT flow (frozen, no grad). Feed LQ — same as the input pipeline sees.
        t_flow_start = time.time()
        flow_back, mask_fb = compute_flow_and_mask(
            raft, lq, threshold_px=raft_threshold, n_iter=raft_iter,
        )
        flow_m.update(time.time() - t_flow_start)

        with torch.autocast(device_type="cuda", dtype=amp_dt, enabled=use_amp):
            out = model(lq, video_hq=hq)
            losses = loss_fn(
                pred_pixels=out["pred_pixels"], hq=hq,
                R_hat=out.get("R_hat"), L_hat=out.get("L_hat"),
                flow_back=flow_back, mask=mask_fb,
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
                          model, optim, sched, ema, step + 1, best, best_ema, val_history)
                return
            step += 1
            t_last = time.time()
            continue

        if use_scaler:
            scaler.scale(total).backward()
        else:
            total.backward()

        did_step = False
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
            did_step = True

        if did_step and (step + 1) >= ema_start:
            ema.update(model.trainable_parameters())

        loss_meter.update(losses["total"].item())
        if "pix"       in losses: pix_m.update(losses["pix"].item())
        if "lpips"     in losses: lpips_m.update(losses["lpips"].item())
        if "retinex"   in losses: ret_m.update(losses["retinex"].item())
        if "L_smooth"  in losses: lsm_m.update(losses["L_smooth"].item())
        if "L_temp"    in losses: ltm_m.update(losses["L_temp"].item())
        if "R_adj"     in losses: radj_m.update(losses["R_adj"].item())
        if "dod"       in losses: dod_m.update(losses["dod"].item())
        if "pred_warp" in losses: pw_m.update(losses["pred_warp"].item())
        if "R_warp"    in losses: rw_m.update(losses["R_warp"].item())
        mask_m.update(mask_fb.float().mean().item())
        step_m.update(time.time() - t_last)
        t_last = time.time()

        if (step + 1) % log_every == 0:
            lr = optim.param_groups[0]["lr"]
            parts = [
                f"it {step+1}/{total_iters}",
                f"loss {loss_meter.avg:.4f}",
                f"pix {pix_m.avg:.4f}" if pix_m.count else "",
                f"lpips {lpips_m.avg:.4f}" if lpips_m.count else "",
                f"ret {ret_m.avg:.4f}" if ret_m.count else "",
                f"Ls {lsm_m.avg:.4f}" if lsm_m.count else "",
                f"Lt {ltm_m.avg:.4f}" if ltm_m.count else "",
                f"Radj {radj_m.avg:.4f}" if radj_m.count else "",
                f"dod {dod_m.avg:.4f}" if dod_m.count else "",
                f"pW {pw_m.avg:.4f}" if pw_m.count else "",
                f"RW {rw_m.avg:.4f}" if rw_m.count else "",
                f"mask {mask_m.avg:.3f}" if mask_m.count else "",
                f"lr {lr:.2e}",
                f"step {step_m.avg:.2f}s",
                f"flow {flow_m.avg:.2f}s",
                f"data {data_m.avg:.2f}s",
            ]
            if nan_count > 0:
                parts.append(f"nan {nan_count}")
            logger.info(" | ".join(p for p in parts if p))
            loss_meter.reset()
            pix_m.reset(); lpips_m.reset(); ret_m.reset()
            lsm_m.reset(); ltm_m.reset(); radj_m.reset(); dod_m.reset()
            pw_m.reset(); rw_m.reset(); mask_m.reset()
            flow_m.reset()

        if val_loader is not None and (step + 1) % val_every == 0:
            m = run_quick_eval(model, val_loader, device)
            val_history.append(m["psnr"])
            sliding = sum(val_history) / len(val_history)
            val_history_full.append(sliding)
            logger.info(
                f"[val @ {step+1}] live PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f} "
                f"| sliding({len(val_history)}) {sliding:.3f}"
            )
            if sliding > best:
                best = sliding
                save_ckpt(os.path.join(ckpt_dir, "best.pt"),
                          model, optim, sched, ema, step + 1, best, best_ema, val_history)
                logger.info(f"        new best (sliding) PSNR {best:.3f}")

            if (step + 1) >= ema_start:
                ema.swap_in(model.trainable_parameters())
                m_e = run_quick_eval(model, val_loader, device)
                ema.swap_out(model.trainable_parameters())
                val_history_ema.append(m_e["psnr"])
                sliding_e = sum(val_history_ema) / len(val_history_ema)
                logger.info(
                    f"[val-ema @ {step+1}] live PSNR {m_e['psnr']:.3f}  "
                    f"SSIM {m_e['ssim']:.4f} | sliding({len(val_history_ema)}) {sliding_e:.3f}"
                )
                if sliding_e > best_ema:
                    best_ema = sliding_e
                    save_ckpt(os.path.join(ckpt_dir, "best_ema.pt"),
                              model, optim, sched, ema, step + 1, best, best_ema, val_history)
                    logger.info(f"        new best-ema (sliding) PSNR {best_ema:.3f}")

            # J1 extension check at iter 9500 / 10000 etc
            if not extended and extend_to > base_iters and (step + 1) >= base_iters - val_every:
                if should_extend(val_history_full, step + 1, base_iters):
                    logger.info(
                        f"[J1] last 3 sliding vals improving at iter {step+1}; "
                        f"extending total_iters {base_iters} -> {extend_to}"
                    )
                    total_iters = extend_to
                    extended = True

        if (step + 1) % save_every == 0:
            save_ckpt(os.path.join(ckpt_dir, f"step_{step+1}.pt"),
                      model, optim, sched, ema, step + 1, best, best_ema, val_history)
            prune_step_ckpts(ckpt_dir, keep_n)

        step += 1

    save_ckpt(os.path.join(ckpt_dir, "final.pt"),
              model, optim, sched, ema, step, best, best_ema, val_history)
    prune_step_ckpts(ckpt_dir, keep_n)
    logger.info(
        f"Training done. Best (sliding) PSNR = {best:.3f} | Best EMA = {best_ema:.3f} "
        f"| NaN: {nan_count} | extended={extended}"
    )


if __name__ == "__main__":
    main()
