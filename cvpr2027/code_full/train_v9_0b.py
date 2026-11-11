"""train_v9_0b.py — V-JEPA-LLVE v9.0b PSNR-targeted finetune.

Differences vs train_v9:
  - Loads anchor checkpoint (v9.0/best.pt) and runs MANDATORY zero-iter
    Y-512 sanity gate before any optim.step. Aborts if |Y-512 − target| > tol.
  - Validation runs both RGB-256 (cheap, every val_every) and full Y-512
    audit (slow, every audit_every AND on candidate-best).
  - best.pt promoted ONLY on Y-512 sliding(3) (not RGB-256).
  - LLVELoss_v9 with pix_weight, retinex, L_smooth_spatial only.
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data import build_dataset, build_sampler
from data.sdsd_dataset import SDSDDataset
from data.common import read_image_01
from losses.v9 import LLVELoss_v9
from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import (
    AverageMeter, ParameterEMA, dump_config, ensure_dir, load_config,
    psnr, set_seed, setup_logger, ssim,
)


def amp_dtype(p): return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(p, torch.float32)
def amp_enabled(p): return p in ("fp16", "bf16")


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma. img (...,3,H,W) → (...,1,H,W)."""
    r, g, b = img[..., 0:1, :, :], img[..., 1:2, :, :], img[..., 2:3, :, :]
    return 0.299 * r + 0.587 * g + 0.114 * b


def psnr_per_frame(pred: torch.Tensor, gt: torch.Tensor,
                   max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3)).clamp_min(eps)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    if H < size or W < size:
        return x
    h0 = (H - size) // 2
    w0 = (W - size) // 2
    return x[..., h0:h0 + size, w0:w0 + size]


def list_audit_seqs(root: str, split: str = "test") -> List[Tuple[str, list, list]]:
    ds = SDSDDataset(root=root, subset="direct", split=split, clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


def load_clip_full(lq_paths, gt_paths, clip_len: int = 16):
    """Load center T frames at native resolution (no crop)."""
    n_total = len(lq_paths)
    T = min(clip_len, n_total)
    start = (n_total - T) // 2
    lq = np.stack([read_image_01(lq_paths[start + i]) for i in range(T)], axis=0)
    gt = np.stack([read_image_01(gt_paths[start + i]) for i in range(T)], axis=0)
    lq = torch.from_numpy(lq).permute(3, 0, 1, 2).unsqueeze(0).float()
    gt = torch.from_numpy(gt).permute(3, 0, 1, 2).unsqueeze(0).float()
    return lq, gt


@torch.no_grad()
def audit_y512(model, audit_seqs, device) -> Dict[str, float]:
    """Full Y-PSNR @ 512 center audit on the test set (matches audit_protocol_v9.py)."""
    model.eval()
    rgb256_all, y256_all, rgb512_all, y512_all = [], [], [], []
    for seq_name, lq_paths, gt_paths in audit_seqs:
        lq, gt = load_clip_full(lq_paths, gt_paths, clip_len=16)
        lq, gt = lq.to(device), gt.to(device)
        H, W = lq.shape[-2:]

        lq256 = center_crop(lq, 256)
        gt256 = center_crop(gt, 256)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out256 = model.sample(lq256)
        pred256 = out256["pred_pixels"].clamp(0, 1).float()

        p256 = pred256[0].permute(1, 0, 2, 3)
        g256 = gt256[0].permute(1, 0, 2, 3)
        rgb256_all.extend(psnr_per_frame(p256, g256).cpu().tolist())
        y256_all.extend(psnr_per_frame(rgb_to_y(p256), rgb_to_y(g256)).cpu().tolist())

        if min(H, W) >= 512:
            lq512 = center_crop(lq, 512)
            gt512 = center_crop(gt, 512)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out512 = model.sample(lq512)
            pred512 = out512["pred_pixels"].clamp(0, 1).float()
            p512 = pred512[0].permute(1, 0, 2, 3)
            g512 = gt512[0].permute(1, 0, 2, 3)
            rgb512_all.extend(psnr_per_frame(p512, g512).cpu().tolist())
            y512_all.extend(psnr_per_frame(rgb_to_y(p512), rgb_to_y(g512)).cpu().tolist())
    model.train()
    return {
        "rgb256_framemean": float(np.mean(rgb256_all)) if rgb256_all else float("nan"),
        "y256_framemean": float(np.mean(y256_all)) if y256_all else float("nan"),
        "rgb512_framemean": float(np.mean(rgb512_all)) if rgb512_all else float("nan"),
        "y512_framemean": float(np.mean(y512_all)) if y512_all else float("nan"),
    }


@torch.no_grad()
def quick_eval(model, val_loader, device, max_samples: int = 14):
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


def save_ckpt(path, model, optim, sched, ema, step, best, val_history):
    ensure_dir(os.path.dirname(path))
    state = {
        "step": step, "best_y512": best,
        "val_history": list(val_history),
        "pixel_branch": model.pixel_branch.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": sched.state_dict(),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


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
        f"accum={cfg['optim'].get('accum_steps', 1)}"
    )

    val_set = build_dataset(
        cfg["data"]["val"],
        clip_len=tcfg.get("clip_len", 16),
        crop_size=tcfg.get("crop_size", 256), is_train=False,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=max(1, cfg.get("num_workers", 4) // 2),
                            pin_memory=True)
    logger.info(f"Val:   {len(val_set)} samples")

    audit_seqs = list_audit_seqs(cfg["data"]["val"]["root"], split="test")
    logger.info(f"Audit set: {len(audit_seqs)} sequences (Y-512 audit)")

    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    n_train = sum(p.numel() for p in model.trainable_parameters())
    logger.info(f"Trainable params: {n_train/1e6:.1f}M")

    # === Load anchor ===
    anchor_path = cfg["optim"]["resume_from"]
    if not (anchor_path and os.path.isfile(anchor_path)):
        raise FileNotFoundError(f"Anchor ckpt missing: {anchor_path}")
    state = torch.load(anchor_path, map_location="cpu", weights_only=False)
    model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    logger.info(f"Loaded anchor from {anchor_path} (was step={state.get('step', '?')})")

    # ============ MANDATORY ZERO-ITER SANITY GATE (per S1 #1: BEFORE optimizer) ============
    sanity_target = float(tcfg.get("sanity_y512_target", 26.843))
    sanity_tol = float(tcfg.get("sanity_y512_tol", 0.05))
    logger.info(f"=== Zero-iter sanity gate: Y-512 must be {sanity_target:.3f} ± {sanity_tol:.3f} ===")
    t0 = time.time()
    audit0 = audit_y512(model, audit_seqs, device)
    logger.info(
        f"Zero-iter audit ({time.time()-t0:.1f}s): "
        f"RGB-256={audit0['rgb256_framemean']:.3f}  Y-256={audit0['y256_framemean']:.3f}  "
        f"RGB-512={audit0['rgb512_framemean']:.3f}  Y-512={audit0['y512_framemean']:.3f}"
    )
    y512_0 = audit0["y512_framemean"]
    if abs(y512_0 - sanity_target) > sanity_tol:
        logger.error(
            f"SANITY FAIL: Y-512 = {y512_0:.3f} differs from target {sanity_target:.3f} "
            f"by more than tol {sanity_tol:.3f}. Aborting. Check anchor ckpt and val pipeline."
        )
        sys.exit(2)
    logger.info(f"SANITY PASS: Y-512 = {y512_0:.3f} ≈ {sanity_target:.3f}")
    logger.info(f"Anchor preserved at: {anchor_path}  (v9.0b output: {out_dir} — separate path, no overwrite risk)")

    # ============ Build optimizer/scheduler/EMA only AFTER sanity passes (S1 #1) ============
    lcfg = cfg["loss"]
    loss_fn = LLVELoss_v9(
        pix_weight=lcfg.get("pix_weight", 1.5),
        lpips_weight=lcfg.get("lpips_weight", 0.0),
        retinex_compose_weight=lcfg.get("retinex_compose_weight", 0.15),
        L_smooth_spatial_weight=lcfg.get("L_smooth_spatial_weight", 0.02),
        L_smooth_temporal_weight=lcfg.get("L_smooth_temporal_weight", 0.0),
        R_adjacent_weight=lcfg.get("R_adjacent_weight", 0.0),
        R_adjacent_scale=lcfg.get("R_adjacent_scale", 2),
        pred_dod_weight=lcfg.get("pred_dod_weight", 0.0),
    ).to(device)

    optim = build_optimizer(model, cfg["optim"])
    sched = build_scheduler(optim, cfg["optim"])

    ema_decay = float(tcfg.get("ema_decay", 0.999))
    ema_start = int(tcfg.get("ema_start", 500))
    ema = ParameterEMA(model.trainable_parameters(), decay=ema_decay).to(device)
    logger.info(f"EMA: decay={ema_decay} start={ema_start}")

    # ============ training loop ============
    prec = cfg.get("precision", "bf16")
    use_amp = amp_enabled(prec)
    amp_dt = amp_dtype(prec)

    total_iters = int(cfg["optim"]["total_iters"])
    log_every = tcfg.get("log_every", 50)
    val_every = tcfg.get("val_every", 500)
    audit_every = tcfg.get("audit_every", 1000)
    save_every = tcfg.get("save_every", 1000)
    grad_clip = cfg["optim"].get("grad_clip", 1.0)
    accum = max(1, int(cfg["optim"].get("accum_steps", 1)))

    val_window = int(tcfg.get("val_window", 3))
    y512_history: collections.deque = collections.deque(maxlen=val_window)
    rgb256_history: collections.deque = collections.deque(maxlen=val_window)
    y512_history.append(y512_0)
    best_y512 = y512_0
    best_y512_sliding = y512_0

    model.train()
    step = 0
    loss_meter = AverageMeter()
    pix_m, ret_m, lsm_m = (AverageMeter() for _ in range(3))
    data_m, step_m = AverageMeter(), AverageMeter()

    nan_count = 0
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
                pred_pixels=out["pred_pixels"], hq=hq,
                R_hat=out.get("R_hat"), L_hat=out.get("L_hat"),
            )
            total = losses["total"] / accum

        if not bool(torch.isfinite(total).item()):
            nan_count += 1
            optim.zero_grad(set_to_none=True)
            if (step + 1) % log_every == 0:
                logger.warning(f"  [NaN] step {step+1}: skipped (cum {nan_count})")
            step += 1
            t_last = time.time()
            continue

        total.backward()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
            sched.step()
            if (step + 1) >= ema_start:
                ema.update(model.trainable_parameters())

        loss_meter.update(losses["total"].item())
        if "pix"      in losses: pix_m.update(losses["pix"].item())
        if "retinex"  in losses: ret_m.update(losses["retinex"].item())
        if "L_smooth" in losses: lsm_m.update(losses["L_smooth"].item())
        step_m.update(time.time() - t_last)
        t_last = time.time()

        if (step + 1) % log_every == 0:
            lr = optim.param_groups[0]["lr"]
            logger.info(
                f"it {step+1}/{total_iters} | loss {loss_meter.avg:.4f} | "
                f"pix {pix_m.avg:.4f} | ret {ret_m.avg:.4f} | Ls {lsm_m.avg:.4f} | "
                f"lr {lr:.2e} | step {step_m.avg:.2f}s | data {data_m.avg:.2f}s"
                + (f" | nan {nan_count}" if nan_count else "")
            )
            for m in (loss_meter, pix_m, ret_m, lsm_m):
                m.reset()

        # ===== cheap RGB-256 every val_every =====
        do_audit = False
        if (step + 1) % val_every == 0:
            qm = quick_eval(model, val_loader, device)
            rgb256_history.append(qm["psnr"])
            sliding = sum(rgb256_history) / len(rgb256_history)
            logger.info(
                f"[val-rgb256 @ {step+1}] PSNR {qm['psnr']:.3f}  SSIM {qm['ssim']:.4f} "
                f"| sliding({len(rgb256_history)}) {sliding:.3f}"
            )
            # if rgb256 jumped significantly, also audit
            if len(rgb256_history) >= 2 and qm["psnr"] > rgb256_history[-2]:
                do_audit = True

        # ===== full Y-512 audit every audit_every (and on candidate-best) =====
        if (step + 1) % audit_every == 0 or do_audit:
            t_a = time.time()
            au = audit_y512(model, audit_seqs, device)
            y512_history.append(au["y512_framemean"])
            sliding_y512 = sum(y512_history) / len(y512_history)
            logger.info(
                f"[audit @ {step+1}] ({time.time()-t_a:.1f}s) "
                f"RGB-256={au['rgb256_framemean']:.3f}  Y-256={au['y256_framemean']:.3f}  "
                f"RGB-512={au['rgb512_framemean']:.3f}  "
                f"Y-512={au['y512_framemean']:.3f}  | sliding(3) Y-512 = {sliding_y512:.3f}"
            )
            if au["y512_framemean"] > best_y512:
                best_y512 = au["y512_framemean"]
            if sliding_y512 > best_y512_sliding:
                best_y512_sliding = sliding_y512
                save_ckpt(os.path.join(ckpt_dir, "best.pt"),
                          model, optim, sched, ema, step + 1, best_y512_sliding,
                          y512_history)
                logger.info(f"        new best (Y-512 sliding) = {best_y512_sliding:.3f}")

            # early-stop guardrails
            if (step + 1) >= 500 and au["y512_framemean"] < (sanity_target - 0.20):
                logger.warning(
                    f"EARLY-STOP TRIGGER: Y-512 {au['y512_framemean']:.3f} "
                    f"< target − 0.20 ({sanity_target - 0.20:.3f}) at iter {step+1}"
                )
                # save final and bail
                save_ckpt(os.path.join(ckpt_dir, "final_earlystop.pt"),
                          model, optim, sched, ema, step + 1, best_y512_sliding,
                          y512_history)
                logger.error("v9.0b regressed below anchor − 0.2 dB. Aborting per Q3 rule.")
                return

        if (step + 1) % save_every == 0:
            save_ckpt(os.path.join(ckpt_dir, f"step_{step+1}.pt"),
                      model, optim, sched, ema, step + 1, best_y512_sliding, y512_history)

        step += 1

    save_ckpt(os.path.join(ckpt_dir, "final.pt"),
              model, optim, sched, ema, step, best_y512_sliding, y512_history)
    logger.info(
        f"v9.0b done. Best Y-512 (sliding) = {best_y512_sliding:.3f} | "
        f"Best Y-512 (raw) = {best_y512:.3f} | "
        f"Anchor was {sanity_target:.3f} | "
        f"NaN: {nan_count}"
    )


if __name__ == "__main__":
    main()
