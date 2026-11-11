"""NAFNet++: NAFNet + EMA + warmup + LPIPS perceptual loss + 30k iter.

Targets: incremental training-strategy improvements on the NAFNet 15k = 20.436
baseline. Pure training tricks, no architectural change.

Additions vs v11_nafnet_train.py:
  + EMA (exponential moving average) on weights, decay=0.999, used at eval
  + Linear warmup 500 iter → cosine annealing
  + LPIPS perceptual loss weight 0.1 (improves perceptual metrics)
  + 30k iter (2x training)
"""
from __future__ import annotations
import argparse, copy, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from models.vjepa_backbone import build_vjepa_backbone
from tools.iqa_metrics import compute_iqa, fmt_iqa, save_compare_video
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)
# Reuse NAFNet from existing trainer
from tools.v11_nafnet_train import NAFNet3D, REPAProjector, safe_cos_loss, VJEPA_GRID


class EMA:
    """Standard exponential moving average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def apply_to(self, model):
        """Returns saved-state-dict for restore; applies EMA weights to model."""
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return original

    def restore(self, model, original):
        model.load_state_dict(original)


def lr_warmup_cosine(step, warmup, total, base_lr, min_lr_frac=0.05):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    net = NAFNet3D(base=args.base,
                   enc_blocks=tuple(args.enc_blocks),
                   dec_blocks=tuple(args.dec_blocks),
                   bottleneck_blocks=args.bn_blocks).to(device)
    n_n = sum(p.numel() for p in net.parameters() if p.requires_grad)

    projector = REPAProjector(in_ch=net.bottleneck_c, out_dim=1024).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    # LPIPS loss module (cached, GPU-resident)
    lpips_model = None
    if args.lambda_lpips > 0:
        import pyiqa
        lpips_model = pyiqa.create_metric("lpips", device=device, as_loss=True).eval()
        for p in lpips_model.parameters(): p.requires_grad_(False)

    print(f"[NAFNet++] V-JEPA encoder 304M frozen (REPA target only)")
    print(f"           NAFNet3D: {n_n/1e6:.2f}M trainable (base={args.base})")
    print(f"           REPAProjector: {n_p/1e6:.2f}M (training only)")
    print(f"           Loss: Charbonnier + {args.lambda_repa}·REPA + {args.lambda_lpips}·LPIPS")
    print(f"           LR: warmup {args.warmup_iters} + cosine, base lr={args.lr}")
    print(f"           EMA decay={args.ema_decay}, HASTE off @ iter {args.early_stop_iter}")

    train_params = list(net.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    ema = None
    log = []
    if args.eval_only:
        ckpt_d = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        net.load_state_dict(ckpt_d["net"])
        if "ema_shadow" in ckpt_d:
            ema = EMA(net, decay=args.ema_decay)
            ema.shadow = {k: v.to(device) for k, v in ckpt_d["ema_shadow"].items()}
        if "projector" in ckpt_d:
            projector.load_state_dict(ckpt_d["projector"])
        print(f"[eval-only] loaded ckpt → {args.out_dir}/ckpt.pt")
    else:
        ema = EMA(net, decay=args.ema_decay)
        t0 = time.time()
        for it in range(args.iters):
            # LR
            cur_lr = lr_warmup_cosine(it, args.warmup_iters, args.iters, args.lr)
            for g in optim.param_groups: g["lr"] = cur_lr

            idx = int(rng.integers(0, len(train_seqs)))
            _, lq_p, gt_p = train_seqs[idx]
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, h_b = net(lq)
            L_rgb = charbonnier(pred.float(), gt)

            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                with torch.no_grad():
                    z_L = encode(backbone, lq)[args.ref_layer].float()
                z_stu = projector(h_b.float(), VJEPA_GRID)
                m = z_L.mean(dim=1, keepdim=True); s = z_L.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_L - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L_lpips = torch.zeros((), device=device)
            if args.lambda_lpips > 0 and lpips_model is not None:
                # LPIPS expects (N, 3, H, W) in [-1, 1]
                B, C, T, H, W = pred.shape
                pred_lp = pred.float().permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).clamp(0, 1)
                gt_lp   = gt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).clamp(0, 1)
                # Subsample 4 frames to save memory
                idx_lp = torch.linspace(0, T - 1, 4).long().to(device)
                pred_lp = pred_lp.reshape(B, T, C, H, W)[:, idx_lp].reshape(-1, C, H, W)
                gt_lp = gt_lp.reshape(B, T, C, H, W)[:, idx_lp].reshape(-1, C, H, W)
                L_lpips = lpips_model(pred_lp, gt_lp).mean()

            L = L_rgb + args.lambda_repa * L_repa + args.lambda_lpips * L_lpips
            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step()
            ema.update(net)

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} lr={cur_lr:.2e} L={L.item():.4f} "
                      f"L_rgb={L_rgb.item():.4f} L_repa={L_repa.item():.4f} "
                      f"L_lpips={L_lpips.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "lr": cur_lr, "L": L.item(),
                            "L_rgb": L_rgb.item(), "L_repa": L_repa.item(),
                            "L_lpips": L_lpips.item()})
                t0 = time.time()

        torch.save({"net": net.state_dict(), "projector": projector.state_dict(),
                    "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
                    "args": vars(args), "log": log, "stage": "nafnet_pp"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Eval with EMA weights
    print(f"\n=== EVAL: NAFNet++ (EMA weights) ===")
    backup = ema.apply_to(net) if ema is not None else None
    net.eval()
    psnrs = []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, _ = net(lq)
            pred_f = pred.float()
            p = mean_psnr(to_u8(pred_f), to_u8(gt))
            iqa = compute_iqa(pred_f, gt, device=device)
            psnrs.append(p)
            for k in iqa_acc: iqa_acc[k].append(iqa[k])
            print(f"    [{name}] PSNR={p:.3f} {fmt_iqa(iqa)}")
            if args.save_viz:
                viz_root = args.viz_dir if args.viz_dir else os.path.join(args.out_dir, "viz")
                save_compare_video(os.path.join(viz_root, f"{name}.mp4"),
                                   lq, pred_f, gt, fps=10)

    if ema is not None:
        ema.restore(net, backup)

    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "NAFNETPP_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# NAFNet++ (EMA + warmup + LPIPS loss + 30k iter)\n\n")
        f.write(f"NAFNet3D {n_n/1e6:.2f}M; base={args.base}\n")
        f.write(f"Training: warmup {args.warmup_iters} + cosine, EMA={args.ema_decay}, "
                f"λ_LPIPS={args.lambda_lpips}, iters={args.iters}\n\n")
        f.write(f"## SDSD held-out (EMA weights)\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"vs nafnet_15k 20.436 / LPIPS 0.2642, vs clip_30k 20.746 / LPIPS 0.3035\n")
    print(f"\nNAFNet++ PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--lambda_lpips", type=float, default=0.1)
    ap.add_argument("--early_stop_iter", type=int, default=18000)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--enc_blocks", type=int, nargs="+", default=[2, 2, 4])
    ap.add_argument("--dec_blocks", type=int, nargs="+", default=[2, 2, 2])
    ap.add_argument("--bn_blocks", type=int, default=4)
    ap.add_argument("--warmup_iters", type=int, default=500)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--save_viz", action="store_true",
                    help="Save MP4 LQ|Pred|GT comparison videos during eval")
    ap.add_argument("--viz_dir", default=None,
                    help="Override viz output dir (default: out_dir/viz/)")
    main(ap.parse_args())
