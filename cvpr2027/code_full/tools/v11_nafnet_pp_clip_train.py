"""NAFNet++ with CLIP teacher (Exp A control).

Identical to v11_nafnet_pp_train.py EXCEPT:
  - REPA teacher = CLIP-B/32 layer 8 (vs V-JEPA layer 3 in original)

Tests whether NAFNet++ 20.708 success comes from V-JEPA teacher or
from training tricks (EMA + LPIPS + warmup + 30k iter).
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from tools.iqa_metrics import compute_iqa, fmt_iqa, save_compare_video
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr,
                                           charbonnier, build_split_seqs)
from tools.v11_nafnet_train import NAFNet3D, safe_cos_loss
from tools.v11_nafnet_pp_train import EMA, lr_warmup_cosine
from tools.v11_c2_repa_train_clip import (encode_clip, REPAProjectorCLIP,
                                           CLIP_DIM, CLIP_GRID)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import CLIPVisionModel
    clip_t = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir="/cache/zihaohe/hf_cache",
    ).to(device).eval()
    for p in clip_t.parameters(): p.requires_grad_(False)

    net = NAFNet3D(base=args.base,
                   enc_blocks=tuple(args.enc_blocks),
                   dec_blocks=tuple(args.dec_blocks),
                   bottleneck_blocks=args.bn_blocks).to(device)
    n_n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    projector = REPAProjectorCLIP(in_ch=net.bottleneck_c, out_dim=CLIP_DIM).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    lpips_model = None
    if args.lambda_lpips > 0:
        import pyiqa
        lpips_model = pyiqa.create_metric("lpips", device=device, as_loss=True).eval()
        for p in lpips_model.parameters(): p.requires_grad_(False)

    print(f"[NAFNet++ CLIP] CLIP-B/32 87M frozen (REPA target only)")
    print(f"                NAFNet3D: {n_n/1e6:.2f}M trainable (base={args.base})")
    print(f"                REPAProjectorCLIP: {n_p/1e6:.2f}M (training only)")
    print(f"                Loss: Charbonnier + {args.lambda_repa}·REPA(CLIP-{args.clip_layer}) + {args.lambda_lpips}·LPIPS")
    print(f"                LR: warmup {args.warmup_iters} + cosine, base={args.lr}")
    print(f"                EMA decay={args.ema_decay}, HASTE off @ iter {args.early_stop_iter}")

    train_params = list(net.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    ema = None
    log = []
    if args.eval_only:
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        net.load_state_dict(ckpt["net"])
        if "ema_shadow" in ckpt:
            ema = EMA(net, decay=args.ema_decay)
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
        if "projector" in ckpt:
            projector.load_state_dict(ckpt["projector"])
        print(f"[eval-only] loaded ckpt → {args.out_dir}/ckpt.pt")
    else:
        ema = EMA(net, decay=args.ema_decay)
        t0 = time.time()
        for it in range(args.iters):
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
                z_tea = encode_clip(clip_t, lq, args.clip_layer).float()
                z_stu = projector(h_b.float(), CLIP_GRID)
                m = z_tea.mean(dim=1, keepdim=True); s = z_tea.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_tea - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L_lpips = torch.zeros((), device=device)
            if args.lambda_lpips > 0 and lpips_model is not None:
                B, C, T, H, W = pred.shape
                idx_lp = torch.linspace(0, T - 1, 4).long().to(device)
                pred_lp = pred.float().permute(0, 2, 1, 3, 4).reshape(B, T, C, H, W)[:, idx_lp].reshape(-1, C, H, W).clamp(0, 1)
                gt_lp = gt.permute(0, 2, 1, 3, 4).reshape(B, T, C, H, W)[:, idx_lp].reshape(-1, C, H, W).clamp(0, 1)
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
                    "args": vars(args), "log": log, "stage": "nafnet_pp_clip"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    print(f"\n=== EVAL: NAFNet++ CLIP teacher (EMA weights) ===")
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
    md = os.path.join(args.out_dir, "NAFNETPP_CLIP_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# NAFNet++ CLIP teacher (Exp A control)\n\n")
        f.write(f"NAFNet3D {n_n/1e6:.2f}M; base={args.base}; CLIP-B/32 layer {args.clip_layer} teacher\n")
        f.write(f"Training: warmup {args.warmup_iters} + cosine, EMA={args.ema_decay}, "
                f"λ_LPIPS={args.lambda_lpips}, iters={args.iters}\n\n")
        f.write(f"## SDSD held-out (EMA weights)\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}\n\n")
        f.write(f"vs NAFNet++ V-JEPA 20.708 / 0.2608 LPIPS\n")
    print(f"\nNAFNet++ CLIP PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip_layer", type=int, default=8)
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
    ap.add_argument("--save_viz", action="store_true")
    ap.add_argument("--viz_dir", default=None)
    main(ap.parse_args())
