"""Residual-Flow trainer: single-pass V-JEPA latent residual prediction.

Architecture (vjlf cousin, no flow ODE):
  V-JEPA encoder (frozen) → z_LQ, z_HQ  (B, 2048, 1024)
  ResHead = VJepaLatentDiT(z_LQ, t=0) → predicts (z_HQ - z_LQ) directly
  Render = DeltaModulatedDecoder, kv=[z_HQ_pred] at eval, [z_HQ] at train (oracle)
  REPA + HASTE on render bottleneck → V-JEPA(GT) layer 3

Training: L = MSE(ResHead(z_LQ), z_HQ - z_LQ) + λ_pixel·Charbonnier(pred, GT)
              + λ_repa·cos(h_b, V-JEPA(GT)).

Inference: z_HQ_pred = z_LQ + ResHead(z_LQ)  ← 1 forward pass, NO ODE.

Why: vjlf/vjpf use 4-step Euler ODE which needs 4× compute and risks
divergence; residual prediction is one-shot and matches "z_LQ + delta = z_HQ"
literally.
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.latent_dit import VJepaLatentDiT
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.iqa_metrics import compute_iqa, fmt_iqa
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


class REPAProjector(nn.Module):
    def __init__(self, in_ch=256, out_dim=1024):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)
    def forward(self, h_b, vjepa_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=vjepa_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


def safe_cos_loss(z_stu, z_tea):
    z_stu = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
    z_tea = F.normalize(z_tea.float(), dim=-1, eps=1e-3)
    cos = (z_stu * z_tea).sum(dim=-1).clamp(-1.5, 1.5)
    return -cos.mean()


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    # Reuse VJepaLatentDiT but always pass t=0 → time embed degenerates to a
    # constant; model becomes a pure residual transformer over z_LQ.
    res_head = VJepaLatentDiT(
        vjepa_dim=1024, grid=(8, 16, 16),
        dim=args.dit_dim, depth=args.dit_depth, num_heads=args.dit_heads,
        ffn_mult=4.0, use_gradient_checkpoint=True,
    ).to(device)
    n_h = sum(p.numel() for p in res_head.parameters() if p.requires_grad)

    render = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_r = sum(p.numel() for p in render.parameters() if p.requires_grad)

    projector = REPAProjector(in_ch=render.dims[-1], out_dim=render.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    print(f"[ResFlow] V-JEPA encoder 304M frozen")
    print(f"          ResHead: {n_h/1e6:.2f}M trainable (DiT body, used time-free)")
    print(f"          Render: {n_r/1e6:.2f}M, Projector: {n_p/1e6:.2f}M")
    print(f"          Total trainable: {(n_h+n_r+n_p)/1e6:.2f}M")
    print(f"          Loss: MSE(residual) + {args.lambda_pixel}·Charbonnier "
          f"+ {args.lambda_repa}·REPA (HASTE early-stop @ iter {args.early_stop_iter})")
    print(f"          Inference: 1-step (z_HQ_pred = z_LQ + ResHead(z_LQ))")

    train_params = list(res_head.parameters()) + list(render.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        res_head.load_state_dict(ckpt["res_head"])
        render.load_state_dict(ckpt["render"])
        if "projector" in ckpt:
            projector.load_state_dict(ckpt["projector"])
        print(f"[eval-only] loaded ckpt → {args.out_dir}/ckpt.pt")
    else:
        t0 = time.time()
        for it in range(args.iters):
            idx = int(rng.integers(0, len(train_seqs)))
            _, lq_p, gt_p = train_seqs[idx]
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)

            with torch.no_grad():
                z_LQ = encode(backbone, lq)[args.ref_layer].float()
                z_HQ = encode(backbone, gt)[args.ref_layer].float()

            B = z_LQ.shape[0]
            t_zero = torch.zeros(B, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                residual_pred = res_head(z_LQ, t_zero)
            L_res = F.mse_loss(residual_pred.float(), z_HQ - z_LQ)

            kv_layers = [z_HQ]
            h_b, skips = render.encode(lq, kv_layers=kv_layers, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = render.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)
            L_pixel = charbonnier(pred.float(), gt)

            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                z_stu = projector(h_b, tuple(render.vjepa_grid))
                m = z_HQ.mean(dim=1, keepdim=True); s = z_HQ.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_HQ - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L = L_res + args.lambda_pixel * L_pixel + args.lambda_repa * L_repa

            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} "
                      f"L_res={L_res.item():.4f} L_pixel={L_pixel.item():.4f} "
                      f"L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(),
                            "L_res": L_res.item(), "L_pixel": L_pixel.item(),
                            "L_repa": L_repa.item()})
                t0 = time.time()

        torch.save({"res_head": res_head.state_dict(), "render": render.state_dict(),
                    "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "resflow"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    print(f"\n=== FINAL EVAL: ResFlow (1-step residual) ===")
    res_head.eval(); render.eval()
    psnrs_p, psnrs_o = [], []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_LQ = encode(backbone, lq)[args.ref_layer].float()
            z_HQ = encode(backbone, gt)[args.ref_layer].float()
            B = z_LQ.shape[0]
            t_zero = torch.zeros(B, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                residual = res_head(z_LQ, t_zero).float()
            z_HQ_pred = z_LQ + residual

            h_b, skips = render.encode(lq, kv_layers=[z_HQ_pred], delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ_pred], delta_layers=None)
            pred_f = pred.float()
            p = mean_psnr(to_u8(pred_f), to_u8(gt))
            iqa = compute_iqa(pred_f, gt, device=device)
            psnrs_p.append(p)
            for k in iqa_acc: iqa_acc[k].append(iqa[k])

            h_b, skips = render.encode(lq, kv_layers=[z_HQ], delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_o = render.decode(h_b, skips, lq, kv_layers=[z_HQ], delta_layers=None)
            psnrs_o.append(mean_psnr(to_u8(pred_o.float()), to_u8(gt)))
            print(f"    [{name}] Pred={p:.3f} {fmt_iqa(iqa)}  oracle={psnrs_o[-1]:.3f}")

    psnr_p, psnr_o = float(np.mean(psnrs_p)), float(np.mean(psnrs_o))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "RESFLOW_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# ResFlow: 1-step V-JEPA latent residual prediction\n\n")
        f.write(f"ResHead {n_h/1e6:.2f}M + Render {n_r/1e6:.2f}M + Proj {n_p/1e6:.2f}M trainable\n")
        f.write(f"V-JEPA encoder 304M frozen; inference = single forward pass\n\n")
        f.write(f"## SDSD held-out (residual pipeline)\n\n")
        f.write(f"- PSNR    = {psnr_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"## Oracle-render PSNR = {psnr_o:.3f}\n\n")
        f.write(f"vs vjlf 19.0, vjpf 19.15, clip_30k 20.746, v9.0b 20.578\n")
        f.write(f"\noracle - residual gap = {psnr_o - psnr_p:+.3f}\n")
    print(f"\nResFlow PSNR={psnr_p:.3f} {fmt_iqa(mean_iqa)}  oracle={psnr_o:.3f}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lambda_pixel", type=float, default=1.0)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--early_stop_iter", type=int, default=9000)
    ap.add_argument("--dit_dim", type=int, default=512)
    ap.add_argument("--dit_depth", type=int, default=12)
    ap.add_argument("--dit_heads", type=int, default=8)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    main(ap.parse_args())
