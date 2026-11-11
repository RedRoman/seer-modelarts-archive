"""VJLF v1: V-JEPA Latent Flow trainer with REPA + HASTE.

Architecture:
  V-JEPA encoder (frozen 304M)
    LQ ─► z_LQ (B, 2048, 1024)
    GT ─► z_HQ (B, 2048, 1024)

  VJepaLatentDiT (~80M trainable, depth=12, dim=512)
    rectified flow training:
      sample t~U(0,1)
      z_t = (1-t)·z_LQ + t·z_HQ
      v_target = z_HQ - z_LQ
      v_pred = DiT(z_t, t)
      L_flow = MSE(v_pred, v_target)
    inference:
      z_HQ_pred = Euler ODE solve from z_LQ, num_steps=4

  DeltaModulatedDecoder (33M trainable) — the "render" head
    train: kv=[z_HQ] (oracle) → pred
    eval:  kv=[z_HQ_pred from DiT] → pred
    REPA: align render's bottleneck h_b to V-JEPA layer 3 of GT
          (same as E2b; PROVEN +0.79 dB recipe)
    HASTE: REPA off after iter 900

Loss = L_flow + λ_pixel · L_pixel + λ_repa · L_repa  (HASTE-gated)

Total trainable: ~115M (DiT 80 + render 33 + REPA projector 7).
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
    """E2b's proven REPA projector: 3D conv -> spatial-temporal norm."""
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

    dit = VJepaLatentDiT(
        vjepa_dim=1024, grid=(8, 16, 16),
        dim=args.dit_dim, depth=args.dit_depth, num_heads=args.dit_heads,
        ffn_mult=4.0, use_gradient_checkpoint=True,
    ).to(device)
    n_dit = sum(p.numel() for p in dit.parameters() if p.requires_grad)

    render = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_r = sum(p.numel() for p in render.parameters() if p.requires_grad)

    projector = REPAProjector(in_ch=render.dims[-1], out_dim=render.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    print(f"[VJLF v1] V-JEPA encoder 304M frozen")
    print(f"       DiT (rectified flow): {n_dit/1e6:.2f}M trainable "
          f"(dim={args.dit_dim} depth={args.dit_depth} heads={args.dit_heads})")
    print(f"       Render (33M family decoder): {n_r/1e6:.2f}M trainable")
    print(f"       REPA projector: {n_p/1e6:.2f}M trainable")
    print(f"       Total trainable: {(n_dit+n_r+n_p)/1e6:.2f}M")
    print(f"       Loss: L_flow + {args.lambda_pixel}·L_pixel + "
          f"{args.lambda_repa}·L_repa (HASTE early-stop @ iter {args.early_stop_iter})")
    print(f"       Inference: ODE solver {args.eval_steps} Euler steps")

    train_params = list(dit.parameters()) + list(render.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        dit.load_state_dict(ckpt["dit"])
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

            # 1) V-JEPA encode (frozen)
            with torch.no_grad():
                z_LQ = encode(backbone, lq)[args.ref_layer].float()    # (B, N, 1024)
                z_HQ = encode(backbone, gt)[args.ref_layer].float()

            B = z_LQ.shape[0]

            # 2) Rectified flow: sample t, build z_t and v_target
            t_sample = torch.rand(B, device=device)
            t_view = t_sample.view(B, 1, 1)
            z_t = (1.0 - t_view) * z_LQ + t_view * z_HQ
            v_target = z_HQ - z_LQ

            # 3) DiT predicts velocity
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_pred = dit(z_t, t_sample)
            L_flow = F.mse_loss(v_pred.float(), v_target)

            # 4) Render: ORACLE z_HQ as KV (decoupled training)
            kv_layers = [z_HQ]
            h_b, skips = render.encode(lq, kv_layers=kv_layers, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = render.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)
            L_pixel = charbonnier(pred.float(), gt)

            # 5) REPA on render bottleneck → V-JEPA layer 3 of GT (HASTE-gated)
            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                z_stu = projector(h_b, tuple(render.vjepa_grid))
                m = z_HQ.mean(dim=1, keepdim=True); s = z_HQ.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_HQ - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L = L_flow + args.lambda_pixel * L_pixel + args.lambda_repa * L_repa

            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} "
                      f"L_flow={L_flow.item():.4f} L_pixel={L_pixel.item():.4f} "
                      f"L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(),
                            "L_flow": L_flow.item(), "L_pixel": L_pixel.item(),
                            "L_repa": L_repa.item()})
                t0 = time.time()

            # Mid eval at iter 500: report TWO numbers (oracle render vs DiT-piped render)
            if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
                psnr_dit, psnr_oracle, _ = _quick_eval(dit, render, backbone, test_seqs[:3],
                                                       device, args.crop, args.ref_layer,
                                                       args.eval_steps, with_iqa=False)
                print(f"  [@iter {it+1}] DiT-pipe PSNR(3 seqs)={psnr_dit:.3f}  oracle={psnr_oracle:.3f}")

        torch.save({"dit": dit.state_dict(), "render": render.state_dict(),
                    "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "vjlf"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Final eval: full pipeline via DiT ODE + render
    print(f"\n=== FINAL EVAL: VJLF (DiT {args.eval_steps}-step ODE → render) ===")
    psnr_dit, psnr_oracle, mean_iqa = _quick_eval(dit, render, backbone, test_seqs,
                                                   device, args.crop, args.ref_layer,
                                                   args.eval_steps, verbose=True)
    md = os.path.join(args.out_dir, "VJLF_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# VJLF: V-JEPA Latent Flow\n\n")
        f.write(f"DiT {n_dit/1e6:.2f}M + Render {n_r/1e6:.2f}M trainable\n")
        f.write(f"V-JEPA encoder 304M frozen\n")
        f.write(f"rectified flow + render-on-oracle decoupled training\n\n")
        f.write(f"## SDSD held-out (DiT pipeline)\n\n")
        f.write(f"- PSNR    = {psnr_dit:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"## Oracle-render PSNR = {psnr_oracle:.3f} (z_HQ ground truth as KV)\n\n")
        f.write(f"vs E2b 19.556, vs PATH P 19.724, vs pure_rgb 18.533\n")
        f.write(f"\noracle - DiT gap = {psnr_oracle - psnr_dit:+.3f}\n")
    print(f"\nVJLF DiT PSNR={psnr_dit:.3f} {fmt_iqa(mean_iqa)}  oracle={psnr_oracle:.3f}")
    print(f"wrote {md}")


@torch.no_grad()
def _quick_eval(dit, render, backbone, test_seqs, device, crop, ref_layer,
                num_steps, verbose=False, with_iqa=True):
    dit.eval(); render.eval()
    psnrs_dit, psnrs_oracle = [], []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []} if with_iqa else None
    for name, lq_p, gt_p in test_seqs:
        lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
        lq, gt = lq.to(device), gt.to(device)

        z_LQ = encode(backbone, lq)[ref_layer].float()
        z_HQ = encode(backbone, gt)[ref_layer].float()

        # DiT pipeline
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_HQ_pred = dit.sample(z_LQ, num_steps=num_steps).float()
        h_b, skips = render.encode(lq, kv_layers=[z_HQ_pred], delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ_pred], delta_layers=None)
        pred_dit_f = pred.float()
        p_dit = mean_psnr(to_u8(pred_dit_f), to_u8(gt))
        psnrs_dit.append(p_dit)
        if with_iqa:
            iqa = compute_iqa(pred_dit_f, gt, device=device)
            for k_ in iqa_acc: iqa_acc[k_].append(iqa[k_])

        # Oracle render upper bound
        h_b, skips = render.encode(lq, kv_layers=[z_HQ], delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_o = render.decode(h_b, skips, lq, kv_layers=[z_HQ], delta_layers=None)
        p_o = mean_psnr(to_u8(pred_o.float()), to_u8(gt))
        psnrs_oracle.append(p_o)

        if verbose:
            extra = f" {fmt_iqa(iqa)}" if with_iqa else ""
            print(f"    [{name}] DiT={p_dit:.3f}{extra}  oracle={p_o:.3f}")

    dit.train(); render.train()
    if backbone is not None: backbone.eval()
    mean_iqa = ({k: float(np.mean(v)) for k, v in iqa_acc.items()} if with_iqa else None)
    return float(np.mean(psnrs_dit)), float(np.mean(psnrs_oracle)), mean_iqa


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lambda_pixel", type=float, default=1.0)
    ap.add_argument("--lambda_repa", type=float, default=0.5, help="E2b proven default")
    ap.add_argument("--early_stop_iter", type=int, default=900, help="HASTE early-stop on REPA")
    ap.add_argument("--dit_dim", type=int, default=512)
    ap.add_argument("--dit_depth", type=int, default=12)
    ap.add_argument("--dit_heads", type=int, default=8)
    ap.add_argument("--eval_steps", type=int, default=4)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    main(ap.parse_args())
