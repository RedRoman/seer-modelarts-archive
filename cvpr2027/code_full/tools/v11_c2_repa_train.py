"""E2: REPA-style alignment training.

Instead of injecting V-JEPA via cross-attn KV, add a cosine alignment loss
between the decoder bottleneck h_b (projected) and frozen V-JEPA z_L tokens.

Decoder uses NO V-JEPA at inference (kv_layers=None always). The alignment
loss only shapes the decoder during training.

Spec: VJEPA_LLVE_CODE_REVIEW.md §E2.
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.iqa_metrics import compute_iqa, fmt_iqa
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


class REPAProjector(nn.Module):
    """iREPA-style 3x3x3 conv projector + spatial-temporal norm."""
    def __init__(self, in_ch=256, out_dim=1024):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)

    def forward(self, h_b, vjepa_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=vjepa_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    projector = REPAProjector(in_ch=decoder.dims[-1], out_dim=decoder.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())
    print(f"[REPA] decoder {n_d/1e6:.2f}M + projector {n_p/1e6:.2f}M; "
          f"alignment to V-JEPA layer {args.ref_layer}, lambda={args.lambda_repa}, "
          f"early_stop_iter={args.early_stop_iter}")

    optim = torch.optim.AdamW(list(decoder.parameters()) + list(projector.parameters()),
                              lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, _ = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    log = []
    if args.eval_only:
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        decoder.load_state_dict(ckpt["decoder"])
        projector.load_state_dict(ckpt["projector"])
        print(f"[eval-only] loaded ckpt → {args.out_dir}/ckpt.pt")
    else:
        t0 = time.time()
        for it in range(args.iters):
            idx = int(rng.integers(0, len(train_seqs)))
            name, lq_p, gt_p = train_seqs[idx]
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)

            # Decoder forward (NO V-JEPA at inference)
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
            L_rgb = charbonnier(pred.float(), gt)

            # Alignment loss: project h_b → V-JEPA dim, align to V-JEPA z_L tokens
            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                with torch.no_grad():
                    z_L = encode(backbone, lq)
                z_stu = projector(h_b, tuple(decoder.vjepa_grid))         # (B, N_v, D)
                z_tea = z_L[args.ref_layer].float()                       # (B, N_v, D)
                # Spatial norm on teacher too
                m = z_tea.mean(dim=1, keepdim=True); s = z_tea.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_tea - m) / s
                # Negative cosine
                z_stu_n = F.normalize(z_stu, dim=-1)
                z_tea_n = F.normalize(z_tea, dim=-1)
                L_repa = -(z_stu_n * z_tea_n).sum(dim=-1).mean()

            L = args.rgb_weight * L_rgb + args.lambda_repa * L_repa

            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + list(projector.parameters()), 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} L_rgb={L_rgb.item():.4f} "
                      f"L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(),
                            "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
                t0 = time.time()

        torch.save({"decoder": decoder.state_dict(), "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "repa_standalone"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Quick eval: just no_prior mode (decoder doesn't use V-JEPA at inference)
    print("\n=== EVAL (no V-JEPA at inference) ===")
    decoder.eval()
    test_seqs = build_split_seqs()[1]
    psnrs = []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
            pred_f = pred.float()
            p = mean_psnr(to_u8(pred_f), to_u8(gt))
            iqa = compute_iqa(pred_f, gt, device=device)
            psnrs.append(p)
            for k in iqa_acc: iqa_acc[k].append(iqa[k])
            print(f"    [{name}] PSNR={p:.3f} {fmt_iqa(iqa)}")
    mean_psnr_val = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "REPA_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# REPA Standalone (E2) Eval\n\n")
        f.write(f"decoder: {n_d/1e6:.2f}M (no V-JEPA at inference)\n")
        f.write(f"projector: {n_p/1e6:.2f}M (training-only)\n")
        f.write(f"lambda_repa: {args.lambda_repa}, early_stop_iter: {args.early_stop_iter}\n\n")
        f.write(f"## SDSD held-out\n\n")
        f.write(f"- PSNR    = {mean_psnr_val:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"Compare to C-2 Stage 1 no_prior 17.962 — does REPA training raise the no-prior decoder?\n")
    print(f"\nno_prior PSNR={mean_psnr_val:.3f} {fmt_iqa(mean_iqa)}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--early_stop_iter", type=int, default=-1, help="HASTE-style early stop; -1 = never")
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    main(ap.parse_args())
