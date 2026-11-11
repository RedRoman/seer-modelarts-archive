"""More REPA designs: R-GT, R-Blend, R-Random, R-WarmUp, R-MultiDepth, R-NoProj."""
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
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


class ConvProj(nn.Module):
    def __init__(self, in_ch=256, out_dim=1024, k=3):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=k, padding=k//2)
    def forward(self, h_b, vjepa_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=vjepa_grid, mode="trilinear", align_corners=False)
        return x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()


def _norm(z):
    m = z.mean(dim=1, keepdim=True); s = z.std(dim=1, keepdim=True) + 1e-6
    return (z - m) / s


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})

    # backbone
    if args.mode == "R_random":
        # build V-JEPA architecture but DON'T load weights -> random ViT teacher
        backbone = build_vjepa_backbone(cfg["backbone"], pretrained=False).to(device).eval() if False else None
        # fallback: build normally then re-init
        backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
        # re-init most weights to random
        for p in backbone.parameters():
            if p.dim() >= 2: nn.init.xavier_normal_(p)
            else: nn.init.zeros_(p)
        for p in backbone.parameters(): p.requires_grad_(False)
        print("[R_random] backbone re-initialized to random — sanity control")
    else:
        backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
        for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    if args.mode == "R_no_proj":
        # No learnable projection. Decoder bottleneck channels < V-JEPA dim,
        # so zero-pad along channel dim to match.
        projector = None
        n_p = 0
    elif args.mode == "R_multidepth":
        # 3 projectors: shallow / mid / deep decoder layers → V-JEPA layers 1/2/3
        # We approximate by projecting h_b 3 ways
        projector = nn.ModuleList([
            ConvProj(decoder.dims[-1], decoder.vjepa_dim).to(device) for _ in range(3)
        ])
        n_p = sum(p.numel() for p in projector.parameters())
    else:
        projector = ConvProj(decoder.dims[-1], decoder.vjepa_dim).to(device)
        n_p = sum(p.numel() for p in projector.parameters())

    print(f"[{args.mode}] decoder {n_d/1e6:.2f}M + projector {n_p/1e6:.2f}M")

    proj_params = list(projector.parameters()) if projector is not None else []
    optim = torch.optim.AdamW(list(decoder.parameters()) + proj_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, _ = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    log = []
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
        L_rgb = charbonnier(pred.float(), gt)

        # alignment loss
        L_repa = torch.zeros((), device=device)
        # WarmUp: skip alignment for first N iters
        skip_align = (args.mode == "R_warmup") and (it < args.warmup_iters)
        # HASTE early stop
        late_stop = (args.early_stop_iter > 0 and it >= args.early_stop_iter)

        if (not skip_align) and (not late_stop) and args.lambda_repa > 0:
            with torch.no_grad():
                z_L = encode(backbone, lq)
                z_tea_LQ = _norm(z_L[args.ref_layer].float())
                if args.mode in ("R_GT", "R_blend"):
                    z_H = encode(backbone, gt)
                    z_tea_GT = _norm(z_H[args.ref_layer].float())

            vg = tuple(decoder.vjepa_grid)

            if args.mode == "R_GT":
                z_tea = z_tea_GT
                z_stu = projector(h_b, vg)
                L_repa = -(F.normalize(z_stu, -1) * F.normalize(z_tea, -1)).sum(-1).mean()
            elif args.mode == "R_blend":
                # alpha goes 0 → 1 over training: start LQ, end GT
                alpha = it / max(1, args.iters)
                z_tea = (1-alpha) * z_tea_LQ + alpha * z_tea_GT
                z_stu = projector(h_b, vg)
                L_repa = -(F.normalize(z_stu, -1) * F.normalize(z_tea, -1)).sum(-1).mean()
            elif args.mode == "R_random":
                # use the randomly-initialized backbone's output as target
                z_stu = projector(h_b, vg)
                L_repa = -(F.normalize(z_stu, -1) * F.normalize(z_tea_LQ, -1)).sum(-1).mean()
            elif args.mode == "R_warmup":
                z_stu = projector(h_b, vg)
                L_repa = -(F.normalize(z_stu, -1) * F.normalize(z_tea_LQ, -1)).sum(-1).mean()
            elif args.mode == "R_multidepth":
                # align same h_b 3 ways to V-JEPA layers 1/2/3 with separate projectors
                losses = []
                for i, layer in enumerate([1, 2, 3]):
                    z_stu = projector[i](h_b, vg)
                    z_tea = _norm(z_L[layer].float())
                    losses.append(-(F.normalize(z_stu, -1) * F.normalize(z_tea, -1)).sum(-1).mean())
                L_repa = sum(losses) / len(losses)
            elif args.mode == "R_no_proj":
                # zero-pad h_b channels to match V-JEPA dim, then align
                B, C, T, H, W = h_b.shape
                # interpolate to V-JEPA grid
                h_resized = F.interpolate(h_b, size=vg, mode="trilinear", align_corners=False)
                # zero-pad along channel dim
                pad = decoder.vjepa_dim - C
                if pad > 0:
                    h_padded = F.pad(h_resized, (0, 0, 0, 0, 0, 0, 0, pad))
                else:
                    h_padded = h_resized[:, :decoder.vjepa_dim]
                z_stu = h_padded.permute(0, 2, 3, 4, 1).reshape(B, -1, decoder.vjepa_dim)
                z_stu = _norm(z_stu)
                L_repa = -(F.normalize(z_stu, -1) * F.normalize(z_tea_LQ, -1)).sum(-1).mean()
            else:
                raise ValueError(args.mode)

        L = L_rgb + args.lambda_repa * L_repa
        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + proj_params, 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} L_rgb={L_rgb.item():.4f} "
                  f"L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item(), "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
            t0 = time.time()

    save_dict = {"decoder": decoder.state_dict(), "args": vars(args), "log": log,
                 "stage": f"repa_more_{args.mode}"}
    torch.save(save_dict, os.path.join(args.out_dir, "ckpt.pt"))

    decoder.eval()
    test_seqs = build_split_seqs()[1]
    psnrs = []
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
            psnrs.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
    val = float(np.mean(psnrs))
    md = os.path.join(args.out_dir, "REPA_MORE_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# REPA-More variant {args.mode}\n\n")
        f.write(f"## no_prior PSNR = {val:.3f}\n")
        f.write(f"vs E2b 19.556 (∆ = {val-19.556:+.3f})\n")
    print(f"\n[{args.mode}] PSNR = {val:.3f}  (E2b best = 19.556)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["R_GT","R_blend","R_random","R_warmup","R_multidepth","R_no_proj"])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--early_stop_iter", type=int, default=900)
    ap.add_argument("--warmup_iters", type=int, default=500)
    main(ap.parse_args())
