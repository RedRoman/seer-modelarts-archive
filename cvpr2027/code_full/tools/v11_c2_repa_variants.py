"""REPA variants V-A / V-B / V-C / V-D.

V-A Delta-REPA:
    align (h_b(HQ) - h_b(LQ))_proj to (z_H - z_L)
    Closes JDAP narrative loop.

V-B Adversarial REPA:
    align h_b(LQ)_proj to V-JEPA(LQ);
    push h_b(LQ)_proj away from V-JEPA(other_clip).

V-C Temporal REPA:
    align temporal-Gram matrix of decoder features to V-JEPA's temporal-Gram.

V-D Multi-layer REPA:
    align decoder feature to BOTH V-JEPA layer 17 AND layer 23.

All decoders are V-JEPA-free at inference (same as REPA standalone).

Spec: VJEPA_LLVE_CODE_REVIEW.md §V-A..V-D.
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
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


class ConvProj(nn.Module):
    def __init__(self, in_ch=256, out_dim=1024):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)

    def forward(self, h_b, vjepa_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=vjepa_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


def _norm_layer(z):
    """Spatial norm for teacher tokens."""
    m = z.mean(dim=1, keepdim=True); s = z.std(dim=1, keepdim=True) + 1e-6
    return (z - m) / s


def repa_loss_va_delta(projector, h_b_lq, h_b_hq, z_L, z_H, ref_layer, vjepa_grid):
    """V-A: align decoder delta to V-JEPA delta."""
    z_stu_lq = projector(h_b_lq, vjepa_grid)
    z_stu_hq = projector(h_b_hq, vjepa_grid)
    delta_dec = z_stu_hq - z_stu_lq
    delta_vje = z_L[ref_layer].float() - z_H[ref_layer].float()  # WAIT: should be z_H - z_L
    # Correct sign: oracle delta = z_H - z_L
    delta_vje = z_H[ref_layer].float() - z_L[ref_layer].float()
    delta_vje = _norm_layer(delta_vje)
    return -(F.normalize(delta_dec, dim=-1) * F.normalize(delta_vje, dim=-1)).sum(-1).mean()


def repa_loss_vb_contrastive(projector, h_b_lq, z_L, z_L_other, ref_layer, vjepa_grid, margin=0.1):
    """V-B: contrastive REPA — pull to V-JEPA(LQ), push from V-JEPA(other)."""
    z_stu = projector(h_b_lq, vjepa_grid)
    z_stu = F.normalize(z_stu, dim=-1)
    z_pos = F.normalize(_norm_layer(z_L[ref_layer].float()), dim=-1)
    z_neg = F.normalize(_norm_layer(z_L_other[ref_layer].float()), dim=-1)
    pos_sim = (z_stu * z_pos).sum(-1).mean()
    neg_sim = (z_stu * z_neg).sum(-1).mean()
    # we want pos_sim high, neg_sim low
    return -pos_sim + F.relu(neg_sim - pos_sim + margin)


def repa_loss_vc_temporal(projector, h_b_lq, z_L, ref_layer, vjepa_grid):
    """V-C: align temporal-relation Gram matrix.
    Pool features spatially per-frame; compute (T,T) Gram; match across student/teacher."""
    z_stu = projector(h_b_lq, vjepa_grid)        # (B, T*H*W, D)
    B, N, D = z_stu.shape
    T_v, H_v, W_v = vjepa_grid
    z_stu_t = z_stu.view(B, T_v, H_v * W_v, D).mean(dim=2)   # (B, T, D)
    z_tea = z_L[ref_layer].float().view(B, T_v, H_v * W_v, -1).mean(dim=2)  # (B, T, D)
    G_s = torch.bmm(F.normalize(z_stu_t, dim=-1), F.normalize(z_stu_t, dim=-1).transpose(1, 2))
    G_t = torch.bmm(F.normalize(z_tea, dim=-1), F.normalize(z_tea, dim=-1).transpose(1, 2))
    return F.l1_loss(G_s, G_t)


def repa_loss_vd_multilayer(projectors, h_b_lq, z_L, ref_layers, vjepa_grid):
    """V-D: align to multiple V-JEPA layers in parallel."""
    losses = []
    for proj, layer in zip(projectors, ref_layers):
        z_stu = F.normalize(proj(h_b_lq, vjepa_grid), dim=-1)
        z_tea = F.normalize(_norm_layer(z_L[layer].float()), dim=-1)
        losses.append(-(z_stu * z_tea).sum(-1).mean())
    return sum(losses) / len(losses)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    if args.variant == "VD_multilayer":
        ref_layers = [int(x) for x in args.multilayer_ref.split(",")]
        projectors = nn.ModuleList([ConvProj(in_ch=decoder.dims[-1], out_dim=decoder.vjepa_dim).to(device)
                                    for _ in ref_layers])
        proj_params = list(projectors.parameters())
        n_p = sum(p.numel() for p in proj_params)
    else:
        projector = ConvProj(in_ch=decoder.dims[-1], out_dim=decoder.vjepa_dim).to(device)
        proj_params = list(projector.parameters())
        n_p = sum(p.numel() for p in proj_params)

    print(f"[{args.variant}] decoder {n_d/1e6:.2f}M + projector(s) {n_p/1e6:.2f}M; "
          f"lambda={args.lambda_repa}, ref_layer={args.ref_layer}")

    optim = torch.optim.AdamW(list(decoder.parameters()) + proj_params,
                              lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, _ = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    log = []
    t0 = time.time()
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)
        # decoder forward (no V-JEPA)
        h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
        L_rgb = charbonnier(pred.float(), gt)

        L_repa = torch.zeros((), device=device)
        if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
            with torch.no_grad():
                z_L = encode(backbone, lq)
            vg = tuple(decoder.vjepa_grid)
            if args.variant == "VA_delta":
                with torch.no_grad():
                    z_H = encode(backbone, gt)
                    h_b_hq, _ = decoder.encode(gt, kv_layers=None, delta_layers=None)
                L_repa = repa_loss_va_delta(projector, h_b, h_b_hq, z_L, z_H, args.ref_layer, vg)
            elif args.variant == "VB_contrastive":
                # pick another sequence for negative
                other = int(rng.integers(0, len(train_seqs)))
                while other == idx: other = int(rng.integers(0, len(train_seqs)))
                _, lq_o, gt_o = train_seqs[other]
                lq_oo, _ = load_clip(lq_o, gt_o, crop_size=args.crop)
                with torch.no_grad():
                    z_L_other = encode(backbone, lq_oo.to(device))
                L_repa = repa_loss_vb_contrastive(projector, h_b, z_L, z_L_other, args.ref_layer, vg, margin=args.margin)
            elif args.variant == "VC_temporal":
                L_repa = repa_loss_vc_temporal(projector, h_b, z_L, args.ref_layer, vg)
            elif args.variant == "VD_multilayer":
                L_repa = repa_loss_vd_multilayer(projectors, h_b, z_L, ref_layers, vg)
            else:
                raise ValueError(args.variant)

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
                 "stage": f"repa_{args.variant}"}
    if args.variant == "VD_multilayer":
        save_dict["projectors"] = projectors.state_dict()
    else:
        save_dict["projector"] = projector.state_dict()
    torch.save(save_dict, os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Eval: just no_prior (decoder doesn't use V-JEPA)
    print("\n=== EVAL ===")
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
            p = mean_psnr(to_u8(pred.float()), to_u8(gt))
            psnrs.append(p)
            print(f"    [{name}] no_prior pred = {p:.3f}")
    mean_psnr_val = float(np.mean(psnrs))
    md = os.path.join(args.out_dir, "REPA_VARIANT_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# REPA Variant {args.variant} Eval\n\n")
        f.write(f"variant: {args.variant}\n")
        f.write(f"decoder: {n_d/1e6:.2f}M (no V-JEPA at inference)\n")
        f.write(f"projector(s): {n_p/1e6:.2f}M\n")
        f.write(f"lambda_repa: {args.lambda_repa}, ref_layer: {args.ref_layer}\n\n")
        f.write(f"## no_prior PSNR = {mean_psnr_val:.3f}\n\n")
        f.write(f"vs Stage 1 baseline 17.962 (∆ = {mean_psnr_val - 17.962:+.3f})\n")
        f.write(f"vs E2 REPA standalone 19.433 (∆ = {mean_psnr_val - 19.433:+.3f})\n")
    print(f"\n  no_prior PSNR = {mean_psnr_val:.3f}  (E2 REPA standalone = 19.433)")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["VA_delta", "VB_contrastive", "VC_temporal", "VD_multilayer"])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--early_stop_iter", type=int, default=-1)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--multilayer_ref", default="2,3", help="VD: comma-separated adapter indices (V-JEPA layer 17,23)")
    main(ap.parse_args())
