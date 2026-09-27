"""V: Velocity-Retinex architecture.

Architecture:
  R-branch: DeltaModulatedDecoder (33M) outputs R (3-ch, illumination-invariant)
            R-branch's bottleneck h_b is REPA-aligned to V-JEPA layer 3
            (V-JEPA features ARE illumination-invariant by training, so R
            should align with them)
  L-net:    Tiny 3D conv stack (~1M) outputs L_log (1-ch, log-illumination)
            Supervised by spatial smoothness regularizer
  Recombination:
            pred = R * exp(L_log) clamped to [0, 1]

Loss:
  L_rgb     = charbonnier(pred, GT)
  L_repa_R  = -cos(projector(R-branch h_b), z_LQ_layer3)   (HASTE early-stop)
  L_smooth  = mean(|grad_x L_log|^2 + |grad_y L_log|^2 + 0.1*|grad_t L_log|^2)
  L_total   = L_rgb + 0.5 * L_repa_R + 0.01 * L_smooth

The architectural innovation: V-JEPA's illumination-invariance is used as
the supervisory target for an INTERNAL R-branch (not the final output).
This is a third V-JEPA usage mode beyond REPA-on-bottleneck-of-pixel-output
and cross-attn KV. Output is forced to be a Retinex decomposition.
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
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
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


class IlluminationNet(nn.Module):
    """Tiny 3D conv stack predicting log-illumination (1-ch, smooth)."""
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, base, 5, padding=2),
            nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv3d(base, base, 5, padding=2),
            nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv3d(base, base, 5, padding=2),
            nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv3d(base, 1, 5, padding=2),
        )

    def forward(self, lq):
        # lq: (B, 3, T, H, W) in [0,1]
        return self.net(lq)   # (B, 1, T, H, W) log-illumination


def safe_cos_loss(z_stu, z_tea):
    z_stu = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
    z_tea = F.normalize(z_tea.float(), dim=-1, eps=1e-3)
    cos = (z_stu * z_tea).sum(dim=-1).clamp(-1.5, 1.5)
    return -cos.mean()


def smoothness_loss(L_log):
    """Spatial-temporal gradient penalty on log-illumination."""
    dx = L_log[..., 1:] - L_log[..., :-1]
    dy = L_log[..., 1:, :] - L_log[..., :-1, :]
    dt = L_log[:, :, 1:] - L_log[:, :, :-1]
    return (dx.pow(2).mean() + dy.pow(2).mean() + 0.1 * dt.pow(2).mean())


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    # R-branch: standard decoder, output is reflectance (3-ch, will be softplus'd)
    r_decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_r = sum(p.numel() for p in r_decoder.parameters() if p.requires_grad)

    # L-net: tiny illumination predictor
    l_net = IlluminationNet(base=args.l_base).to(device)
    n_l = sum(p.numel() for p in l_net.parameters() if p.requires_grad)

    # REPA projector for R-branch bottleneck
    projector = REPAProjector(in_ch=r_decoder.dims[-1], out_dim=r_decoder.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    print(f"[V Velocity-Retinex]")
    print(f"  R-branch (decoder): {n_r/1e6:.2f}M trainable")
    print(f"  L-net:              {n_l/1e6:.2f}M trainable")
    print(f"  REPA projector:     {n_p/1e6:.2f}M trainable")
    print(f"  V-JEPA encoder:     304M frozen")
    print(f"  TOTAL trainable:    {(n_r+n_l+n_p)/1e6:.2f}M")
    print(f"  output: pred = R * exp(L_log), clamp to [0,1]")
    print(f"  losses: L_rgb + {args.lambda_repa}*L_repa_R + {args.lambda_smooth}*L_smooth")

    train_params = list(r_decoder.parameters()) + list(l_net.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    t0 = time.time()
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        _, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        # R-branch forward (DBN5: R = clamp(R_raw, 0, 1) via sigmoid for bounded output)
        h_b, skips = r_decoder.encode(lq, kv_layers=None, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            R_raw = r_decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
        R = torch.sigmoid(R_raw.float())   # bounded reflectance image in [0,1]

        # L-net forward (DBN5: bound L_log to [-2, +2] via tanh)
        L_log_raw = l_net(lq)              # (B,1,T,H,W)
        L_log = args.l_log_bound * torch.tanh(L_log_raw / args.l_log_bound)  # bounded
        L = torch.exp(L_log)               # in [exp(-2), exp(2)] ≈ [0.135, 7.39]

        # Recombine (DBN5: clamp final output)
        pred = (R * L).clamp(0, 1)

        L_rgb = charbonnier(pred, gt)

        # REPA on R-branch bottleneck
        L_repa = torch.zeros((), device=device)
        if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
            with torch.no_grad():
                z_L = encode(backbone, lq)[args.ref_layer].float()
            z_stu = projector(h_b, tuple(r_decoder.vjepa_grid))
            m = z_L.mean(dim=1, keepdim=True); s = z_L.std(dim=1, keepdim=True) + 1e-6
            z_tea = (z_L - m) / s
            L_repa = safe_cos_loss(z_stu, z_tea)

        L_smooth = smoothness_loss(L_log)

        # DBN6: L_balance prevents R/L scale ambiguity (R shrink while L grow)
        L_balance = (L_log.mean() ** 2) + 0.1 * L_log.var()

        L_total = (L_rgb
                   + args.lambda_repa * L_repa
                   + args.lambda_smooth * L_smooth
                   + args.lambda_balance * L_balance)

        optim.zero_grad(set_to_none=True); L_total.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            r_mean = float(R.detach().mean().item())
            l_mean = float(L.detach().mean().item())
            l_log_mean = float(L_log.detach().mean().item())
            l_log_std = float(L_log.detach().std().item())
            print(f"  it {it+1}/{args.iters} L_rgb={L_rgb.item():.4f} "
                  f"L_repa={L_repa.item():.4f} L_smooth={L_smooth.item():.5f} "
                  f"L_bal={L_balance.item():.5f} "
                  f"R_mean={r_mean:.3f} L_mean={l_mean:.3f} "
                  f"L_log_mean={l_log_mean:+.3f} L_log_std={l_log_std:.3f} "
                  f"dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L_rgb": L_rgb.item(), "L_repa": L_repa.item(),
                        "L_smooth": L_smooth.item(), "L_balance": L_balance.item(),
                        "R_mean": r_mean, "L_mean": l_mean,
                        "L_log_mean": l_log_mean, "L_log_std": l_log_std})
            t0 = time.time()

    torch.save({"r_decoder": r_decoder.state_dict(), "l_net": l_net.state_dict(),
                "projector": projector.state_dict(),
                "args": vars(args), "log": log, "stage": "velocity_retinex"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Eval
    r_decoder.eval(); l_net.eval()
    print("\n=== EVAL ===")
    psnrs = []
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            h_b, skips = r_decoder.encode(lq, kv_layers=None, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                R_raw = r_decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
            R = F.softplus(R_raw.float()) + 1e-3
            L_log = l_net(lq)
            L = torch.exp(L_log.clamp(-10.0, 5.0))
            pred = (R * L).clamp(0, 1)
            p = mean_psnr(to_u8(pred), to_u8(gt))
            psnrs.append(p)
            print(f"    [{name}] LQ→pred = {p:.3f}  (R_mean={float(R.mean()):.3f} L_mean={float(L.mean()):.3f})")
    mean_p = float(np.mean(psnrs))
    md = os.path.join(args.out_dir, "V_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# V: Velocity-Retinex\n\n")
        f.write(f"R-branch {n_r/1e6:.2f}M + L-net {n_l/1e6:.2f}M + projector {n_p/1e6:.2f}M\n")
        f.write(f"V-JEPA encoder 304M frozen; pred = R * exp(L_log)\n")
        f.write(f"lambda_repa={args.lambda_repa}, lambda_smooth={args.lambda_smooth}\n")
        f.write(f"early_stop_iter={args.early_stop_iter}, lr={args.lr}, iters={args.iters}\n\n")
        f.write(f"## SDSD held-out PSNR = {mean_p:.3f}\n")
        f.write(f"vs E2b 19.556, vs pure_rgb 18.533, vs PATH P 19.724\n\n")
        f.write(f"per-seq PSNRs: {psnrs}\n")
    print(f"\nV PSNR = {mean_p:.3f}  (target: beat E2b 19.556 by demonstrating Retinex+REPA combo works)")
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
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lambda_repa", type=float, default=0.1, help="DBN6: lower than E2b's 0.5")
    ap.add_argument("--lambda_smooth", type=float, default=0.05, help="DBN6: spatial-temporal TV on L_log")
    ap.add_argument("--lambda_balance", type=float, default=0.01, help="DBN6: scale-disambiguation between R and L")
    ap.add_argument("--early_stop_iter", type=int, default=900)
    ap.add_argument("--l_base", type=int, default=32)
    ap.add_argument("--l_log_bound", type=float, default=2.0, help="DBN5: |L_log| bounded via tanh")
    main(ap.parse_args())
