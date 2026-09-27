"""V-LDM-LLVE: latent low-light video enhancement via SD-VAE.

Architecture:
    LQ pixels (B,3,T,256,256) → SD-VAE encode (per-frame, frozen) → z_LQ (B,4,T,32,32)
    z_LQ → 3D U-Net latent operator → z_HQ_pred (B,4,T,32,32)
    z_HQ_pred → SD-VAE decode (per-frame, frozen) → pixel pred

Loss = latent L1(z_pred, z_HQ) + RGB Charbonnier (after VAE decode).

SD-VAE is frozen throughout; only latent operator trains.

For 256² input, 16-frame clip:
  - Per-frame VAE encode/decode at fp16
  - Latent net operates on (B, 4, 16, 32, 32) — much cheaper than pixel-space U-Net
"""
from __future__ import annotations
import argparse, os, sys, time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr,
                                           charbonnier, build_split_seqs)

VAE_SNAP = "/cache/zihaohe/hf_cache/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493"
VAE_SCALE = 0.18215  # SD scaling factor


# ---- VAE helpers (per-frame, in chunks for memory) -----------------------
@torch.no_grad()
def vae_encode(vae, video, chunk=8):
    """video: (B,3,T,H,W) in [0,1]. Returns (B,4,T,H/8,W/8) latent (scaled)."""
    B, C, T, H, W = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    frames = frames * 2.0 - 1.0   # [0,1] → [-1,1]
    zs = []
    for i in range(0, frames.shape[0], chunk):
        z = vae.encode(frames[i:i+chunk]).latent_dist.sample() * VAE_SCALE
        zs.append(z)
    z = torch.cat(zs, dim=0)
    z = z.reshape(B, T, 4, H // 8, W // 8).permute(0, 2, 1, 3, 4).contiguous()
    return z


def vae_decode(vae, z, chunk=8):
    """z: (B,4,T,Hl,Wl). Returns (B,3,T,H,W) in [0,1]."""
    B, C, T, Hl, Wl = z.shape
    z = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, Hl, Wl) / VAE_SCALE
    pxs = []
    for i in range(0, z.shape[0], chunk):
        px = vae.decode(z[i:i+chunk]).sample
        pxs.append(px)
    px = torch.cat(pxs, dim=0)
    px = (px + 1.0) / 2.0   # [-1,1] → [0,1]
    px = px.clamp(0, 1)
    px = px.reshape(B, T, 3, Hl * 8, Wl * 8).permute(0, 2, 1, 3, 4).contiguous()
    return px


# ---- Latent operator: small 3D U-Net -------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c); self.c1 = nn.Conv3d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c); self.c2 = nn.Conv3d(c, c, 3, padding=1)
    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class LatentOperator(nn.Module):
    """3D U-Net on (B,4,T,32,32). 3 scales: 32, 16, 8."""
    def __init__(self, in_ch=4, base=128):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 3
        self.in_proj = nn.Conv3d(in_ch, c1, 3, padding=1)
        self.enc1 = nn.Sequential(_ResBlock(c1), _ResBlock(c1))
        self.down1 = nn.Conv3d(c1, c2, 3, stride=(1, 2, 2), padding=1)
        self.enc2 = nn.Sequential(_ResBlock(c2), _ResBlock(c2))
        self.down2 = nn.Conv3d(c2, c3, 3, stride=(1, 2, 2), padding=1)
        self.mid  = nn.Sequential(_ResBlock(c3), _ResBlock(c3), _ResBlock(c3))
        self.up2  = nn.ConvTranspose3d(c3, c2, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))
        self.dec2 = nn.Sequential(_ResBlock(c2), _ResBlock(c2))
        self.up1  = nn.ConvTranspose3d(c2, c1, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))
        self.dec1 = nn.Sequential(_ResBlock(c1), _ResBlock(c1))
        self.out  = nn.Conv3d(c1, in_ch, 3, padding=1)

    def forward(self, z):
        h = self.in_proj(z)
        s1 = self.enc1(h)
        h = self.down1(s1)
        s2 = self.enc2(h)
        h = self.down2(s2)
        h = self.mid(h)
        h = self.up2(h)
        h = self.dec2(h + s2)
        h = self.up1(h)
        h = self.dec1(h + s1)
        delta = self.out(h)
        return z + delta   # residual prediction


def _load_vae(device):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_SNAP, local_files_only=True).to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return vae


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    vae = _load_vae(device)
    op = LatentOperator(in_ch=4, base=args.base).to(device)
    n_op = sum(p.numel() for p in op.parameters() if p.requires_grad)
    print(f"[V-LDM] SD-VAE 83.65M frozen + LatentOp {n_op/1e6:.2f}M trainable")

    optim = torch.optim.AdamW(op.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

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

        # Encode (frozen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_lq = vae_encode(vae, lq).float()
            z_gt = vae_encode(vae, gt).float()

        # Latent operator forward
        z_pred = op(z_lq)
        L_lat = F.l1_loss(z_pred, z_gt)

        # RGB loss after VAE decode (use bfloat16 for VAE decode to save mem)
        L_rgb = torch.zeros((), device=device)
        if args.rgb_weight > 0:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_pix = vae_decode(vae, z_pred)
            L_rgb = charbonnier(pred_pix.float(), gt)

        L = args.lat_weight * L_lat + args.rgb_weight * L_rgb
        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(op.parameters(), 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} "
                  f"L_lat={L_lat.item():.4f} L_rgb={L_rgb.item():.4f} "
                  f"dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item(),
                        "L_lat": L_lat.item(), "L_rgb": L_rgb.item()})
            t0 = time.time()

    torch.save({"op": op.state_dict(), "args": vars(args), "log": log,
                "stage": "vae_latent_baseline"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Eval
    op.eval()
    print("\n=== EVAL: SDSD held-out (V-LDM-LLVE baseline) ===")
    psnrs, lq_psnrs = [], []
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_lq = vae_encode(vae, lq).float()
                z_pred = op(z_lq)
                pred = vae_decode(vae, z_pred)
            p = mean_psnr(to_u8(pred.float()), to_u8(gt))
            p_lq = mean_psnr(to_u8(lq), to_u8(gt))
            psnrs.append(p); lq_psnrs.append(p_lq)
            print(f"    [{name}] LQ={p_lq:.2f}  pred={p:.3f}")
    mean_p = float(np.mean(psnrs))
    md = os.path.join(args.out_dir, "VAE_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# V-LDM-LLVE (VAE latent restoration) Eval\n\n")
        f.write(f"SD-VAE 83.65M frozen + LatentOp {n_op/1e6:.2f}M trainable\n")
        f.write(f"latent base={args.base}, lat_w={args.lat_weight}, rgb_w={args.rgb_weight}\n\n")
        f.write(f"## SDSD held-out PSNR = {mean_p:.3f}  (vs E2b 19.556, vs pure_rgb 18.533)\n")
    print(f"\nSDSD PSNR = {mean_p:.3f}  (V-LDM baseline; vs E2b 19.556)")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", type=int, default=128, help="latent net base channels")
    ap.add_argument("--lat_weight", type=float, default=1.0)
    ap.add_argument("--rgb_weight", type=float, default=0.5)
    main(ap.parse_args())
