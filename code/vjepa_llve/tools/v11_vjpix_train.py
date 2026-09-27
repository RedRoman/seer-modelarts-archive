"""v5f VJ-Pixel-Diffusion: DDPM directly in pixel space, V-JEPA cross-attn condition.

Tests if all v1-v5a's "operate in some latent space" assumption is wrong.
Here: pixel-space DDPM, V-JEPA only as conditioning.

Architecture:
  V-JEPA encoder (frozen 304M) on LQ → z_LQ_vjepa (B, 2048, 1024)
  Pixel UNet on (B, 6, T, 256, 256) [concat(noisy_HQ, LQ)] with V-JEPA cross-attn
  → eps prediction (B, 3, T, 256, 256)

Training:
  z_t = sqrt(a_t) * GT + sqrt(1-a_t) * eps
  input = concat(z_t, LQ)
  eps_pred = UNet(input, t, kv=z_LQ_vjepa)
  L = MSE(eps_pred, eps)

Inference: DDIM 50-step starting from LQ-warmed noise.
"""
from __future__ import annotations
import argparse, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)
from tools.iqa_metrics import compute_iqa, fmt_iqa, save_compare_frame


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c); self.c1 = nn.Conv3d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c); self.c2 = nn.Conv3d(c, c, 3, padding=1)
    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class CrossAttn3D(nn.Module):
    def __init__(self, c_q, c_kv=1024, heads=4):
        super().__init__()
        self.h = heads; self.d = c_q // heads
        self.norm_q = nn.LayerNorm(c_q); self.norm_kv = nn.LayerNorm(c_kv)
        self.to_q = nn.Linear(c_q, c_q)
        self.to_k = nn.Linear(c_kv, c_q); self.to_v = nn.Linear(c_kv, c_q)
        self.to_out = nn.Linear(c_q, c_q)
        nn.init.zeros_(self.to_out.weight); nn.init.zeros_(self.to_out.bias)
    def forward(self, x, kv):
        B, C, T, H, W = x.shape
        x_t = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)
        q = self.to_q(self.norm_q(x_t)).reshape(B, T*H*W, self.h, self.d).permute(0, 2, 1, 3)
        kn = self.norm_kv(kv)
        k = self.to_k(kn).reshape(B, kv.shape[1], self.h, self.d).permute(0, 2, 1, 3)
        v = self.to_v(kn).reshape(B, kv.shape[1], self.h, self.d).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(B, T*H*W, C)
        out = self.to_out(out).reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return x + out


class PixelUNet(nn.Module):
    """3D UNet on pixels (256x256). Stem downsample 2x for memory."""
    def __init__(self, in_ch=6, base=24, T_diff=1000):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, base*6
        self.t_emb = nn.Sequential(nn.Linear(256, c4), nn.SiLU(), nn.Linear(c4, c4))
        # Stem: spatial down 2x
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, c1, (1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, c1), nn.SiLU())
        self.enc1 = nn.Sequential(_Res(c1), _Res(c1))
        self.down1 = nn.Conv3d(c1, c2, 3, stride=(1,2,2), padding=1)
        self.enc2 = nn.Sequential(_Res(c2), _Res(c2))
        self.down2 = nn.Conv3d(c2, c3, 3, stride=(1,2,2), padding=1)
        self.enc3 = nn.Sequential(_Res(c3), _Res(c3))
        self.down3 = nn.Conv3d(c3, c4, 3, stride=(1,2,2), padding=1)
        self.mid = nn.Sequential(_Res(c4), _Res(c4), _Res(c4))
        self.xattn = CrossAttn3D(c4, c_kv=1024, heads=4)
        self.up3 = nn.ConvTranspose3d(c4, c3, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.dec3 = nn.Sequential(_Res(c3), _Res(c3))
        self.up2 = nn.ConvTranspose3d(c3, c2, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.dec2 = nn.Sequential(_Res(c2), _Res(c2))
        self.up1 = nn.ConvTranspose3d(c2, c1, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1))
        self.dec1 = nn.Sequential(_Res(c1), _Res(c1))
        # Head: spatial up 2x to recover full 256
        self.head = nn.Sequential(
            nn.ConvTranspose3d(c1, c1, (1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, c1), nn.SiLU(),
            nn.Conv3d(c1, 3, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _sin(t, dim=256):
        half = dim // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        return torch.cat([torch.cos(t.float()[:, None] * f[None, :]), torch.sin(t.float()[:, None] * f[None, :])], dim=-1)

    def forward(self, x_in, t, kv):
        # x_in: (B, 6, T, 256, 256), t: (B,), kv: (B, 2048, 1024)
        t_e = self.t_emb(self._sin(t))
        h = self.stem(x_in)
        s1 = self.enc1(h); h = self.down1(s1)
        s2 = self.enc2(h); h = self.down2(s2)
        s3 = self.enc3(h); h = self.down3(s3)
        h = self.mid(h)
        h = h + t_e.view(t_e.shape[0], -1, 1, 1, 1)
        h = self.xattn(h, kv)
        h = self.up3(h); h = self.dec3(h + s3)
        h = self.up2(h); h = self.dec2(h + s2)
        h = self.up1(h); h = self.dec1(h + s1)
        return self.head(h)


def make_betas(T=1000): return torch.linspace(1e-4, 2e-2, T)


def ddim_sample(model, lq, kv, alphas_bar, num_steps=50):
    T_full = alphas_bar.shape[0]
    timesteps = torch.linspace(T_full - 1, 0, num_steps + 1).long()
    eps0 = torch.randn_like(lq)
    a_T = alphas_bar[timesteps[0]]
    z = a_T.sqrt() * lq + (1 - a_T).sqrt() * eps0   # warm start from LQ
    for i in range(num_steps):
        t_cur = timesteps[i].item()
        t_next = timesteps[i + 1].item()
        a_cur = alphas_bar[t_cur]
        a_next = alphas_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=z.device)
        t_tensor = torch.full((z.shape[0],), t_cur, device=z.device, dtype=torch.float32) / T_full
        x_in = torch.cat([z, lq], dim=1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eps_pred = model(x_in, t_tensor, kv).float()
        x0 = (z - (1 - a_cur).sqrt() * eps_pred) / a_cur.sqrt()
        x0 = x0.clamp(0, 1)  # static thresholding — prevents divergence at small a_t
        z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps_pred
    return z.clamp(0, 1)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    unet = PixelUNet(in_ch=6, base=args.base, T_diff=args.T_diff).to(device)
    n_u = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    betas = make_betas(args.T_diff).to(device)
    alphas_bar = torch.cumprod(1.0 - betas, dim=0)

    print(f"[VJ-Pix-DDPM] V-JEPA encoder 304M frozen")
    print(f"              PixelUNet: {n_u/1e6:.2f}M trainable (base={args.base})")
    print(f"              DDPM T_diff={args.T_diff}, DDIM {args.eval_steps}-step")

    optim = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        unet.load_state_dict(ckpt["unet"])
        print(f"[eval-only] loaded ckpt → {args.out_dir}/ckpt.pt")
    else:
        t0 = time.time()
        for it in range(args.iters):
            idx = int(rng.integers(0, len(train_seqs)))
            _, lq_p, gt_p = train_seqs[idx]
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)

            with torch.no_grad():
                kv = encode(backbone, lq)[args.ref_layer].float()

            B = gt.shape[0]
            t_int = torch.randint(0, args.T_diff, (B,), device=device)
            a_t = alphas_bar[t_int].view(B, 1, 1, 1, 1)
            eps = torch.randn_like(gt)
            z_t = a_t.sqrt() * gt + (1 - a_t).sqrt() * eps
            x_in = torch.cat([z_t, lq], dim=1)
            t_norm = t_int.float() / args.T_diff

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eps_pred = unet(x_in, t_norm, kv)
            L = F.mse_loss(eps_pred.float(), eps)

            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L_eps={L.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L_eps": L.item()})
                t0 = time.time()

        torch.save({"unet": unet.state_dict(), "args": vars(args), "log": log,
                    "stage": "vj_pix_ddpm"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    print(f"\n=== EVAL: VJ-Pix-DDPM (DDIM {args.eval_steps}-step) ===")
    unet.eval()
    psnrs = []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            kv = encode(backbone, lq)[args.ref_layer].float()
            pred = ddim_sample(unet, lq, kv, alphas_bar, num_steps=args.eval_steps)
            pred_f = pred.float()
            p = mean_psnr(to_u8(pred_f), to_u8(gt))
            iqa = compute_iqa(pred_f, gt, device=device)
            psnrs.append(p)
            for k in iqa_acc: iqa_acc[k].append(iqa[k])
            print(f"    [{name}] PSNR={p:.3f} {fmt_iqa(iqa)}")
            if args.save_viz:
                save_compare_frame(os.path.join(args.out_dir, "viz", f"{name}.png"),
                                   lq, pred_f, gt)
    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "VJPIX_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# v5f VJ-Pixel-Diffusion: pixel-space DDPM + V-JEPA cross-attn\n\n")
        f.write(f"PixelUNet {n_u/1e6:.2f}M trainable; V-JEPA 304M frozen\n\n")
        f.write(f"## SDSD held-out\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"vs E2b15k 20.028, vs CLIP15k 20.213, vs v9.0b 20.578\n")
    print(f"\nVJ-Pix-DDPM PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")


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
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--T_diff", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    ap.add_argument("--save_viz", action="store_true",
                    help="Save LQ|Pred|GT comparison PNGs to out_dir/viz/ during eval")
    main(ap.parse_args())
