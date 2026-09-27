"""SuperNAFNet++ V-JEPA (Exp B): architectural swaps, no size increase.

vs SuperNAFNet (PSNR 20.674):
  Architectural swaps:
    1. NAFBlock SimpleGate → SwiGLU (modern gated FFN, same params)
    2. Bottleneck TemporalAttn (softmax, ~2.4M) → DepthwiseConv3D temporal kernel 5
       (different inductive bias: local conv vs global content attn, ~600K, FEWER params)
  Teacher swap:
    3. Multi-teacher REPA (CLIP + DINOv2) → single V-JEPA layer 3 (training-focus)
  Loss addition:
    4. + LPIPS perceptual loss weight 0.1 (NAFNet++ trick)

Same: NAFNet3D base=96 backbone, V-JEPA cross-attn at bottleneck, EMA + warmup + 30k iter.
"""
from __future__ import annotations
import argparse, os, sys, time

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
from tools.v11_nafnet_train import LayerNorm3D, safe_cos_loss
from tools.v11_nafnet_pp_train import EMA, lr_warmup_cosine
from tools.v11_supernafnet_train import VJepaCrossAttn  # reuse


# ---------- Modified NAFBlock with SwiGLU instead of SimpleGate -----------
class SwiGLU(nn.Module):
    """SwiGLU = SiLU(x_a) * x_b where (x_a, x_b) = chunk(x, 2, dim=1).
    Same param count as SimpleGate but with SiLU nonlinearity (modern LLM/diffusion default)."""
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return F.silu(a) * b


class NAFBlock3D_SwiGLU(nn.Module):
    """Same structure as NAFBlock3D but SimpleGate → SwiGLU."""
    def __init__(self, c, dw_expand=2, ffn_expand=2, use_ckpt=False):
        super().__init__()
        self.use_ckpt = use_ckpt
        c_dw = c * dw_expand
        c_ffn = c * ffn_expand
        self.norm1 = LayerNorm3D(c)
        self.conv1 = nn.Conv3d(c, c_dw, 1)
        self.conv2 = nn.Conv3d(c_dw, c_dw, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=c_dw)
        self.sg1 = SwiGLU()                                       # ⭐ swap from SimpleGate
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c_dw // 2, c_dw // 2, 1),
        )
        self.conv3 = nn.Conv3d(c_dw // 2, c, 1)
        self.norm2 = LayerNorm3D(c)
        self.conv4 = nn.Conv3d(c, c_ffn, 1)
        self.sg2 = SwiGLU()                                       # ⭐ swap from SimpleGate
        self.conv5 = nn.Conv3d(c_ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1, 1))

    def _forward(self, x):
        h = self.norm1(x); h = self.conv1(h); h = self.conv2(h)
        h = self.sg1(h); h = h * self.sca(h); h = self.conv3(h)
        x = x + h * self.beta
        h = self.norm2(x); h = self.conv4(h); h = self.sg2(h); h = self.conv5(h)
        return x + h * self.gamma

    def forward(self, x):
        if self.use_ckpt and self.training and x.requires_grad:
            return ckpt.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


# ---------- Temporal mixing: DepthwiseConv3D (replaces softmax TemporalAttn) -----
class TemporalConvMixer(nn.Module):
    """Replace softmax TemporalAttn (O(T²)) with depthwise temporal conv (O(T)).

    DepthwiseConv3D kernel (5,1,1) + pointwise mixing.
    For T=16, kernel 5 covers ±2 frames neighborhood.
    """
    def __init__(self, c, t_kernel=5):
        super().__init__()
        pad_t = t_kernel // 2
        self.norm = LayerNorm3D(c)
        self.dw = nn.Conv3d(c, c, kernel_size=(t_kernel, 1, 1),
                            padding=(pad_t, 0, 0), groups=c)
        self.pw = nn.Conv3d(c, c, kernel_size=1)
        nn.init.zeros_(self.pw.weight)                            # zero-init residual
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1, 1))

    def forward(self, x):
        h = self.norm(x)
        h = self.dw(h)
        h = self.pw(h)
        return x + self.gamma * h


# ---------- SuperNAFNet++ V-JEPA model -----------------------------------
class SuperNAFNetPP(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=96,
                 enc_blocks=(2, 2, 4), dec_blocks=(2, 2, 2),
                 bottleneck_blocks=4, use_grad_ckpt=True,
                 use_temporal_mix=True, use_vjepa_cross=True, vj_dim=1024):
        super().__init__()
        self.intro = nn.Conv3d(in_ch, base, 3, padding=1)
        gck = use_grad_ckpt
        c = base
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock3D_SwiGLU(c, use_ckpt=gck) for _ in range(n)]))
            self.downs.append(nn.Conv3d(c, c * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            c *= 2
        self.bottleneck_c = c
        self.bottleneck = nn.Sequential(*[NAFBlock3D_SwiGLU(c, use_ckpt=gck) for _ in range(bottleneck_blocks)])
        # ⭐ component swap: TemporalAttn → temporal conv mixer
        self.temporal_mix = TemporalConvMixer(c, t_kernel=5) if use_temporal_mix else nn.Identity()
        self.vjepa_cross = VJepaCrossAttn(c, vj_dim=vj_dim) if use_vjepa_cross else None
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.ConvTranspose3d(c, c // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock3D_SwiGLU(c, use_ckpt=gck) for _ in range(n)]))
        self.outro = nn.Conv3d(base, out_ch, 3, padding=1)

    def forward(self, lq, z_vjepa=None):
        x = self.intro(lq)
        skips = []
        for enc, dn in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = dn(x)
        h_b = self.bottleneck(x)
        h_b = self.temporal_mix(h_b)                              # NEW (was TemporalAttn)
        if self.vjepa_cross is not None and z_vjepa is not None:
            h_b = self.vjepa_cross(h_b, z_vjepa)
        x = h_b
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x); x = x + skip; x = dec(x)
        out = self.outro(x)
        pred = (lq + out).clamp(0, 1)
        return pred, h_b


class REPAProjectorVJ(nn.Module):
    def __init__(self, in_ch=768, out_dim=1024):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)
    def forward(self, h_b, vjepa_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=vjepa_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


VJEPA_GRID = (8, 16, 16)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    net = SuperNAFNetPP(base=args.base,
                        enc_blocks=tuple(args.enc_blocks),
                        dec_blocks=tuple(args.dec_blocks),
                        bottleneck_blocks=args.bn_blocks,
                        use_temporal_mix=True, use_vjepa_cross=True).to(device)
    n_n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    proj_vj = REPAProjectorVJ(in_ch=net.bottleneck_c, out_dim=1024).to(device)
    n_p = sum(p.numel() for p in proj_vj.parameters())

    lpips_model = None
    if args.lambda_lpips > 0:
        import pyiqa
        lpips_model = pyiqa.create_metric("lpips", device=device, as_loss=True).eval()
        for p in lpips_model.parameters(): p.requires_grad_(False)

    print(f"[SuperNAFNet++ V-JEPA] V-JEPA encoder 304M frozen (REPA + cross-attn)")
    print(f"                       SuperNAFNetPP: {n_n/1e6:.2f}M trainable")
    print(f"                         - SwiGLU NAFBlocks (replaces SimpleGate)")
    print(f"                         - TemporalConvMixer (replaces TemporalAttn)")
    print(f"                       REPAProjectorVJ: {n_p/1e6:.2f}M (training only)")
    print(f"                       Loss: Charbonnier + {args.lambda_repa}·REPA(V-JEPA) + {args.lambda_lpips}·LPIPS")

    train_params = list(net.parameters()) + list(proj_vj.parameters())
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
        if "proj_vj" in ckpt_d:
            proj_vj.load_state_dict(ckpt_d["proj_vj"])
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

            with torch.no_grad():
                z_vjepa = encode(backbone, lq)[args.ref_layer].float()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, h_b = net(lq, z_vjepa=z_vjepa)
            L_rgb = charbonnier(pred.float(), gt)

            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                z_stu = proj_vj(h_b.float(), VJEPA_GRID)
                m = z_vjepa.mean(dim=1, keepdim=True); s = z_vjepa.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_vjepa - m) / s
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
                log.append({"iter": it+1, "lr": cur_lr, "L": L.item()})
                t0 = time.time()

        torch.save({"net": net.state_dict(), "proj_vj": proj_vj.state_dict(),
                    "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
                    "args": vars(args), "log": log, "stage": "supernafnet_pp"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    print(f"\n=== EVAL: SuperNAFNet++ V-JEPA (EMA + V-JEPA active) ===")
    backup = ema.apply_to(net)
    net.eval()
    psnrs = []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_vjepa = encode(backbone, lq)[args.ref_layer].float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, _ = net(lq, z_vjepa=z_vjepa)
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
    ema.restore(net, backup)

    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "SUPERNAFNETPP_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# SuperNAFNet++ V-JEPA (Exp B: SwiGLU + Conv-temporal + V-JEPA single teacher + LPIPS)\n\n")
        f.write(f"SuperNAFNetPP {n_n/1e6:.2f}M trainable; V-JEPA frozen (REPA target + cross-attn)\n")
        f.write(f"Architectural swaps: SwiGLU (vs SimpleGate); TemporalConvMixer (vs TemporalAttn)\n")
        f.write(f"Training: warmup {args.warmup_iters} + cosine, EMA={args.ema_decay}, iters={args.iters}\n\n")
        f.write(f"## SDSD held-out (EMA + V-JEPA active)\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}\n\n")
        f.write(f"vs SuperNAFNet 20.674 / 0.2745 LPIPS, vs NAFNet++ 20.708 / 0.2608\n")
    print(f"\nSuperNAFNet++ V-JEPA PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")


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
    ap.add_argument("--save_viz", action="store_true")
    ap.add_argument("--viz_dir", default=None)
    main(ap.parse_args())
