"""NAFNet-3D + V-JEPA-REPA: NAFNet (Chen et al. 2022) lightweight reimpl + REPA.

Architecture:
  V-JEPA encoder (frozen) — only used for REPA target during training.
  At inference: ZERO V-JEPA forward → pure feed-forward decoder.

  NAFNet3D (~30-50M trainable, configurable via --base):
    SimpleGate + Simplified Channel Attention (SCA) + LayerNorm3D
    3-stage encoder/decoder U-Net with skips, downsample 2x in spatial only
    (T=16 preserved through depth).

  Training:
    pred = NAFNet(LQ)                    # 3D conv, residual on LQ
    L_rgb = Charbonnier(pred, GT)
    REPA: project bottleneck h_b → align to V-JEPA(LQ) layer 3 (HASTE-gated)

Why try this:
  Current best (clip_30k 20.746) uses DeltaModulatedDecoder (33M conv U-Net)
  + REPA. NAFNet beats vanilla conv U-Nets on image restoration (SIDD,
  GoPro, REDS) by ~2-3 dB at same param budget. Hypothesis: same advantage
  carries over to LLVE under REPA training.
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

from models.vjepa_backbone import build_vjepa_backbone
from tools.iqa_metrics import compute_iqa, fmt_iqa, save_compare_frame
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


# ---------- NAFNet building blocks ------------------------------------
class LayerNorm3D(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps
    def forward(self, x):
        # x: (B, C, T, H, W) — channel-wise norm
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        return x * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock3D(nn.Module):
    """NAFNet block adapted for 3D (B, C, T, H, W).

    Two sub-blocks: depthwise spatial conv + SCA, then point-wise FFN.
    Both wrapped in pre-LayerNorm + learnable scale residual.
    Optional gradient checkpointing for memory.
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2, use_ckpt=False):
        super().__init__()
        self.use_ckpt = use_ckpt
        c_dw = c * dw_expand
        c_ffn = c * ffn_expand
        # block 1: depthwise spatial-temporal conv + SCA
        self.norm1 = LayerNorm3D(c)
        self.conv1 = nn.Conv3d(c, c_dw, 1)
        # depthwise spatial-only kernel (cheap)
        self.conv2 = nn.Conv3d(c_dw, c_dw, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=c_dw)
        self.sg1 = SimpleGate()
        c_after_sg1 = c_dw // 2
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c_after_sg1, c_after_sg1, 1),
        )
        self.conv3 = nn.Conv3d(c_after_sg1, c, 1)
        # block 2: pointwise FFN
        self.norm2 = LayerNorm3D(c)
        self.conv4 = nn.Conv3d(c, c_ffn, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv3d(c_ffn // 2, c, 1)
        # learnable residual scales
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1, 1))

    def _forward(self, x):
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.sg1(h)
        h = h * self.sca(h)
        h = self.conv3(h)
        x = x + h * self.beta
        h = self.norm2(x)
        h = self.conv4(h)
        h = self.sg2(h)
        h = self.conv5(h)
        x = x + h * self.gamma
        return x

    def forward(self, x):
        if self.use_ckpt and self.training and x.requires_grad:
            return ckpt.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class NAFNet3D(nn.Module):
    """3D NAFNet for video LLVE.

    Down/up-sample in SPATIAL only (T preserved for temporal coherence).
    Total params at default base=48: ~32M.
    """
    def __init__(self, in_ch=3, out_ch=3, base=48,
                 enc_blocks=(2, 2, 4), dec_blocks=(2, 2, 2),
                 bottleneck_blocks=4, use_grad_ckpt=True):
        super().__init__()
        assert len(enc_blocks) == len(dec_blocks)
        self.intro = nn.Conv3d(in_ch, base, 3, padding=1)
        gck = use_grad_ckpt

        c = base
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock3D(c, use_ckpt=gck) for _ in range(n)]))
            self.downs.append(nn.Conv3d(c, c * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            c *= 2

        self.bottleneck_c = c
        self.bottleneck = nn.Sequential(*[NAFBlock3D(c, use_ckpt=gck) for _ in range(bottleneck_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.ConvTranspose3d(c, c // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock3D(c, use_ckpt=gck) for _ in range(n)]))

        self.outro = nn.Conv3d(base, out_ch, 3, padding=1)

    def forward(self, lq):
        x = self.intro(lq)
        skips = []
        for enc, dn in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = dn(x)
        h_b = self.bottleneck(x)
        x = h_b
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)
        out = self.outro(x)
        pred = (lq + out).clamp(0, 1)
        return pred, h_b


# ---------- REPA projector (E2b's recipe) -----------------------------
class REPAProjector(nn.Module):
    def __init__(self, in_ch, out_dim=1024):
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


VJEPA_GRID = (8, 16, 16)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    net = NAFNet3D(base=args.base,
                   enc_blocks=tuple(args.enc_blocks),
                   dec_blocks=tuple(args.dec_blocks),
                   bottleneck_blocks=args.bn_blocks).to(device)
    n_n = sum(p.numel() for p in net.parameters() if p.requires_grad)

    projector = REPAProjector(in_ch=net.bottleneck_c, out_dim=1024).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    print(f"[NAFNet] V-JEPA encoder 304M frozen (REPA target only)")
    print(f"         NAFNet3D: {n_n/1e6:.2f}M trainable (base={args.base})")
    print(f"         REPAProjector: {n_p/1e6:.2f}M (training only)")
    print(f"         Loss: Charbonnier + {args.lambda_repa}·REPA "
          f"(HASTE off @ iter {args.early_stop_iter})")

    train_params = list(net.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr * 0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        net.load_state_dict(ckpt["net"])
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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, h_b = net(lq)
            L_rgb = charbonnier(pred.float(), gt)

            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                with torch.no_grad():
                    z_L = encode(backbone, lq)[args.ref_layer].float()
                z_stu = projector(h_b.float(), VJEPA_GRID)
                m = z_L.mean(dim=1, keepdim=True); s = z_L.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_L - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L = L_rgb + args.lambda_repa * L_repa
            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} "
                      f"L_rgb={L_rgb.item():.4f} L_repa={L_repa.item():.4f} "
                      f"dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(),
                            "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
                t0 = time.time()

        torch.save({"net": net.state_dict(), "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "nafnet_repa"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    print(f"\n=== EVAL: NAFNet-V-JEPA-REPA ===")
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
                save_compare_frame(os.path.join(viz_root, f"{name}.png"),
                                   lq, pred_f, gt)

    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "NAFNET_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# NAFNet-3D + V-JEPA-REPA\n\n")
        f.write(f"NAFNet3D {n_n/1e6:.2f}M trainable + REPAProjector {n_p/1e6:.2f}M (train-only)\n")
        f.write(f"V-JEPA encoder 304M frozen (REPA target only, NOT used at inference)\n")
        f.write(f"base={args.base}, enc={args.enc_blocks}, dec={args.dec_blocks}, bn={args.bn_blocks}\n\n")
        f.write(f"## SDSD held-out\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"vs clip_30k 20.746, vs v9.0b 20.578, vs E2b 19.556\n")
    print(f"\nNAFNet PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")
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
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--early_stop_iter", type=int, default=9000)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--enc_blocks", type=int, nargs="+", default=[2, 2, 4])
    ap.add_argument("--dec_blocks", type=int, nargs="+", default=[2, 2, 2])
    ap.add_argument("--bn_blocks", type=int, default=4)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    ap.add_argument("--save_viz", action="store_true",
                    help="Save LQ|Pred|GT comparison PNGs during eval")
    ap.add_argument("--viz_dir", default=None,
                    help="Override viz output dir (default: out_dir/viz/)")
    main(ap.parse_args())
