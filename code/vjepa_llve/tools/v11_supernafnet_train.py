"""SuperNAFNet: Frankenstein combining validated modules.

Modules borrowed:
  - NAFNet 3D backbone (NAFNet 15k = 20.436, BEST perceptual: LPIPS=0.264)
  - Bottleneck temporal attention (NEW — fixes NAFNet's spatial-only kernel)
  - V-JEPA cross-attn at bottleneck (PATH C: V-JEPA at inference, not just REPA target)
  - Multi-teacher REPA: CLIP-B/8 + DINOv2-B/8 in parallel (each individually validated)
  - EMA on weights, decay=0.999

Hypothesis: combining best modules synergizes — NAFNet wins perceptual,
CLIP+DINOv2 multi-teacher complementary signals, V-JEPA cross-attn adds
inference-time semantic conditioning, temporal attn fixes NAFNet's hole.
"""
from __future__ import annotations
import argparse, copy, os, sys, time

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
from tools.v11_nafnet_train import NAFBlock3D, LayerNorm3D, safe_cos_loss
from tools.v11_nafnet_pp_train import EMA, lr_warmup_cosine
from tools.v11_c2_repa_train_clip import encode_clip, REPAProjectorCLIP, CLIP_DIM, CLIP_GRID
from tools.v11_dinov2_repa_train import encode_dino, REPAProjectorDINO, DINO_DIM, DINO_GRID


# ---------- bottleneck modules ---------------------------------------
class TemporalAttn(nn.Module):
    """Self-attention over T tokens at each spatial location.
    Input: (B, C, T, H, W). Output: same shape, residual-added."""
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads; self.h_dim = c // heads
        self.norm = LayerNorm3D(c)
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        nn.init.zeros_(self.proj.weight)  # zero-init residual
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1, 1))

    def forward(self, x):
        h = self.norm(x)                                                  # (B, C, T, H, W)
        B, C, T, H, W = h.shape
        h = h.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)             # (B*H*W, T, C)
        qkv = self.qkv(h).reshape(B * H * W, T, 3, self.heads, self.h_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B * H * W, T, C)
        out = self.proj(out)
        out = out.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)            # (B, C, T, H, W)
        return x + self.gamma * out


class VJepaCrossAttn(nn.Module):
    """Cross-attn: Q from bottleneck h_b, K/V from V-JEPA features.
    Input h_b: (B, C, T, H, W), z_vj: (B, N, D_vj). Output: same as h_b."""
    def __init__(self, c, vj_dim=1024, heads=4):
        super().__init__()
        self.heads = heads; self.h_dim = c // heads
        self.norm_q = LayerNorm3D(c)
        self.kv_proj = nn.Linear(vj_dim, 2 * c, bias=False)
        self.q_proj = nn.Linear(c, c, bias=False)
        self.out = nn.Linear(c, c, bias=False)
        nn.init.zeros_(self.out.weight)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1, 1))

    def forward(self, h_b, z_vj):
        x = self.norm_q(h_b)                                              # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)             # (B, N_h, C)
        q = self.q_proj(x).reshape(B, T * H * W, self.heads, self.h_dim).transpose(1, 2)
        # Project V-JEPA tokens to bottleneck dim
        kv = self.kv_proj(z_vj)                                           # (B, N_vj, 2C)
        kv = kv.reshape(B, -1, 2, self.heads, self.h_dim).permute(2, 0, 3, 1, 4)
        k, v_ = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v_)
        out = out.transpose(1, 2).reshape(B, T * H * W, C)
        out = self.out(out)
        out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return h_b + self.gamma * out


# ---------- model -----------------------------------------------------
class SuperNAFNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=96,
                 enc_blocks=(2, 2, 4), dec_blocks=(2, 2, 2),
                 bottleneck_blocks=4, use_grad_ckpt=True,
                 use_temporal_attn=True, use_vjepa_cross=True, vj_dim=1024):
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
        # NEW modules
        self.temporal_attn = TemporalAttn(c) if use_temporal_attn else nn.Identity()
        self.vjepa_cross = VJepaCrossAttn(c, vj_dim=vj_dim) if use_vjepa_cross else None
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.ConvTranspose3d(c, c // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock3D(c, use_ckpt=gck) for _ in range(n)]))
        self.outro = nn.Conv3d(base, out_ch, 3, padding=1)

    def forward(self, lq, z_vjepa=None):
        x = self.intro(lq)
        skips = []
        for enc, dn in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = dn(x)
        h_b = self.bottleneck(x)
        h_b = self.temporal_attn(h_b)                                      # NEW
        if self.vjepa_cross is not None and z_vjepa is not None:
            h_b = self.vjepa_cross(h_b, z_vjepa)                           # NEW
        x = h_b
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x); x = x + skip; x = dec(x)
        out = self.outro(x)
        pred = (lq + out).clamp(0, 1)
        return pred, h_b


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    from transformers import CLIPVisionModel, Dinov2Model
    clip_t = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir="/cache/zihaohe/hf_cache"
    ).to(device).eval()
    for p in clip_t.parameters(): p.requires_grad_(False)
    dino_t = Dinov2Model.from_pretrained(
        "facebook/dinov2-base", cache_dir="/cache/zihaohe/hf_cache"
    ).to(device).eval()
    for p in dino_t.parameters(): p.requires_grad_(False)

    net = SuperNAFNet(base=args.base,
                      enc_blocks=tuple(args.enc_blocks),
                      dec_blocks=tuple(args.dec_blocks),
                      bottleneck_blocks=args.bn_blocks,
                      use_temporal_attn=True, use_vjepa_cross=True).to(device)
    n_n = sum(p.numel() for p in net.parameters() if p.requires_grad)

    proj_clip = REPAProjectorCLIP(in_ch=net.bottleneck_c, out_dim=CLIP_DIM).to(device)
    proj_dino = REPAProjectorDINO(in_ch=net.bottleneck_c, out_dim=DINO_DIM).to(device)
    n_p = sum(p.numel() for p in proj_clip.parameters()) + sum(p.numel() for p in proj_dino.parameters())

    print(f"[SuperNAFNet] V-JEPA encoder 304M frozen (cross-attn + REPA target)")
    print(f"              CLIP-B/32 87M + DINOv2-B 86M frozen (REPA targets)")
    print(f"              SuperNAFNet: {n_n/1e6:.2f}M trainable")
    print(f"              Projectors (CLIP+DINOv2): {n_p/1e6:.2f}M training-only")
    print(f"              Modules: NAFNet + temporal-attn + V-JEPA cross-attn + multi-teacher REPA + EMA")

    train_params = list(net.parameters()) + list(proj_clip.parameters()) + list(proj_dino.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt_d = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        net.load_state_dict(ckpt_d["net"])
        ema = EMA(net, decay=args.ema_decay)
        if "ema_shadow" in ckpt_d:
            ema.shadow = {k: v.to(device) for k, v in ckpt_d["ema_shadow"].items()}
        proj_clip.load_state_dict(ckpt_d["proj_clip"])
        proj_dino.load_state_dict(ckpt_d["proj_dino"])
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

            L_repa_c = torch.zeros((), device=device)
            L_repa_d = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                # CLIP REPA
                z_tea_c = encode_clip(clip_t, lq, args.clip_layer).float()
                z_stu_c = proj_clip(h_b.float(), CLIP_GRID)
                m = z_tea_c.mean(dim=1, keepdim=True); s = z_tea_c.std(dim=1, keepdim=True) + 1e-6
                z_tea_c = (z_tea_c - m) / s
                L_repa_c = safe_cos_loss(z_stu_c, z_tea_c)
                # DINOv2 REPA
                z_tea_d = encode_dino(dino_t, lq, args.dino_layer).float()
                z_stu_d = proj_dino(h_b.float(), DINO_GRID)
                m = z_tea_d.mean(dim=1, keepdim=True); s = z_tea_d.std(dim=1, keepdim=True) + 1e-6
                z_tea_d = (z_tea_d - m) / s
                L_repa_d = safe_cos_loss(z_stu_d, z_tea_d)

            L = L_rgb + args.lambda_repa * (L_repa_c + L_repa_d)
            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step()
            ema.update(net)

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} lr={cur_lr:.2e} L={L.item():.4f} "
                      f"L_rgb={L_rgb.item():.4f} L_clip={L_repa_c.item():.4f} "
                      f"L_dino={L_repa_d.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "lr": cur_lr, "L": L.item(),
                            "L_rgb": L_rgb.item(), "L_clip": L_repa_c.item(),
                            "L_dino": L_repa_d.item()})
                t0 = time.time()

        torch.save({"net": net.state_dict(),
                    "proj_clip": proj_clip.state_dict(),
                    "proj_dino": proj_dino.state_dict(),
                    "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
                    "args": vars(args), "log": log, "stage": "supernafnet"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Eval with EMA weights
    print(f"\n=== EVAL: SuperNAFNet (EMA weights, V-JEPA active at inference) ===")
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
    md = os.path.join(args.out_dir, "SUPERNAFNET_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# SuperNAFNet (Frankenstein: NAFNet + temporal + V-JEPA cross + multi-teacher REPA + EMA)\n\n")
        f.write(f"SuperNAFNet {n_n/1e6:.2f}M trainable; CLIP+DINOv2+V-JEPA frozen\n")
        f.write(f"Training: warmup {args.warmup_iters} + cosine, EMA={args.ema_decay}, iters={args.iters}\n\n")
        f.write(f"## SDSD held-out (EMA + V-JEPA active at inference)\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"vs nafnet_15k 20.436, vs clip_30k 20.746\n")
    print(f"\nSuperNAFNet PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_layer", type=int, default=3, help="V-JEPA layer for cross-attn + REPA")
    ap.add_argument("--clip_layer", type=int, default=8)
    ap.add_argument("--dino_layer", type=int, default=8)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--early_stop_iter", type=int, default=18000)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--enc_blocks", type=int, nargs="+", default=[2, 2, 4])
    ap.add_argument("--dec_blocks", type=int, nargs="+", default=[2, 2, 2])
    ap.add_argument("--bn_blocks", type=int, default=4)
    ap.add_argument("--warmup_iters", type=int, default=500)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--save_viz", action="store_true",
                    help="Save MP4 LQ|Pred|GT comparison videos during eval")
    ap.add_argument("--viz_dir", default=None,
                    help="Override viz output dir (default: out_dir/viz/)")
    main(ap.parse_args())
