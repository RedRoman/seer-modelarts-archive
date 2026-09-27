"""F5: REPA with CLIP-ViT-B/32 teacher (image SSL, per-frame).

Tests "is video SSL necessary, vs image SSL?" — key paper ablation against E2b.

Uses HuggingFace transformers CLIPVisionModel (already cached at
/cache/zihaohe/hf_cache/hub/models--openai--clip-vit-base-patch32).

For each clip (B, 3, T, H, W):
  - Per-frame pass through CLIP vision encoder, request hidden_states.
  - Pick layer args.ref_layer; tokens are (B*T, 1+P, 768) → drop CLS → (B, T, P, 768).
  - At 256x256 input, CLIP-B/32 patch size = 32 → grid 8x8 = 64 tokens per frame.
  - Treat as a 3D grid (T, 8, 8); align decoder bottleneck h_b to this grid.
"""
from __future__ import annotations
import argparse, os, sys, time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.iqa_metrics import compute_iqa, fmt_iqa
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr,
                                           charbonnier, build_split_seqs)


CLIP_DIM = 768
CLIP_GRID = (16, 8, 8)  # (T, H, W) for 16-frame clip at 256 input, patch 32


class REPAProjectorCLIP(nn.Module):
    def __init__(self, in_ch=256, out_dim=CLIP_DIM):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)

    def forward(self, h_b, target_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=target_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


@torch.no_grad()
def encode_clip(clip_vision, vid, ref_layer):
    """vid: (B, 3, T, H, W) in [0,1]; returns (B, T*P, 768) at layer ref_layer."""
    B, C, T, H, W = vid.shape
    # CLIP expects (B, 3, H, W), values normalized with CLIP mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=vid.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=vid.device).view(1, 3, 1, 1)
    frames = vid.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    frames = (frames - mean) / std
    out = clip_vision(pixel_values=frames, output_hidden_states=True,
                      interpolate_pos_encoding=True)
    h = out.hidden_states[ref_layer]              # (B*T, 1+P, D)
    h = h[:, 1:, :]                               # drop CLS → (B*T, P, D)
    P = h.size(1)
    h = h.reshape(B, T * P, h.size(-1))           # (B, T*P, D)
    return h


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import CLIPVisionModel
    clip_vision = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir="/cache/zihaohe/hf_cache",
    ).to(device).eval()
    for p in clip_vision.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    projector = REPAProjectorCLIP(in_ch=decoder.dims[-1], out_dim=CLIP_DIM).to(device)
    n_p = sum(p.numel() for p in projector.parameters())
    print(f"[REPA-CLIP] decoder {n_d/1e6:.2f}M + projector {n_p/1e6:.2f}M; "
          f"teacher=CLIP-ViT-B/32, target_grid={CLIP_GRID}, "
          f"alignment to layer {args.ref_layer}, lambda={args.lambda_repa}, "
          f"early_stop_iter={args.early_stop_iter}")

    optim = torch.optim.AdamW(list(decoder.parameters()) + list(projector.parameters()),
                              lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

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
            _, lq_p, gt_p = train_seqs[idx]
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)

            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
            L_rgb = charbonnier(pred.float(), gt)

            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                z_tea = encode_clip(clip_vision, lq, args.ref_layer).float()       # (B, N, 768)
                z_stu = projector(h_b, CLIP_GRID)                                   # (B, N, 768)
                m = z_tea.mean(dim=1, keepdim=True); s = z_tea.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_tea - m) / s
                z_stu_n = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
                z_tea_n = F.normalize(z_tea, dim=-1, eps=1e-3)
                cos = (z_stu_n * z_tea_n).sum(dim=-1).clamp(-1.5, 1.5)
                L_repa = -cos.mean()

            L = args.rgb_weight * L_rgb + args.lambda_repa * L_repa
            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + list(projector.parameters()), 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} "
                      f"L_rgb={L_rgb.item():.4f} L_repa={L_repa.item():.4f} "
                      f"dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(),
                            "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
                t0 = time.time()

        torch.save({"decoder": decoder.state_dict(), "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "repa_clip"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    decoder.eval()
    print("\n=== EVAL: SDSD held-out ===")
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
    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "REPA_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# REPA-CLIP teacher Eval\n\n")
        f.write(f"teacher: CLIP-ViT-B/32 layer {args.ref_layer} (image SSL)\n")
        f.write(f"decoder: {n_d/1e6:.2f}M (no V-JEPA / no CLIP at inference)\n")
        f.write(f"lambda: {args.lambda_repa}, early_stop: {args.early_stop_iter}, seed: {args.seed}\n\n")
        f.write(f"## SDSD held-out\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"vs E2b 19.556 video-SSL, vs pure_rgb 18.533\n")
    print(f"\nSDSD PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}  (CLIP teacher)")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--ref_layer", type=int, default=8, help="CLIP-B/32 has 12 layers; ~mid is 6-8")
    ap.add_argument("--early_stop_iter", type=int, default=900)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    main(ap.parse_args())
