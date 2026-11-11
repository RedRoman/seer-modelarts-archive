"""DINOv2-B REPA teacher: image-only SSL teacher (no text).

Tests teacher diversity: clip uses image-text contrastive (CLIP-B/32 87M),
dinov2 uses image-only self-distillation (DINOv2-B/14 86M). Same model size,
different SSL paradigm. Cleanest comparison of "what kind of image SSL features
help LLVE most".

Per-frame DINOv2 forward at resized 224x224 (matches DINOv2 native patch grid:
224/14 = 16 patches per side → 16x16 = 256 tokens per frame).
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use cached only

from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.iqa_metrics import compute_iqa, fmt_iqa
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr,
                                           charbonnier, build_split_seqs)


DINO_DIM = 768
DINO_GRID = (16, 16, 16)  # (T, H, W) for 16-frame clip resized to 224, patch 14


class REPAProjectorDINO(nn.Module):
    def __init__(self, in_ch=256, out_dim=DINO_DIM):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)

    def forward(self, h_b, target_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=target_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


@torch.no_grad()
def encode_dino(dino_vision, vid, ref_layer):
    """vid: (B, 3, T, H, W) in [0,1]; returns (B, T*P=4096, 768) at layer ref_layer.
    Resizes to 224 internally for DINOv2 patch alignment.
    """
    B, C, T, H, W = vid.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=vid.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=vid.device).view(1, 3, 1, 1)
    frames = vid.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    if H != 224 or W != 224:
        frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
    frames = (frames - mean) / std
    out = dino_vision(pixel_values=frames, output_hidden_states=True)
    h = out.hidden_states[ref_layer]              # (B*T, 1+P, D)
    h = h[:, 1:, :]                               # drop CLS → (B*T, P=256, D)
    P = h.size(1)
    h = h.reshape(B, T * P, h.size(-1))           # (B, T*P, D)
    return h


def safe_cos_loss(z_stu, z_tea):
    z_stu = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
    z_tea = F.normalize(z_tea.float(), dim=-1, eps=1e-3)
    cos = (z_stu * z_tea).sum(dim=-1).clamp(-1.5, 1.5)
    return -cos.mean()


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import Dinov2Model
    dino_vision = Dinov2Model.from_pretrained(
        "facebook/dinov2-base",
        cache_dir="/cache/zihaohe/hf_cache",
    ).to(device).eval()
    for p in dino_vision.parameters(): p.requires_grad_(False)
    n_dino = sum(p.numel() for p in dino_vision.parameters())
    print(f"[DINOv2] teacher loaded: {n_dino/1e6:.1f}M params, layer {args.ref_layer}, "
          f"target_grid={DINO_GRID}, dim={DINO_DIM}")

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    projector = REPAProjectorDINO(in_ch=decoder.dims[-1], out_dim=DINO_DIM).to(device)
    n_p = sum(p.numel() for p in projector.parameters())
    print(f"[REPA-DINO] decoder {n_d/1e6:.2f}M + projector {n_p/1e6:.2f}M; "
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
                z_tea = encode_dino(dino_vision, lq, args.ref_layer).float()
                z_stu = projector(h_b, DINO_GRID)
                m = z_tea.mean(dim=1, keepdim=True); s = z_tea.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_tea - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

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
                    "args": vars(args), "log": log, "stage": "repa_dinov2"},
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
        f.write(f"# REPA-DINOv2 teacher Eval\n\n")
        f.write(f"teacher: DINOv2-B/14 layer {args.ref_layer} (image-only SSL)\n")
        f.write(f"decoder: {n_d/1e6:.2f}M (no DINOv2 at inference)\n")
        f.write(f"lambda: {args.lambda_repa}, early_stop: {args.early_stop_iter}, seed: {args.seed}\n\n")
        f.write(f"## SDSD held-out\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"vs CLIP30k 20.605 (mean), vs E2b30k 20.446 (mean), vs v9.0b 20.578\n")
    print(f"\nDINOv2 PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--ref_layer", type=int, default=8, help="DINOv2-base has 12 layers; 8 ~ middle")
    ap.add_argument("--early_stop_iter", type=int, default=9000)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    main(ap.parse_args())
