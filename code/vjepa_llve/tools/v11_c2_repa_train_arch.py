"""G1-G4: Architecture variants of REPA training.

--arch flag selects:
  default     : E2b baseline (single-scale, conv-3 projector, full-grad)
  multiscale  : align scale-3 (skips[3]→V-JEPA layer 2) AND scale-4 (h_b→layer 3)
  velocity    : align temporal differences (z[t+1]-z[t]) instead of static features
  frozen_proj : projector frozen at random init, only decoder gets alignment grad
  linear_proj : 1x1 conv projector (much smaller, less flexible)
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
    def __init__(self, in_ch=256, out_dim=1024, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=kernel_size, padding=pad)

    def forward(self, h, target_grid):
        x = self.conv(h)
        x = F.interpolate(x, size=target_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


def _norm_teacher(z):
    m = z.mean(dim=1, keepdim=True); s = z.std(dim=1, keepdim=True) + 1e-6
    return (z - m) / s


def _safe_cos_loss(z_stu, z_tea):
    z_stu_n = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
    z_tea_n = F.normalize(z_tea.float(), dim=-1, eps=1e-3)
    cos = (z_stu_n * z_tea_n).sum(dim=-1).clamp(-1.5, 1.5)
    return -cos.mean()


def _velocity(z, T_grid):
    """z: (B, T*HW, D); reshape to (B, T, HW, D) and diff along T → (B, (T-1)*HW, D)."""
    B, N, D = z.shape
    HW = N // T_grid
    z = z.view(B, T_grid, HW, D)
    dz = z[:, 1:, :, :] - z[:, :-1, :, :]            # (B, T-1, HW, D)
    return dz.reshape(B, (T_grid - 1) * HW, D)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    arch = args.arch
    Tg = decoder.vjepa_grid[0]   # 8

    # Build projector(s) per arch
    kernel = 1 if arch == "linear_proj" else 3
    if arch == "multiscale":
        # scale 4 → layer 3, scale 3 → layer 2
        proj_4 = REPAProjector(in_ch=decoder.dims[-1], out_dim=decoder.vjepa_dim, kernel_size=kernel).to(device)
        proj_3 = REPAProjector(in_ch=decoder.dims[-2], out_dim=decoder.vjepa_dim, kernel_size=kernel).to(device)
        projectors = nn.ModuleList([proj_4, proj_3])
    else:
        projectors = nn.ModuleList([
            REPAProjector(in_ch=decoder.dims[-1], out_dim=decoder.vjepa_dim, kernel_size=kernel).to(device)
        ])

    # Frozen-projector: only decoder gets alignment grad
    if arch == "frozen_proj":
        for p in projectors.parameters(): p.requires_grad_(False)

    n_p = sum(p.numel() for p in projectors.parameters())
    n_p_train = sum(p.numel() for p in projectors.parameters() if p.requires_grad)
    print(f"[REPA-arch={arch}] decoder {n_d/1e6:.2f}M + projector {n_p/1e6:.2f}M "
          f"(trainable {n_p_train/1e6:.2f}M); kernel={kernel}; "
          f"alignment to V-JEPA layer {args.ref_layer}, lambda={args.lambda_repa}, "
          f"early_stop_iter={args.early_stop_iter}")

    train_params = list(decoder.parameters()) + [p for p in projectors.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
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

        h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
        L_rgb = charbonnier(pred.float(), gt)

        L_repa = torch.zeros((), device=device)
        if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
            with torch.no_grad():
                z_L = encode(backbone, lq)

            if arch == "multiscale":
                # scale 4
                z_stu_4 = projectors[0](h_b, tuple(decoder.vjepa_grid))
                z_tea_4 = _norm_teacher(z_L[args.ref_layer].float())   # layer 3 by default
                # scale 3 (skips[3])
                z_stu_3 = projectors[1](skips[3], tuple(decoder.vjepa_grid))
                ref_layer_3 = max(0, args.ref_layer - 1)
                z_tea_3 = _norm_teacher(z_L[ref_layer_3].float())
                L_repa = 0.5 * (_safe_cos_loss(z_stu_4, z_tea_4) + _safe_cos_loss(z_stu_3, z_tea_3))

            elif arch == "velocity":
                z_stu = projectors[0](h_b, tuple(decoder.vjepa_grid))
                z_tea = _norm_teacher(z_L[args.ref_layer].float())
                # both reshape (B, T*HW, D); diff along T
                dz_stu = _velocity(z_stu, Tg)
                dz_tea = _velocity(z_tea, Tg)
                L_repa = _safe_cos_loss(dz_stu, dz_tea)

            else:  # default | frozen_proj | linear_proj
                z_stu = projectors[0](h_b, tuple(decoder.vjepa_grid))
                z_tea = _norm_teacher(z_L[args.ref_layer].float())
                L_repa = _safe_cos_loss(z_stu, z_tea)

        L = args.rgb_weight * L_rgb + args.lambda_repa * L_repa

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

    torch.save({"decoder": decoder.state_dict(),
                "projectors": projectors.state_dict(),
                "args": vars(args), "log": log, "stage": f"repa_arch_{arch}"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    decoder.eval()
    print(f"\n=== EVAL: SDSD held-out (arch={arch}) ===")
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
            print(f"    [{name}] LQ→pred = {p:.3f}")
    mean_p = float(np.mean(psnrs))
    md = os.path.join(args.out_dir, "REPA_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# REPA-arch={arch} Eval\n\n")
        f.write(f"decoder: {n_d/1e6:.2f}M (no V-JEPA at inference)\n")
        f.write(f"projector: {n_p/1e6:.2f}M (trainable {n_p_train/1e6:.2f}M)\n")
        f.write(f"arch: {arch}, kernel={kernel}, lambda={args.lambda_repa}, "
                f"ref_layer={args.ref_layer}, early_stop={args.early_stop_iter}, seed={args.seed}\n\n")
        f.write(f"## no_prior PSNR = {mean_p:.3f}  (vs E2b 19.556, vs pure_rgb 18.533, vs CLIP 19.230)\n")
    print(f"\nSDSD no_prior PSNR = {mean_p:.3f}  (arch={arch})")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arch", required=True,
                    choices=["default", "multiscale", "velocity", "frozen_proj", "linear_proj"])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--early_stop_iter", type=int, default=900)
    main(ap.parse_args())
