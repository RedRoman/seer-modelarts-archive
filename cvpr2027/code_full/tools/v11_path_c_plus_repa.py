"""PATH C + REPA combo: decoder uses kv=[z_L] at TRAIN+INFERENCE AND has REPA alignment loss.

The only V-JEPA-using combination we have not tested:
  - z_L as cross-attn KV at every train step AND inference (PATH C trick)
  - REPA cosine alignment of decoder bottleneck h_b → V-JEPA layer 3 (E2b trick)

Tests if "double dipping" beats either alone.
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


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    projector = REPAProjector(in_ch=decoder.dims[-1], out_dim=decoder.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())
    print(f"[PATH C+REPA] decoder {n_d/1e6:.2f}M + projector {n_p/1e6:.2f}M; "
          f"kv=[z_L] at TRAIN+INFER + REPA alignment to layer {args.ref_layer}; "
          f"lambda={args.lambda_repa}, early_stop={args.early_stop_iter}")

    optim = torch.optim.AdamW(list(decoder.parameters()) + list(projector.parameters()),
                              lr=args.lr, weight_decay=1e-4)
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

        with torch.no_grad():
            z_L = encode(backbone, lq)[args.ref_layer].float()

        kv_layers = [z_L]
        h_b, skips = decoder.encode(lq, kv_layers=kv_layers, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)
        L_rgb = charbonnier(pred.float(), gt)

        L_repa = torch.zeros((), device=device)
        if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
            z_stu = projector(h_b, tuple(decoder.vjepa_grid))
            z_tea = z_L.float()
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
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} L_rgb={L_rgb.item():.4f} "
                  f"L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item(),
                        "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
            t0 = time.time()

    torch.save({"decoder": decoder.state_dict(), "projector": projector.state_dict(),
                "args": vars(args), "log": log, "stage": "path_c_plus_repa"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    decoder.eval()
    print("\n=== EVAL: combined eval ===")
    psnrs_passive, psnrs_no_prior = [], []
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_L = encode(backbone, lq)[args.ref_layer].float()
            # passive_zL eval (canonical)
            kv = [z_L]
            h_b, skips = decoder.encode(lq, kv_layers=kv, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=kv, delta_layers=None)
            psnrs_passive.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
            # no_prior eval
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
            psnrs_no_prior.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
            print(f"    [{name}] passive={psnrs_passive[-1]:.3f} no_prior={psnrs_no_prior[-1]:.3f}")
    mean_pass = float(np.mean(psnrs_passive))
    mean_np = float(np.mean(psnrs_no_prior))
    md = os.path.join(args.out_dir, "PATH_CR_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# PATH C + REPA combo eval\n\n")
        f.write(f"decoder {n_d/1e6:.2f}M + projector {n_p/1e6:.2f}M (training-only)\n")
        f.write(f"V-JEPA used at TRAIN (kv=[z_L]+REPA loss) AND INFERENCE (kv=[z_L])\n\n")
        f.write(f"## Eval modes\n\n")
        f.write(f"| mode | mean PSNR |\n|---|---|\n")
        f.write(f"| passive_zL (canonical) | {mean_pass:.3f} |\n")
        f.write(f"| no_prior (drop kv at inference) | {mean_np:.3f} |\n\n")
        f.write(f"Compare: E2b={'19.556'} (REPA only), PATH C={'18.49'} (kv only), pure_rgb={'18.533'}\n")
    print(f"\nPATH C+REPA passive PSNR = {mean_pass:.3f}, no_prior = {mean_np:.3f}")
    print(f"  E2b 1500 (REPA only) = 19.556")
    print(f"  PATH C 1500 (kv only)= 18.49 mean")
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
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--early_stop_iter", type=int, default=900)
    main(ap.parse_args())
