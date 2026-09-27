"""E2b extended: REPA standalone with --dims (decoder scale) and --data (sdsd|bvi|both).

Inherits everything from v11_c2_repa_train: V-JEPA teacher, REPA cosine alignment,
HASTE early-stop, no V-JEPA at inference.

New flags:
  --dims "32,48,96,192,256"  : decoder dims (default = current 33M)
  --num_blocks "1,2,2,3,4"   : decoder block counts per scale
  --data sdsd|bvi|both       : training data source
  --bvi_root <path>          : BVI-RLV root (default cached)
  --bvi_eval                 : also eval on BVI-RLV held-out at end
"""
from __future__ import annotations
import argparse, os, random, sys, time
from typing import Tuple

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
from data.bvi_rlv_dataset import BVIRLVDataset


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


def _parse_tuple(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _bvi_sample(ds, rng):
    item = ds[int(rng.integers(0, len(ds)))]
    return item["lq"].unsqueeze(0), item["hq"].unsqueeze(0)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    dims = _parse_tuple(args.dims)
    num_blocks = _parse_tuple(args.num_blocks)
    decoder = DeltaModulatedDecoder(delta_in_encoder=False, dims=dims, num_blocks=num_blocks).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    projector = REPAProjector(in_ch=decoder.dims[-1], out_dim=decoder.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())
    print(f"[REPA-v2] decoder {n_d/1e6:.2f}M + projector {n_p/1e6:.2f}M; "
          f"dims={dims}, num_blocks={num_blocks}, data={args.data}, "
          f"alignment to V-JEPA layer {args.ref_layer}, lambda={args.lambda_repa}, "
          f"early_stop_iter={args.early_stop_iter}")

    optim = torch.optim.AdamW(list(decoder.parameters()) + list(projector.parameters()),
                              lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    sdsd_train, sdsd_test = build_split_seqs()
    bvi_train = bvi_test = None
    if args.data in ("bvi", "both") or args.bvi_eval:
        bvi_train = BVIRLVDataset(root=args.bvi_root, split="train", clip_len=16,
                                   crop_size=args.crop, is_train=True)
        bvi_test = BVIRLVDataset(root=args.bvi_root, split="test", clip_len=16,
                                  crop_size=args.crop, is_train=False)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)

    log = []
    t0 = time.time()
    for it in range(args.iters):
        # Pick source
        if args.data == "sdsd":
            src = "sdsd"
        elif args.data == "bvi":
            src = "bvi"
        else:  # both — alternate
            src = "sdsd" if (it % 2 == 0) else "bvi"

        if src == "sdsd":
            idx = int(rng.integers(0, len(sdsd_train)))
            _, lq_p, gt_p = sdsd_train[idx]
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        else:
            lq, gt = _bvi_sample(bvi_train, rng)
        lq, gt = lq.to(device), gt.to(device)

        h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
        L_rgb = charbonnier(pred.float(), gt)

        L_repa = torch.zeros((), device=device)
        if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
            with torch.no_grad():
                z_L = encode(backbone, lq)
            z_stu = projector(h_b, tuple(decoder.vjepa_grid))
            z_tea = z_L[args.ref_layer].float()
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
            print(f"  it {it+1}/{args.iters} src={src} L={L.item():.4f} "
                  f"L_rgb={L_rgb.item():.4f} L_repa={L_repa.item():.4f} "
                  f"dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "src": src, "L": L.item(),
                        "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
            t0 = time.time()

    torch.save({"decoder": decoder.state_dict(), "projector": projector.state_dict(),
                "args": vars(args), "log": log, "stage": "repa_v2"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    decoder.eval()
    print("\n=== EVAL: SDSD held-out (LLVE-STCD-12) ===")
    sdsd_psnrs = []
    with torch.no_grad():
        for name, lq_p, gt_p in sdsd_test:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
            p = mean_psnr(to_u8(pred.float()), to_u8(gt))
            sdsd_psnrs.append(p)
            print(f"    [{name}] LQ→pred = {p:.3f}")
    sdsd_mean = float(np.mean(sdsd_psnrs))

    bvi_mean = None
    if bvi_test is not None and args.bvi_eval:
        print("\n=== EVAL: BVI-RLV held-out ===")
        bvi_psnrs = []
        with torch.no_grad():
            for i in range(len(bvi_test)):
                item = bvi_test[i]
                lq = item["lq"].unsqueeze(0).to(device)
                gt = item["hq"].unsqueeze(0).to(device)
                h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
                p = mean_psnr(to_u8(pred.float()), to_u8(gt))
                bvi_psnrs.append(p)
                print(f"    [BVI {i}] LQ→pred = {p:.3f}")
        bvi_mean = float(np.mean(bvi_psnrs))

    md = os.path.join(args.out_dir, "REPA_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# REPA-v2 Eval\n\n")
        f.write(f"decoder: {n_d/1e6:.2f}M (dims={dims}, num_blocks={num_blocks})\n")
        f.write(f"projector: {n_p/1e6:.2f}M (training-only)\n")
        f.write(f"data: {args.data}\n")
        f.write(f"lambda_repa: {args.lambda_repa}, ref_layer: {args.ref_layer}, "
                f"early_stop_iter: {args.early_stop_iter}, seed: {args.seed}\n\n")
        f.write(f"## no_prior PSNR = {sdsd_mean:.3f}  (SDSD held-out, vs pure_rgb 18.533, vs E2b 19.556)\n")
        if bvi_mean is not None:
            f.write(f"## BVI-RLV PSNR = {bvi_mean:.3f}\n")
    print(f"\nSDSD no_prior PSNR = {sdsd_mean:.3f}")
    if bvi_mean is not None:
        print(f"BVI-RLV PSNR     = {bvi_mean:.3f}")
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
    ap.add_argument("--dims", type=str, default="32,48,96,192,256")
    ap.add_argument("--num_blocks", type=str, default="1,2,2,3,4")
    ap.add_argument("--data", type=str, default="sdsd", choices=["sdsd", "bvi", "both"])
    ap.add_argument("--bvi_root", type=str, default="/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos")
    ap.add_argument("--bvi_eval", action="store_true")
    main(ap.parse_args())
