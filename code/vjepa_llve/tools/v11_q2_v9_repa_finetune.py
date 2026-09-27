"""Q2: v9.0b finetune + low-weight REPA loss on bottleneck.

Loads strong v9.0b ckpt (which already gets 20.578 zero-shot), adds REPA
cosine alignment of pixel_branch's cfim_bottleneck feature to V-JEPA layer 3.
Tests if REPA can push v9.0b past 21+ dB.

Per DBL5/DBM6:
  lr=2e-5 (matches v9.0b finetune setting)
  lambda_repa=0.05 (low to avoid disturbing strong baseline)
  early_stop_iter=1500 (HASTE-style; REPA off after iter 1500)
  iters=3000
  kill gates at iter 500/1000/3000 per DBL5
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

from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import load_config
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr,
                                           charbonnier, build_split_seqs)


class REPAProjector(nn.Module):
    """3D conv projector aligning pixel_branch bottleneck (5D, dims[-1]=1024)
    to V-JEPA layer 3 grid (B, N=2048, 1024)."""
    def __init__(self, in_ch=1024, out_dim=1024, vjepa_grid=(8, 16, 16)):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)
        self.vjepa_grid = vjepa_grid

    def forward(self, x):
        # x is (B, C=1024, T', H', W') from cfim_bottleneck output
        # cfim_bottleneck operates per-frame (2D) on flattened (B*T, C, H, W)
        # The hook captures (B*T, C, H, W); we need to reshape back to (B, C, T, H, W)
        x = self.conv(x)
        x = F.interpolate(x, size=self.vjepa_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


def safe_cos_loss(z_stu, z_tea):
    z_stu = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
    z_tea = F.normalize(z_tea.float(), dim=-1, eps=1e-3)
    cos = (z_stu * z_tea).sum(dim=-1).clamp(-1.5, 1.5)
    return -cos.mean()


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(args.cfg, overrides=[])
    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp = model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    print(f"[Q2] loaded {args.ckpt} | missing={len(miss)} unexpected={len(unexp)}")
    n_pb = sum(p.numel() for p in model.pixel_branch.parameters() if p.requires_grad)

    # Hook the cfim_bottleneck output (deepest pixel_branch feature)
    bottleneck_buf = []
    def _hook(module, inputs, output):
        # cfim_bottleneck input/output is (B*T, C=1024, H', W')
        bottleneck_buf.clear()
        bottleneck_buf.append(output)
    if model.pixel_branch.cfim_bottleneck is None:
        raise RuntimeError("v9 pixel_branch has no cfim_bottleneck; check config")
    h_handle = model.pixel_branch.cfim_bottleneck.register_forward_hook(_hook)

    projector = REPAProjector(in_ch=1024, out_dim=1024, vjepa_grid=(8, 16, 16)).to(device)
    n_p = sum(p.numel() for p in projector.parameters())
    print(f"[Q2] pixel_branch {n_pb/1e6:.1f}M trainable + projector {n_p/1e6:.2f}M")
    print(f"     V-JEPA encoder 304M frozen; lambda_repa={args.lambda_repa}, "
          f"early_stop={args.early_stop_iter}, lr={args.lr}, iters={args.iters}")

    train_params = [p for p in model.pixel_branch.parameters() if p.requires_grad] + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.1)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    best_psnr = -1.0
    t0 = time.time()
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        _, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(lq)
        pred = out["pred_pixels"].clamp(0, 1)
        L_rgb = charbonnier(pred.float(), gt)

        L_repa = torch.zeros((), device=device)
        if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
            bn = bottleneck_buf[0]
            # cfim_bottleneck is called in 5D space: (B, C, T, H, W). Handle both shapes.
            if bn.ndim == 5:
                bn5 = bn
            elif bn.ndim == 4:
                BT, C, H, W = bn.shape
                B = lq.shape[0]; T = BT // B
                bn5 = bn.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
            else:
                raise RuntimeError(f"unexpected bottleneck shape {bn.shape}")
            z_stu = projector(bn5)
            z_tea = out["e_lq_layers"][args.ref_layer].float()
            m = z_tea.mean(dim=1, keepdim=True); s = z_tea.std(dim=1, keepdim=True) + 1e-6
            z_tea = (z_tea - m) / s
            L_repa = safe_cos_loss(z_stu, z_tea)

        L = L_rgb + args.lambda_repa * L_repa
        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} L_rgb={L_rgb.item():.4f} L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
            t0 = time.time()

        # Mid-eval at kill-gate checkpoints per DBL5
        if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            psnr = _quick_eval(model, test_seqs, device, args.crop)
            print(f"  [@iter {it+1}] held-out PSNR = {psnr:.3f}")
            best_psnr = max(best_psnr, psnr)
            # Kill gates per DBL5
            kill = False
            if (it+1) >= 500 and psnr < 20.30:
                print(f"  [KILL GATE] iter {it+1} PSNR {psnr:.3f} < 20.30 → abort")
                kill = True
            if kill:
                _write_md(args.out_dir, args, log, best_psnr, n_pb, n_p, aborted=True)
                sys.exit(2)

    h_handle.remove()
    torch.save({"pixel_branch": model.pixel_branch.state_dict(),
                "projector": projector.state_dict(),
                "args": vars(args), "log": log, "stage": "v9_repa_finetune_q2"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Final eval
    print("\n=== FINAL EVAL ===")
    psnrs = _quick_eval(model, test_seqs, device, args.crop, verbose=True, return_list=True)
    mean_p = float(np.mean(psnrs))
    best_psnr = max(best_psnr, mean_p)
    _write_md(args.out_dir, args, log, mean_p, n_pb, n_p, aborted=False, per_seq=psnrs)
    print(f"\nQ2 final PSNR = {mean_p:.3f}  (vs v9.0b zero-shot 20.578, gate >=21.0)")


@torch.no_grad()
def _quick_eval(model, test_seqs, device, crop, verbose=False, return_list=False):
    model.eval()
    psnrs = []
    for name, lq_p, gt_p in test_seqs:
        lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
        lq, gt = lq.to(device), gt.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.sample(lq)
        pred = out["pred_pixels"].clamp(0, 1).float()
        p = mean_psnr(to_u8(pred), to_u8(gt))
        psnrs.append(p)
        if verbose: print(f"    [{name}] {p:.3f}")
    model.train()
    if model.backbone is not None: model.backbone.eval()
    if return_list: return psnrs
    return float(np.mean(psnrs))


def _write_md(out_dir, args, log, mean_p, n_pb, n_p, aborted, per_seq=None):
    md = os.path.join(out_dir, "Q2_EVAL.md" if not aborted else "Q2_ABORT.md")
    with open(md, "w") as f:
        f.write(f"# Q2: v9.0b + REPA finetune\n\n")
        f.write(f"ckpt source: {args.ckpt}\n")
        f.write(f"pixel_branch {n_pb/1e6:.1f}M + projector {n_p/1e6:.2f}M\n")
        f.write(f"lambda_repa={args.lambda_repa}, lr={args.lr}, iters={args.iters}, "
                f"early_stop={args.early_stop_iter}, seed={args.seed}\n\n")
        f.write(f"## final PSNR = {mean_p:.3f}\n")
        f.write(f"vs v9.0b zero-shot baseline 20.578\n")
        f.write(f"DBL5 mainline gate: PSNR >= 21.0  -> {'PASS' if mean_p >= 21.0 else 'FAIL'}\n")
        if per_seq:
            f.write(f"\nper-seq: {per_seq}\n")
        if aborted:
            f.write(f"\nABORTED at iter {log[-1]['iter'] if log else 'N/A'} due to kill gate\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_v9_0b.yaml")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_repa", type=float, default=0.05)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--early_stop_iter", type=int, default=1500, help="HASTE-style early stop")
    main(ap.parse_args())
