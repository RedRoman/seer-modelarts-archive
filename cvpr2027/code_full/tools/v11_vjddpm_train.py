"""VJ-DDPM (v4): conditional DDPM in V-JEPA latent space.

Different from VJLF v1 (rectified flow) and VJ-Mamba v1 / VJ-Predictor-Flow v1:
this is a proper variance-preserving DDPM with eps-prediction.

Forward (training):
  sample t ~ U(1, T_diff=1000)
  alpha_bar_t = cumprod(1 - betas)[t]
  eps ~ N(0, I)
  z_t = sqrt(alpha_bar_t) · z_HQ + sqrt(1 - alpha_bar_t) · eps
  cond_input = concat(z_t, z_LQ) along channel dim
  eps_pred = DiT(cond_input, t)
  L_diff = MSE(eps_pred, eps)

Inference (DDIM 50 steps):
  start z_T = z_LQ + small noise (warm start, conditioning)
  for k in T..1: z_{k-1} = ddim_step(z_k, eps_pred(z_k, k))
  z_HQ_pred = z_0

Render head + REPA + HASTE same as VJLF v1.
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
from models.latent_dit import VJepaLatentDiT
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.iqa_metrics import compute_iqa, fmt_iqa
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


class CondDiTWrapper(nn.Module):
    """Wrap VJepaLatentDiT to take (z_t, z_LQ, t) and output eps prediction.
    Use a 1024->1024 input projection on concat(z_t, z_LQ)=2048-d."""
    def __init__(self, dit_dim=512, depth=12, num_heads=8):
        super().__init__()
        # Project concat(z_t, z_LQ) (2048 dims) down to vjepa_dim (1024) for DiT input
        self.cond_proj = nn.Linear(2 * 1024, 1024, bias=True)
        nn.init.zeros_(self.cond_proj.bias)
        # Identity-at-init: project z_t through (zero out z_LQ part of weight)
        with torch.no_grad():
            self.cond_proj.weight.data[:, :1024] = torch.eye(1024)
            self.cond_proj.weight.data[:, 1024:] = 0.0

        self.dit = VJepaLatentDiT(
            vjepa_dim=1024, grid=(8, 16, 16),
            dim=dit_dim, depth=depth, num_heads=num_heads,
            ffn_mult=4.0, use_gradient_checkpoint=True,
        )

    def forward(self, z_t, z_LQ, t):
        x = torch.cat([z_t, z_LQ], dim=-1)
        x = self.cond_proj(x)
        return self.dit(x, t)


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


def safe_cos_loss(z_stu, z_tea):
    z_stu = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
    z_tea = F.normalize(z_tea.float(), dim=-1, eps=1e-3)
    cos = (z_stu * z_tea).sum(dim=-1).clamp(-1.5, 1.5)
    return -cos.mean()


def make_betas(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def ddim_sample(model, z_LQ, alphas_bar, num_inference_steps=50, eta=0.0):
    """DDIM sampling. Start from z_LQ + eps for restoration warm-start.
    z_LQ: (B, N, 1024). Returns z_HQ_pred (B, N, 1024)."""
    T_full = alphas_bar.shape[0]
    timesteps = torch.linspace(T_full - 1, 0, num_inference_steps + 1).long()
    # Warm start: z_T = sqrt(alpha_T) * z_LQ + sqrt(1-alpha_T) * eps
    eps0 = torch.randn_like(z_LQ)
    a_T = alphas_bar[timesteps[0]]
    z = a_T.sqrt() * z_LQ + (1 - a_T).sqrt() * eps0

    for i in range(num_inference_steps):
        t_cur = timesteps[i].item()
        t_next = timesteps[i + 1].item()
        a_cur = alphas_bar[t_cur]
        a_next = alphas_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=z.device)
        t_tensor = torch.full((z.shape[0],), t_cur, device=z.device, dtype=torch.float32) / T_full
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eps_pred = model(z, z_LQ, t_tensor).float()
        # x0 prediction
        x0_pred = (z - (1 - a_cur).sqrt() * eps_pred) / a_cur.sqrt()
        # DDIM deterministic (eta=0): z_next = sqrt(a_next)*x0 + sqrt(1-a_next)*eps_pred
        z = a_next.sqrt() * x0_pred + (1 - a_next).sqrt() * eps_pred
    return z


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    ddpm = CondDiTWrapper(dit_dim=args.dit_dim, depth=args.dit_depth, num_heads=args.dit_heads).to(device)
    n_d = sum(p.numel() for p in ddpm.parameters() if p.requires_grad)

    render = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_r = sum(p.numel() for p in render.parameters() if p.requires_grad)

    projector = REPAProjector(in_ch=render.dims[-1], out_dim=render.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    # Diffusion schedule
    betas = make_betas(T=args.T_diff).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    print(f"[VJ-DDPM] V-JEPA encoder 304M frozen")
    print(f"          CondDiT: {n_d/1e6:.2f}M trainable (dim={args.dit_dim} depth={args.dit_depth})")
    print(f"          Render: {n_r/1e6:.2f}M, REPA proj: {n_p/1e6:.2f}M")
    print(f"          Total trainable: {(n_d+n_r+n_p)/1e6:.2f}M")
    print(f"          DDPM: T_diff={args.T_diff}, eps-prediction, DDIM {args.eval_steps}-step inference")
    print(f"          Loss: L_eps + {args.lambda_pixel}·L_pixel + {args.lambda_repa}·L_repa "
          f"(HASTE @ iter {args.early_stop_iter})")

    train_params = list(ddpm.parameters()) + list(render.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        ddpm.load_state_dict(ckpt["ddpm"])
        render.load_state_dict(ckpt["render"])
        projector.load_state_dict(ckpt["projector"])
        print(f"[eval-only] loaded ckpt → {args.out_dir}/ckpt.pt")
    else:
        t0 = time.time()
        for it in range(args.iters):
            idx = int(rng.integers(0, len(train_seqs)))
            _, lq_p, gt_p = train_seqs[idx]
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)

            with torch.no_grad():
                z_LQ = encode(backbone, lq)[args.ref_layer].float()
                z_HQ = encode(backbone, gt)[args.ref_layer].float()

            B = z_LQ.shape[0]
            t_int = torch.randint(0, args.T_diff, (B,), device=device)
            a_t = alphas_bar[t_int].view(B, 1, 1)
            eps = torch.randn_like(z_HQ)
            z_t_d = a_t.sqrt() * z_HQ + (1 - a_t).sqrt() * eps
            t_norm = t_int.float() / args.T_diff

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eps_pred = ddpm(z_t_d, z_LQ, t_norm)
            L_eps = F.mse_loss(eps_pred.float(), eps)

            kv_layers = [z_HQ]
            h_b, skips = render.encode(lq, kv_layers=kv_layers, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = render.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)
            L_pixel = charbonnier(pred.float(), gt)

            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                z_stu = projector(h_b, tuple(render.vjepa_grid))
                m = z_HQ.mean(dim=1, keepdim=True); s = z_HQ.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_HQ - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L = L_eps + args.lambda_pixel * L_pixel + args.lambda_repa * L_repa
            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} L_eps={L_eps.item():.4f} "
                      f"L_pixel={L_pixel.item():.4f} L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(), "L_eps": L_eps.item(),
                            "L_pixel": L_pixel.item(), "L_repa": L_repa.item()})
                t0 = time.time()

        torch.save({"ddpm": ddpm.state_dict(), "render": render.state_dict(),
                    "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "vj_ddpm"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    print(f"\n=== EVAL: VJ-DDPM (DDIM {args.eval_steps}-step) ===")
    psnr_d, psnr_o, mean_iqa = _eval(ddpm, render, backbone, test_seqs, device, args.crop,
                                      args.ref_layer, alphas_bar, args.eval_steps, verbose=True)
    md = os.path.join(args.out_dir, "VJDDPM_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# VJ-DDPM v4: V-JEPA latent DDPM (eps-prediction)\n\n")
        f.write(f"CondDiT {n_d/1e6:.2f}M + Render {n_r/1e6:.2f}M + Proj {n_p/1e6:.2f}M trainable\n")
        f.write(f"V-JEPA encoder 304M frozen, T_diff={args.T_diff}, DDIM {args.eval_steps}-step\n\n")
        f.write(f"## SDSD held-out (DDPM pipeline)\n\n")
        f.write(f"- PSNR    = {psnr_d:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"## Oracle-render PSNR = {psnr_o:.3f}\n\n")
        f.write(f"vs E2b 19.556, vs PATH P 19.724, vs pure_rgb 18.533\n")
    print(f"\nVJ-DDPM PSNR={psnr_d:.3f} {fmt_iqa(mean_iqa)}  oracle={psnr_o:.3f}")
    print(f"wrote {md}")


@torch.no_grad()
def _eval(ddpm, render, backbone, test_seqs, device, crop, ref_layer,
          alphas_bar, num_steps, verbose=False, with_iqa=True):
    ddpm.eval(); render.eval()
    psnrs_d, psnrs_o = [], []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []} if with_iqa else None
    for name, lq_p, gt_p in test_seqs:
        lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
        lq, gt = lq.to(device), gt.to(device)
        z_LQ = encode(backbone, lq)[ref_layer].float()
        z_HQ = encode(backbone, gt)[ref_layer].float()

        z_HQ_pred = ddim_sample(ddpm, z_LQ, alphas_bar, num_inference_steps=num_steps)

        h_b, skips = render.encode(lq, kv_layers=[z_HQ_pred], delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ_pred], delta_layers=None)
        pred_d_f = pred.float()
        psnrs_d.append(mean_psnr(to_u8(pred_d_f), to_u8(gt)))
        if with_iqa:
            iqa = compute_iqa(pred_d_f, gt, device=device)
            for k_ in iqa_acc: iqa_acc[k_].append(iqa[k_])

        h_b, skips = render.encode(lq, kv_layers=[z_HQ], delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ], delta_layers=None)
        psnrs_o.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
        if verbose:
            extra = f" {fmt_iqa(iqa)}" if with_iqa else ""
            print(f"    [{name}] DDPM={psnrs_d[-1]:.3f}{extra}  oracle={psnrs_o[-1]:.3f}")

    ddpm.train(); render.train()
    if backbone is not None: backbone.eval()
    mean_iqa = ({k: float(np.mean(v)) for k, v in iqa_acc.items()} if with_iqa else None)
    return float(np.mean(psnrs_d)), float(np.mean(psnrs_o)), mean_iqa


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lambda_pixel", type=float, default=1.0)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--early_stop_iter", type=int, default=900)
    ap.add_argument("--dit_dim", type=int, default=512)
    ap.add_argument("--dit_depth", type=int, default=12)
    ap.add_argument("--dit_heads", type=int, default=8)
    ap.add_argument("--T_diff", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    main(ap.parse_args())
