"""VJ-Mamba: V-JEPA Latent Mamba (replace DiT with selective SSM).

Architecture (same as VJLF v1, only the temporal mixer changes):
  V-JEPA encoder (frozen 304M)
    LQ ─► z_LQ (B, N=2048, 1024)
    GT ─► z_HQ (B, 2048, 1024)

  VJepaLatentMamba (~50M trainable, depth=12, dim=384)
    rectified flow training:
      sample t~U(0,1)
      z_t = (1-t)·z_LQ + t·z_HQ
      v_target = z_HQ - z_LQ
      v_pred = Mamba(z_t, t)
      L_flow = MSE(v_pred, v_target)
    inference:
      z_HQ_pred = Euler ODE solve from z_LQ, num_steps=4

  DeltaModulatedDecoder (33M render head)
    train: kv=[z_HQ] → pred → charbonnier(pred, GT)
    eval:  kv=[z_HQ_pred from Mamba] → pred
    REPA on bottleneck h_b → V-JEPA(GT) layer 3
    HASTE early-stop @ iter 900

Implementation note:
  Hand-coded selective SSM (Mamba-2 simplified) with sequential scan.
  Spatial scan over H*W=256 per frame, temporal scan over T=8 per location.
  No mamba_ssm dependency; pure PyTorch.
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
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


# =========================================================================
# Selective SSM (simplified Mamba-2 in pure PyTorch, sequential scan)
# =========================================================================
class SelectiveSSM(nn.Module):
    """Hand-coded selective state-space model.

    Per-token: h_t = A_bar_t · h_{t-1} + B_bar_t · x_t
               y_t = C_t · h_t + D · x_t
    where A_bar = exp(delta · A), B_bar = delta · B; A, delta are per-channel,
    B and C are selective (depend on x).
    """
    def __init__(self, dim: int, state_size: int = 16):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        # A is parametrized as -exp(A_log) for stability (always negative)
        A_init = torch.log(torch.arange(1, state_size + 1).float()).expand(dim, state_size).contiguous()
        self.A_log = nn.Parameter(A_init.clone())
        self.D = nn.Parameter(torch.ones(dim))
        # Selective projection: produces delta (D), B (S), C (S) from input
        self.x_proj = nn.Linear(dim, 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        # Init dt bias for positive softplus
        nn.init.constant_(self.dt_proj.bias, 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → y: (B, L, D)."""
        B, L, D = x.shape
        S = self.state_size

        BC = self.x_proj(x)                                 # (B, L, 2S)
        B_p = BC[..., :S]                                   # (B, L, S)
        C_p = BC[..., S:]                                   # (B, L, S)
        delta = F.softplus(self.dt_proj(x))                 # (B, L, D), positive

        A = -torch.exp(self.A_log.float())                  # (D, S)
        # Discretize per-token: A_bar(b,l,d,s) = exp(delta(b,l,d) * A(d,s))
        A_bar = torch.exp(delta.float().unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))   # (B, L, D, S)
        B_bar = delta.float().unsqueeze(-1) * B_p.float().unsqueeze(2)                  # (B, L, D, S)
        C_p_f = C_p.float()
        x_f = x.float()

        # Sequential scan (slow for large L, fine for L<=256)
        h = torch.zeros(B, D, S, device=x.device, dtype=torch.float32)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x_f[:, t].unsqueeze(-1)        # (B, D, S)
            y = (C_p_f[:, t].unsqueeze(1) * h).sum(dim=-1)                     # (B, D)
            ys.append(y)
        y = torch.stack(ys, dim=1).to(x.dtype)                                  # (B, L, D)
        return y + self.D.to(x.dtype) * x


# =========================================================================
# Mamba block: AdaLN-Zero (time conditioning) + spatial SSM + temporal SSM + FFN
# =========================================================================
class TimeEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        emb = torch.cat([torch.cos(t.float()[:, None] * freqs[None, :]),
                          torch.sin(t.float()[:, None] * freqs[None, :])], dim=-1)
        if dim % 2: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t, self.freq_dim))


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MambaBlock(nn.Module):
    """One block: AdaLN-Zero modulation + spatial SSM + temporal SSM + FFN."""
    def __init__(self, dim: int, T: int, H: int, W: int, ffn_mult: float = 4.0,
                 state_size: int = 16, use_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.T = T; self.H = H; self.W = W
        self.use_checkpoint = use_checkpoint

        self.adaln = nn.Linear(dim, 9 * dim)
        nn.init.zeros_(self.adaln.weight); nn.init.zeros_(self.adaln.bias)

        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ssm_s = SelectiveSSM(dim, state_size)

        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ssm_t = SelectiveSSM(dim, state_size)

        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * ffn_mult)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))

    def _spatial(self, x, sh, sc, ga):
        B, T, H, W, D = x.shape[0], self.T, self.H, self.W, self.dim
        h = _modulate(self.norm_s(x), sh, sc)
        h = h.reshape(B * T, H * W, D)
        h = self.ssm_s(h)
        return x + ga.unsqueeze(1) * h.reshape(B, T * H * W, D)

    def _temporal(self, x, sh, sc, ga):
        B, T, H, W, D = x.shape[0], self.T, self.H, self.W, self.dim
        h = _modulate(self.norm_t(x), sh, sc)
        h = h.reshape(B, T, H * W, D).transpose(1, 2).reshape(B * H * W, T, D)
        h = self.ssm_t(h)
        return x + ga.unsqueeze(1) * h.reshape(B, H * W, T, D).transpose(1, 2).reshape(B, T * H * W, D)

    def _mlp(self, x, sh, sc, ga):
        return x + ga.unsqueeze(1) * self.mlp(_modulate(self.norm_f(x), sh, sc))

    def forward(self, x, t_emb):
        sh_s, sc_s, ga_s, sh_t, sc_t, ga_t, sh_f, sc_f, ga_f = self.adaln(t_emb).chunk(9, dim=1)
        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self._spatial, x, sh_s, sc_s, ga_s, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._temporal, x, sh_t, sc_t, ga_t, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._mlp, x, sh_f, sc_f, ga_f, use_reentrant=False)
        else:
            x = self._spatial(x, sh_s, sc_s, ga_s)
            x = self._temporal(x, sh_t, sc_t, ga_t)
            x = self._mlp(x, sh_f, sc_f, ga_f)
        return x


class VJepaLatentMamba(nn.Module):
    """Mamba in V-JEPA latent space, identity-at-init, ODE-sampleable."""
    def __init__(self, vjepa_dim=1024, grid=(8, 16, 16), dim=384, depth=12,
                 ffn_mult=4.0, state_size=16, use_gradient_checkpoint=True):
        super().__init__()
        self.vjepa_dim = vjepa_dim
        self.dim = dim
        self.T, self.H, self.W = grid
        self.n_tokens = self.T * self.H * self.W
        self.in_proj = nn.Linear(vjepa_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.t_embedder = TimeEmbedder(dim)
        self.blocks = nn.ModuleList([
            MambaBlock(dim, self.T, self.H, self.W, ffn_mult, state_size, use_gradient_checkpoint)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.final_adaln.weight); nn.init.zeros_(self.final_adaln.bias)
        self.final_proj = nn.Linear(dim, vjepa_dim)
        nn.init.zeros_(self.final_proj.weight); nn.init.zeros_(self.final_proj.bias)

    def forward(self, z_t, t):
        x = self.in_proj(z_t) + self.pos_embed
        t_emb = self.t_embedder(t)
        for blk in self.blocks: x = blk(x, t_emb)
        shift, scale = self.final_adaln(t_emb).chunk(2, dim=1)
        return self.final_proj(_modulate(self.final_norm(x), shift, scale))

    @torch.no_grad()
    def sample(self, z_lq, num_steps=4):
        z = z_lq.clone()
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_k = torch.full((z.shape[0],), k * dt + 0.5 * dt, device=z.device, dtype=torch.float32)
            v = self(z, t_k)
            z = z + dt * v
        return z


# =========================================================================
# REPA projector (same as VJLF v1)
# =========================================================================
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


# =========================================================================
# Trainer
# =========================================================================
def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    mamba = VJepaLatentMamba(
        vjepa_dim=1024, grid=(8, 16, 16),
        dim=args.mamba_dim, depth=args.mamba_depth,
        state_size=args.state_size, use_gradient_checkpoint=True,
    ).to(device)
    n_m = sum(p.numel() for p in mamba.parameters() if p.requires_grad)

    render = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_r = sum(p.numel() for p in render.parameters() if p.requires_grad)

    projector = REPAProjector(in_ch=render.dims[-1], out_dim=render.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    print(f"[VJ-Mamba] V-JEPA encoder 304M frozen")
    print(f"           Mamba (rectified flow): {n_m/1e6:.2f}M trainable "
          f"(dim={args.mamba_dim} depth={args.mamba_depth} state={args.state_size})")
    print(f"           Render: {n_r/1e6:.2f}M, REPA proj: {n_p/1e6:.2f}M")
    print(f"           Total trainable: {(n_m+n_r+n_p)/1e6:.2f}M")
    print(f"           Loss: L_flow + {args.lambda_pixel}·L_pixel + "
          f"{args.lambda_repa}·L_repa (HASTE off @ iter {args.early_stop_iter})")

    train_params = list(mamba.parameters()) + list(render.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
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
            z_LQ = encode(backbone, lq)[args.ref_layer].float()
            z_HQ = encode(backbone, gt)[args.ref_layer].float()

        B = z_LQ.shape[0]
        t_sample = torch.rand(B, device=device)
        t_view = t_sample.view(B, 1, 1)
        z_t = (1.0 - t_view) * z_LQ + t_view * z_HQ
        v_target = z_HQ - z_LQ

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = mamba(z_t, t_sample)
        L_flow = F.mse_loss(v_pred.float(), v_target)

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

        L = L_flow + args.lambda_pixel * L_pixel + args.lambda_repa * L_repa

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} L_flow={L_flow.item():.4f} "
                  f"L_pixel={L_pixel.item():.4f} L_repa={L_repa.item():.4f} "
                  f"dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item(), "L_flow": L_flow.item(),
                        "L_pixel": L_pixel.item(), "L_repa": L_repa.item()})
            t0 = time.time()

    torch.save({"mamba": mamba.state_dict(), "render": render.state_dict(),
                "projector": projector.state_dict(),
                "args": vars(args), "log": log, "stage": "vj_mamba"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Final eval: full pipeline + oracle upper bound
    print(f"\n=== FINAL EVAL: VJ-Mamba (Mamba {args.eval_steps}-step ODE → render) ===")
    psnr_m, psnr_o = _quick_eval(mamba, render, backbone, test_seqs,
                                  device, args.crop, args.ref_layer,
                                  args.eval_steps, verbose=True)
    md = os.path.join(args.out_dir, "VJM_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# VJ-Mamba: V-JEPA Latent Mamba\n\n")
        f.write(f"Mamba {n_m/1e6:.2f}M + Render {n_r/1e6:.2f}M + projector {n_p/1e6:.2f}M trainable\n")
        f.write(f"V-JEPA encoder 304M frozen\n")
        f.write(f"rectified flow + REPA + HASTE\n\n")
        f.write(f"## Mamba-pipeline PSNR = {psnr_m:.3f}\n")
        f.write(f"## Oracle-render PSNR = {psnr_o:.3f}\n\n")
        f.write(f"vs E2b 19.556, vs PATH P 19.724, vs pure_rgb 18.533\n")
    print(f"\nVJ-Mamba PSNR = {psnr_m:.3f}  (vs E2b 19.556)")
    print(f"oracle PSNR    = {psnr_o:.3f}  (upper bound)")
    print(f"wrote {md}")


@torch.no_grad()
def _quick_eval(mamba, render, backbone, test_seqs, device, crop, ref_layer,
                num_steps, verbose=False):
    mamba.eval(); render.eval()
    psnrs_m, psnrs_o = [], []
    for name, lq_p, gt_p in test_seqs:
        lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
        lq, gt = lq.to(device), gt.to(device)
        z_LQ = encode(backbone, lq)[ref_layer].float()
        z_HQ = encode(backbone, gt)[ref_layer].float()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_HQ_pred = mamba.sample(z_LQ, num_steps=num_steps).float()
        h_b, skips = render.encode(lq, kv_layers=[z_HQ_pred], delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ_pred], delta_layers=None)
        psnrs_m.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))

        h_b, skips = render.encode(lq, kv_layers=[z_HQ], delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ], delta_layers=None)
        psnrs_o.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))

        if verbose: print(f"    [{name}] Mamba={psnrs_m[-1]:.3f}  oracle={psnrs_o[-1]:.3f}")

    mamba.train(); render.train()
    if backbone is not None: backbone.eval()
    return float(np.mean(psnrs_m)), float(np.mean(psnrs_o))


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
    ap.add_argument("--mamba_dim", type=int, default=384)
    ap.add_argument("--mamba_depth", type=int, default=12)
    ap.add_argument("--state_size", type=int, default=16)
    ap.add_argument("--eval_steps", type=int, default=4)
    main(ap.parse_args())
