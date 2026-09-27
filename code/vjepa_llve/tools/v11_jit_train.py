"""JiT-LLVE: Just image Transformer for LLVE (Li & He Nov 2025, arxiv 2511.13720).

Core idea (from "Back to Basics: Let Denoising Generative Models Denoise"):
  - Predict clean data DIRECTLY (no noise/velocity prediction)
  - Large-patch Transformer on pixels (no tokenizer, no SD prior)
  - "Back to basics" — Transformer that genuinely denoises

LLVE adaptation (deterministic, no diffusion):
  Input  LQ video (B, 3, T=16, 256, 256)
    │ patchify into 32×32 spatial patches → (B, T·8·8=1024 tokens, 32·32·3=3072)
    │ proj → dim 512
    │ + sinusoidal pos embedding (T,H,W)
    │ Factorized spatial-temporal Transformer (depth=12, dim=512, heads=8)
    │ proj 512 → 3072
    │ unpatchify → (B, 3, T, 256, 256)
    │ residual: pred = (LQ + output).clamp(0,1)
    ▼
  HQ pred

Loss: Charbonnier(pred, GT) + 0.5·REPA(CLIP teacher) [HASTE-gated]

vs vjpix (failed pixel-DDPM): direct prediction is more stable than predicting noise.
vs NAFNet (CNN): Transformer inductive bias on pixel patches.
vs vjlf (latent DiT): pixel-space, no V-JEPA latent dependence.
"""
from __future__ import annotations
import argparse, math, os, sys, time

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

from tools.iqa_metrics import compute_iqa, fmt_iqa, save_compare_frame
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr,
                                           charbonnier, build_split_seqs)
from tools.v11_c2_repa_train_clip import (encode_clip, REPAProjectorCLIP,
                                           CLIP_DIM, CLIP_GRID)


# ---------- helpers ---------------------------------------------------
def _sin_emb_3d(T, H, W, dim, device):
    """Concatenated sinusoidal positional embedding for (T, H, W) grid."""
    assert dim % 6 == 0, "dim must be divisible by 6 for 3D sin embedding"
    d = dim // 3
    half = d // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half)
    def axis(N):
        a = torch.arange(N, device=device, dtype=torch.float32)[:, None] * f[None, :]
        return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)             # (N, d)
    et = axis(T); eh = axis(H); ew = axis(W)
    et = et[:, None, None, :].expand(T, H, W, d)
    eh = eh[None, :, None, :].expand(T, H, W, d)
    ew = ew[None, None, :, :].expand(T, H, W, d)
    return torch.cat([et, eh, ew], dim=-1).reshape(T * H * W, dim)


class _SelfAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads; self.h_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.h_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class JiTBlock(nn.Module):
    """Factorized spatial + temporal SA + MLP. No t conditioning (deterministic)."""
    def __init__(self, dim, heads, T, H, W, ffn_mult=4.0, use_ckpt=True):
        super().__init__()
        self.T, self.H, self.W = T, H, W; self.dim = dim
        self.use_ckpt = use_ckpt
        self.norm_s = nn.LayerNorm(dim, eps=1e-6)
        self.attn_s = _SelfAttn(dim, heads)
        self.norm_t = nn.LayerNorm(dim, eps=1e-6)
        self.attn_t = _SelfAttn(dim, heads)
        self.norm_f = nn.LayerNorm(dim, eps=1e-6)
        h = int(dim * ffn_mult)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))

    def _spatial(self, x):
        B = x.shape[0]; T, H, W, D = self.T, self.H, self.W, self.dim
        h = self.norm_s(x)
        h = h.reshape(B, T, H * W, D).reshape(B * T, H * W, D)
        h = self.attn_s(h).reshape(B, T * H * W, D)
        return x + h

    def _temporal(self, x):
        B = x.shape[0]; T, H, W, D = self.T, self.H, self.W, self.dim
        h = self.norm_t(x)
        h = h.reshape(B, T, H * W, D).transpose(1, 2).reshape(B * H * W, T, D)
        h = self.attn_t(h).reshape(B, H * W, T, D).transpose(1, 2).reshape(B, T * H * W, D)
        return x + h

    def _mlp(self, x):
        return x + self.mlp(self.norm_f(x))

    def forward(self, x):
        if self.use_ckpt and self.training and x.requires_grad:
            x = ckpt.checkpoint(self._spatial, x, use_reentrant=False)
            x = ckpt.checkpoint(self._temporal, x, use_reentrant=False)
            x = ckpt.checkpoint(self._mlp, x, use_reentrant=False)
        else:
            x = self._spatial(x); x = self._temporal(x); x = self._mlp(x)
        return x


class JiTLLVE(nn.Module):
    """Just-image-Transformer for LLVE (deterministic feed-forward)."""
    def __init__(self, in_ch=3, out_ch=3, image_size=256, patch_h=32, patch_w=32,
                 num_frames=16, dim=512, depth=12, heads=8, ffn_mult=4.0):
        super().__init__()
        assert image_size % patch_h == 0 and image_size % patch_w == 0
        self.T = num_frames
        self.H = image_size // patch_h
        self.W = image_size // patch_w
        self.patch_h, self.patch_w = patch_h, patch_w
        self.in_ch, self.out_ch = in_ch, out_ch
        patch_dim = in_ch * patch_h * patch_w

        self.in_proj = nn.Linear(patch_dim, dim)
        self.register_buffer("pos", _sin_emb_3d(self.T, self.H, self.W, dim, torch.device("cpu")))
        self.blocks = nn.ModuleList([
            JiTBlock(dim, heads, self.T, self.H, self.W, ffn_mult) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        self.out_proj = nn.Linear(dim, out_ch * patch_h * patch_w)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

        # Bottleneck h_b for REPA: take features after middle block
        self.bottleneck_idx = depth // 2
        self.bottleneck_dim = dim

    def patchify(self, x):
        # x: (B, C, T, H, W) → (B, T*H'*W', C*ph*pw)
        B, C, T, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        Hp, Wp = H // ph, W // pw
        x = x.reshape(B, C, T, Hp, ph, Wp, pw)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous()             # (B, T, Hp, Wp, C, ph, pw)
        x = x.reshape(B, T * Hp * Wp, C * ph * pw)
        return x

    def unpatchify(self, x):
        # x: (B, T*Hp*Wp, C*ph*pw) → (B, C, T, H, W)
        B = x.shape[0]
        T, Hp, Wp = self.T, self.H, self.W
        C, ph, pw = self.out_ch, self.patch_h, self.patch_w
        x = x.reshape(B, T, Hp, Wp, C, ph, pw)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()             # (B, C, T, Hp, ph, Wp, pw)
        return x.reshape(B, C, T, Hp * ph, Wp * pw)

    def forward(self, lq):
        x = self.patchify(lq)                                       # (B, N, patch_dim)
        x = self.in_proj(x) + self.pos[None]                        # (B, N, D)
        h_b = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.bottleneck_idx:
                # Snapshot for REPA loss target alignment
                h_b = x  # (B, N, D)
        x = self.norm_out(x)
        out_patches = self.out_proj(x)                              # (B, N, C*ph*pw)
        out_video = self.unpatchify(out_patches)                    # (B, C, T, H, W)
        pred = (lq + out_video).clamp(0, 1)
        return pred, h_b


class REPAProjectorJiT(nn.Module):
    """Project token bottleneck h_b (B, N, D=512) → CLIP-aligned tokens (B, N_clip, 768)."""
    def __init__(self, in_dim=512, out_dim=CLIP_DIM, T=16, H=8, W=8, target_grid=CLIP_GRID):
        super().__init__()
        self.T, self.H, self.W = T, H, W
        self.proj = nn.Linear(in_dim, out_dim)
        self.target_grid = target_grid

    def forward(self, h_b):
        x = self.proj(h_b)                                          # (B, T*H*W, out_dim)
        B, N, D = x.shape
        # Reshape to spatial grid then interp to target_grid (T, 8, 8) — already match if patch=32
        x = x.reshape(B, self.T, self.H, self.W, D).permute(0, 4, 1, 2, 3)   # (B, D, T, H, W)
        if (self.T, self.H, self.W) != self.target_grid:
            x = F.interpolate(x, size=self.target_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, D).contiguous()
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

    from transformers import CLIPVisionModel
    clip_t = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir="/cache/zihaohe/hf_cache",
    ).to(device).eval()
    for p in clip_t.parameters(): p.requires_grad_(False)

    net = JiTLLVE(image_size=args.crop, patch_h=args.patch, patch_w=args.patch,
                  num_frames=16, dim=args.dim, depth=args.depth, heads=args.heads).to(device)
    n_n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    projector = REPAProjectorJiT(in_dim=args.dim, T=net.T, H=net.H, W=net.W).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    print(f"[JiT-LLVE] CLIP-B/32 87M frozen (REPA target only)")
    print(f"           JiT: {n_n/1e6:.2f}M trainable (patch={args.patch}, dim={args.dim}, depth={args.depth})")
    print(f"           Token grid: T={net.T}, H={net.H}, W={net.W} → {net.T*net.H*net.W} tokens")
    print(f"           REPAProjector: {n_p/1e6:.2f}M (training only)")
    print(f"           Loss: Charbonnier + {args.lambda_repa}·REPA(CLIP layer 8) [HASTE @ {args.early_stop_iter}]")
    print(f"           Inference: deterministic 1-pass (LQ + JiT(LQ) → HQ)")

    train_params = list(net.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr * 0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt_d = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        net.load_state_dict(ckpt_d["net"])
        if "projector" in ckpt_d:
            projector.load_state_dict(ckpt_d["projector"])
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
                z_tea = encode_clip(clip_t, lq, args.clip_layer).float()
                z_stu = projector(h_b.float())
                m = z_tea.mean(dim=1, keepdim=True); s = z_tea.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_tea - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L = L_rgb + args.lambda_repa * L_repa
            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} "
                      f"L_rgb={L_rgb.item():.4f} L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(), "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
                t0 = time.time()

        torch.save({"net": net.state_dict(), "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "jit_llve"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    print(f"\n=== EVAL: JiT-LLVE (1-pass, large-patch Transformer) ===")
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
                save_compare_frame(os.path.join(args.out_dir, "viz", f"{name}.png"),
                                   lq, pred_f, gt)

    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "JIT_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# JiT-LLVE: Just image Transformer (Li & He Nov 2025 \"Back to Basics\")\n\n")
        f.write(f"JiT {n_n/1e6:.2f}M trainable; patch={args.patch} dim={args.dim} depth={args.depth}\n")
        f.write(f"Token grid: T={net.T}, H={net.H}, W={net.W} ({net.T*net.H*net.W} tokens)\n")
        f.write(f"Direct prediction (no t/noise/diffusion); REPA(CLIP layer {args.clip_layer}) auxiliary\n\n")
        f.write(f"## SDSD held-out (1-pass)\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"vs nafnet_15k 20.436, vs clip_30k 20.746, vs clip_60k 20.737, vs vjlf_30k 19.858\n")
    print(f"\nJiT-LLVE PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch", type=int, default=32, help="Patch size (32 → 8x8 grid per frame at 256)")
    ap.add_argument("--dim", type=int, default=510, help="Must be divisible by 6 for 3D sin pos emb")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--clip_layer", type=int, default=8)
    ap.add_argument("--early_stop_iter", type=int, default=9000)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--save_viz", action="store_true",
                    help="Save LQ|Pred|GT comparison PNGs to out_dir/viz/ during eval")
    main(ap.parse_args())
