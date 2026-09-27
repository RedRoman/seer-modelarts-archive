"""WanVJ-LLVE Phase 1: Wan2.1-T2V-1.3B + V-JEPA 2.1 + LoRA on BVI-RLV.

Real-time low-light video enhancement via:
  - Wan2.1 VAE (frozen, 127M) — 8x spatial / 4x temporal compression
  - Wan2.1 DiT 1.3B (frozen) — diffusion transformer with cross-attn
  - V-JEPA 2.1 encoder (frozen, 304M) — replaces T5 text condition
  - LoRA on DiT (DiffSynth official recipe: q,k,v,o,ffn.0,ffn.2 rank=32)
  - V-JEPA → 4096-dim projector (~4M trainable)
  - LQ conditioning: zero-initialized add embed (in_channels: 16 stays 16)

Training:
  - Dataset: BVI-RLV (44 scenes × 2 fps = 80 paired sequences, ~50K frames vs SDSD 122 seq × 16 = 1952)
  - Loss: rectified flow MSE (v_pred ↔ z_HQ - z_LQ)
  - LR: 1e-4 (DiffSynth recipe)
  - Iters: 25000 (~5 epochs of BVI-RLV at batch 1)

Inference: 4-step Euler ODE first (validate correctness), then distill to 1-NFE in Phase 3.

NOTE — DRAFT, NOT YET SMOKE-TESTED. Run smoke before launch.

Watchouts:
  - torch.nn.RMSNorm shim (torch 2.3 doesn't have it natively)
  - Wan VAE causal: encode 16 frames → latent 4 frames, decode → 13 RGB frames
    (3-frame loss). Eval must trim GT to 13 frames OR pad input to 17.
  - DiT input shape: (B, 16, T_lat=4, H_lat=32, W_lat=32) at 256x256x16 video
  - text_dim=4096 (UMT5), so V-JEPA(1024) → projector(4096)
  - DiT.forward(hidden_states, timestep, encoder_hidden_states)
    timestep is LongTensor in [0, 1000) (diffusion-style)
"""
from __future__ import annotations
import argparse, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# -------- RMSNorm shim (required for torch < 2.4) --------
class _RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **kw):
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter('weight', None)
    def forward(self, x):
        dtype = x.dtype
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        v = v.to(dtype)
        if self.weight is not None:
            v = v * self.weight
        return v
if not hasattr(torch.nn, "RMSNorm"):
    torch.nn.RMSNorm = _RMSNorm

# Shim 2: torch 2.3's scaled_dot_product_attention doesn't accept `enable_gqa` kwarg
# (added later). Diffusers' Wan attention dispatch passes it. Strip on call.
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention
def _sdpa_compat(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _orig_sdpa(*args, **kwargs)
torch.nn.functional.scaled_dot_product_attention = _sdpa_compat


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from diffusers import AutoencoderKLWan, WanTransformer3DModel
from models.vjepa_backbone import build_vjepa_backbone
from tools.iqa_metrics import compute_iqa, fmt_iqa, save_compare_video
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)
from data.bvi_rlv_dataset import BVIRLVDataset


WAN_PATH = "/cache/zihaohe/checkpoint/wan21_t2v_1_3b"
BVI_RLV_PATH = "/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos"

# -------- YCbCr chroma AdaIN (post-processing fix for chroma drift) --------
_RGB2YCBCR = torch.tensor([[ 0.299,  0.587,  0.114],
                           [-0.169, -0.331,  0.500],
                           [ 0.500, -0.419, -0.081]], dtype=torch.float32)
_YCBCR2RGB = torch.tensor([[ 1.0,  0.0,    1.402  ],
                           [ 1.0, -0.344, -0.714 ],
                           [ 1.0,  1.772,  0.0   ]], dtype=torch.float32)
def rgb_to_ycbcr(x):
    """x: (B,3,T,H,W) in [0,1] -> ycbcr in [0,1]/[-0.5,0.5]/[-0.5,0.5]"""
    M = _RGB2YCBCR.to(x.device, x.dtype)
    return torch.einsum('cd,bdthw->bcthw', M, x)
def ycbcr_to_rgb(x):
    M = _YCBCR2RGB.to(x.device, x.dtype)
    return torch.einsum('cd,bdthw->bcthw', M, x).clamp(0, 1)
def chroma_adain(pred_rgb, ref_rgb):
    """Match Cb/Cr mean/std of pred to ref, keep Y unchanged. ref is GT (oracle) or LQ."""
    pred_y = rgb_to_ycbcr(pred_rgb)
    ref_y  = rgb_to_ycbcr(ref_rgb)
    out = pred_y.clone()
    for c in (1, 2):  # Cb, Cr
        p = pred_y[:, c]
        r = ref_y[:, c]
        p_mean, p_std = p.mean(), p.std() + 1e-6
        r_mean, r_std = r.mean(), r.std() + 1e-6
        out[:, c] = (p - p_mean) / p_std * r_std + r_mean
    return ycbcr_to_rgb(out)



# -------- LoRA on Wan attn linears --------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling


def inject_lora_on_dit(dit, rank=32):
    """DiffSynth-Studio official recipe: q, k, v, o, ffn.0, ffn.2 on every block.

    For Wan: attn1.to_{q,k,v} + attn1.to_out[0] (= o) + attn2.to_{q,k,v} + attn2.to_out[0]
            + ffn.net[0].proj (= ffn.0) + ffn.net[2] (= ffn.2)
    = 10 LoRA modules per block × 30 blocks = 300 modules.
    Returns count of injected.
    """
    n = 0
    for blk in dit.blocks:
        # Self-attn (attn1) and cross-attn (attn2): q, k, v, o
        for kind in ("attn1", "attn2"):
            attn = getattr(blk, kind, None)
            if attn is None: continue
            for proj in ("to_q", "to_k", "to_v"):
                lin = getattr(attn, proj, None)
                if isinstance(lin, nn.Linear):
                    setattr(attn, proj, LoRALinear(lin, rank=rank, alpha=float(rank)))
                    n += 1
            # to_out is a ModuleList: [Linear, Dropout]
            if hasattr(attn, "to_out") and isinstance(attn.to_out[0], nn.Linear):
                attn.to_out[0] = LoRALinear(attn.to_out[0], rank=rank, alpha=float(rank))
                n += 1
        # FFN: ffn.net[0].proj (1536->8960) and ffn.net[2] (8960->1536)
        if hasattr(blk, "ffn") and hasattr(blk.ffn, "net"):
            net = blk.ffn.net
            if hasattr(net[0], "proj") and isinstance(net[0].proj, nn.Linear):
                net[0].proj = LoRALinear(net[0].proj, rank=rank, alpha=float(rank))
                n += 1
            if isinstance(net[2], nn.Linear):
                net[2] = LoRALinear(net[2], rank=rank, alpha=float(rank))
                n += 1
    return n


# -------- V-JEPA → text_dim projector (replaces T5 cross-attn KV) --------
class VJepaToText(nn.Module):
    """Project V-JEPA features (B, N, 1024) → text-conditioning (B, N, text_dim)."""
    def __init__(self, vj_dim=1024, text_dim=4096):
        super().__init__()
        self.proj = nn.Linear(vj_dim, text_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
    def forward(self, z_vj):
        return self.proj(z_vj)


# -------- LQ conditioning (zero-init add to z_t) --------
class LQConditioner(nn.Module):
    """Project z_LQ_wan (B,16,T,H,W) → same shape as z_t for additive condition.
    Zero-init so initial behavior matches pretrained Wan T2V."""
    def __init__(self, ch=16):
        super().__init__()
        self.proj = nn.Conv3d(ch, ch, 3, padding=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    def forward(self, z_lq):
        return self.proj(z_lq)


# -------- Wan VAE normalization (per-channel mean/std from config) --------
def get_wan_latents_norm(vae, device):
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, 16, 1, 1, 1)
    return mean, std


@torch.no_grad()
def wan_encode(vae, video_01, mean, std):
    """video in [0,1] (B,3,T,H,W) → normalized latent (B,16,T_lat,H_lat,W_lat)."""
    # Wan VAE expects [-1, 1]
    v = video_01 * 2.0 - 1.0
    z = vae.encode(v).latent_dist.sample()
    return (z - mean) / std


@torch.no_grad()
def wan_decode(vae, z_norm, mean, std):
    """Latent (B,16,T_lat,H_lat,W_lat) → video [0,1] (B,3,T_out,H,W)."""
    z = z_norm * std + mean
    out = vae.decode(z).sample              # in [-1, 1]
    return ((out + 1.0) * 0.5).clamp(0, 1)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Frozen backbones ----
    print("[WanVJ] loading Wan VAE ...")
    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    n_vae = sum(p.numel() for p in vae.parameters())

    print("[WanVJ] loading Wan DiT 1.3B ...")
    dit = WanTransformer3DModel.from_pretrained(WAN_PATH, subfolder="transformer", torch_dtype=torch.bfloat16).to(device).eval()
    for p in dit.parameters(): p.requires_grad_(False)
    n_dit = sum(p.numel() for p in dit.parameters())

    print("[WanVJ] loading V-JEPA encoder ...")
    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    # ---- Trainable adapters ----
    n_lora = inject_lora_on_dit(dit, rank=args.lora_rank)
    dit.to(device)  # move newly-added LoRA params to GPU
    print(f"[WanVJ] injected LoRA on {n_lora} attn linears")

    vj2text = VJepaToText(vj_dim=1024, text_dim=4096).to(device)
    lq_cond = LQConditioner(ch=16).to(device)
    train_params = ([p for p in dit.parameters() if p.requires_grad]
                    + list(vj2text.parameters())
                    + list(lq_cond.parameters()))
    n_train = sum(p.numel() for p in train_params)

    print(f"[WanVJ] Wan VAE 127M frozen, Wan DiT 1.3B frozen + LoRA")
    print(f"        V-JEPA 304M frozen → text-replacement projector")
    print(f"        Trainable: {n_train/1e6:.2f}M")
    print(f"        Inference: 1-NFE rectified flow (real-time-ish)")

    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr * 0.05)

    # BVI-RLV training data (44 scenes × 2 fps = 80 sequences, ~50K paired frames)
    bvi_train = BVIRLVDataset(BVI_RLV_PATH, split="train", clip_len=16,
                              crop_size=args.crop, is_train=True)
    # SDSD test set (use existing protocol for fair comparison vs prior work)
    _, test_seqs = build_split_seqs()
    print(f"[data] BVI-RLV train sequences: {len(bvi_train)}; SDSD test sequences: {len(test_seqs)}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    mean, std = get_wan_latents_norm(vae, device)
    log = []
    if args.eval_only:
        ckpt_d = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        # LoRA shards: only A/B keys saved; load with strict=False
        if "dit_lora" in ckpt_d:
            dit.load_state_dict({k: v.to(device) for k, v in ckpt_d["dit_lora"].items()},
                                strict=False)
        vj2text.load_state_dict(ckpt_d["vj2text"])
        lq_cond.load_state_dict(ckpt_d["lq_cond"])
        print(f"[eval-only] loaded ckpt → {args.out_dir}/ckpt.pt")
    if args.eval_only: pass
    elif True:
     for it in range(args.iters):
        # Sample one BVI-RLV clip
        sample = bvi_train[int(rng.integers(0, len(bvi_train)))]
        lq = sample["lq"].unsqueeze(0).to(device)              # (1, 3, T, H, W)
        gt = sample["hq"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_lq = wan_encode(vae, lq, mean, std)                 # (B,16,4,32,32)
            z_hq = wan_encode(vae, gt, mean, std)
            z_vj = encode(backbone, lq)[args.ref_layer].float()   # (B, 2048, 1024)

        # Rectified flow: t~U(0,1), z_t=(1-t)z_lq + t z_hq, v_target=z_hq-z_lq
        B = z_lq.shape[0]
        t = torch.rand(B, device=device)
        tv = t.view(B, 1, 1, 1, 1)
        z_t = (1.0 - tv) * z_lq + tv * z_hq
        v_target = z_hq - z_lq

        # LQ conditioning (zero-init add)
        z_t_cond = z_t + lq_cond(z_lq)

        # V-JEPA → text-replacement
        text_emb = vj2text(z_vj)                                  # (B, 2048, 4096)

        # DiT predicts velocity
        timestep = (t * 1000).long()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = dit(z_t_cond.to(torch.bfloat16),
                         timestep,
                         encoder_hidden_states=text_emb.to(torch.bfloat16),
                         return_dict=False)[0].float()
        L_flow = F.mse_loss(v_pred, v_target)

        L = L_flow
        # TODO: optional pixel/REPA losses
        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} L_flow={L_flow.item():.4f}")
            log.append({"iter": it+1, "L_flow": L_flow.item()})

    if not args.eval_only:
        torch.save({"dit_lora": {k: v.cpu() for k, v in dit.state_dict().items() if "A" in k or "B" in k},
                    "vj2text": vj2text.state_dict(), "lq_cond": lq_cond.state_dict(),
                    "args": vars(args), "log": log, "stage": "wanvj_llve"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # -------- 1-NFE inference eval --------
    print("\n=== EVAL: WanVJ-LLVE (1-NFE rectified flow) ===")
    dit.eval(); vj2text.eval(); lq_cond.eval()
    psnrs = []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_lq = wan_encode(vae, lq, mean, std)
            z_vj = encode(backbone, lq)[args.ref_layer].float()
            # V-JEPA causality test
            if args.vjepa_mode == "shuffle_tokens":
                _N = z_vj.shape[1]
                _perm = torch.randperm(_N, device=z_vj.device)
                z_vj = z_vj[:, _perm, :]
            text_emb = vj2text(z_vj)
            if args.vjepa_mode == "zero":
                text_emb = torch.zeros_like(text_emb)
            B = z_lq.shape[0]
            # Multi-step Euler ODE: integrate from t=0 (LQ) to t=1 (HQ)
            # Rectified flow: z_{t+dt} = z_t + dt * v_pred(z_t, t)
            z_cur = z_lq.clone()
            N = max(1, args.steps)
            dt = 1.0 / N
            for k in range(N):
                t_now = torch.full((B,), k * dt * 1000, device=device).long()
                z_cur_cond = z_cur + lq_cond(z_cur)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v_k = dit(z_cur_cond.to(torch.bfloat16), t_now,
                              encoder_hidden_states=text_emb.to(torch.bfloat16),
                              return_dict=False)[0].float()
                z_cur = z_cur + dt * v_k
            pred = wan_decode(vae, z_cur, mean, std)              # (B,3,T_out=13,H,W)

            # Trim GT to match T_out
            T_out = pred.shape[2]
            T_drop = (gt.shape[2] - T_out) // 2
            gt_trim = gt[:, :, T_drop:T_drop + T_out]

            pred_f = pred.float()
            # chroma AdaIN (DK-EC1 Step 0): match pred CbCr to ref CbCr stats
            if args.chroma_adain != "none":
                T_out_t = pred_f.shape[2]
                T_drop_t = (gt.shape[2] - T_out_t) // 2
                ref = (gt[:, :, T_drop_t:T_drop_t+T_out_t] if args.chroma_adain == "oracle_gt"
                       else lq[:, :, T_drop_t:T_drop_t+T_out_t])
                pred_f = chroma_adain(pred_f, ref).clamp(0, 1)
            p = mean_psnr(to_u8(pred_f), to_u8(gt_trim))
            iqa = compute_iqa(pred_f, gt_trim, device=device)
            psnrs.append(p)
            for k in iqa_acc: iqa_acc[k].append(iqa[k])
            print(f"    [{name}] PSNR={p:.3f} {fmt_iqa(iqa)}")
            if args.save_viz:
                viz_root = args.viz_dir if args.viz_dir else os.path.join(args.out_dir, "viz")
                # Trim LQ to match the 13-frame causal-VAE output for clean side-by-side
                lq_trim = lq[:, :, T_drop:T_drop + T_out]
                save_compare_video(os.path.join(viz_root, f"{name}.mp4"),
                                   lq_trim, pred_f, gt_trim, fps=10)

    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "WANVJ_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# WanVJ-LLVE: Wan2.1 + V-JEPA + LoRA + 1-NFE rectified flow\n\n")
        f.write(f"Wan VAE 127M + Wan DiT 1.3B + V-JEPA 304M (all frozen)\n")
        f.write(f"LoRA + adapters: {n_train/1e6:.2f}M trainable\n")
        f.write(f"NOTE: Wan VAE causal — eval trims GT to {T_out} frames (lost 3 frames)\n\n")
        f.write(f"## SDSD held-out (1-NFE)\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}\n\n")
        f.write(f"vs nafnet 20.4, vs clip_30k 20.7, vs vjlf_30k 19.86\n")
    print(f"\nWanVJ-LLVE PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=25000)             # ~5 epochs BVI-RLV
    ap.add_argument("--lr", type=float, default=1e-4)               # DiffSynth recipe
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+viz only")
    ap.add_argument("--save_viz", action="store_true",
                    help="Save MP4 LQ|Pred|GT comparison videos during eval")
    ap.add_argument("--viz_dir", default=None,
                    help="Override viz output dir (default: out_dir/viz/)")
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lora_rank", type=int, default=32)            # DiffSynth recipe
    ap.add_argument("--chroma_adain", default="none",
                    choices=["none","oracle_gt","lq"],
                    help="Post-process: match pred CbCr mean/std to ref")
    ap.add_argument("--vjepa_mode", default="correct",
                    choices=["correct","zero","shuffle_tokens"],
                    help="V-JEPA causality test: zero=null embed, shuffle_tokens=permuted")
    ap.add_argument("--steps", type=int, default=1,
                    help="Inference steps: 1=1-NFE, 4=4-step Euler ODE")
    main(ap.parse_args())
