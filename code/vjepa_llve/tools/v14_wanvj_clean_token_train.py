"""WanVJ-LLVE Phase 2 v14 — counterfactual clean-token prior.

7 architectural / training changes from Phase 1:
  (1) [REMOVED in cleanup] ControlNet was abandoned in v4+ (negative gain in v3/v3.1/v3.2)
  (2) DiT LayerNorm gamma/beta unfrozen                                          ~0.1M
  (3) LoRA rank 32 -> 64 on q/k/v/o + ffn.0/2                                    ~88M
  (4) V-JEPA projector: Linear -> 2-layer GELU MLP + learned PE                 ~14M
  (5) Loss: 5 losses (flow + L1 + chroma_YCbCr + lpips + repa + temporal)
  (6) DOVE t=399 inference start point (best free win for 1-NFE)
  (7) HASTE REPA: only first 8 DiT blocks, drop @ iter 22000
  (8) From scratch (NO ckpt load), 25000 iter, EMA 0.999, 1k ramp

Plus:
  - In-training eval every 2500 iter on 3 seqs (pair31/pair52/pair70) with viz
  - Final eval on 11 SDSD held-out seqs (pair68 excluded for data quality)
  - AdaIN-lq chroma post-processing default
  - BVI:SDSD = 1:1 alternation training
  - Continue from Phase 1's torch.nn.RMSNorm shim, SDPA enable_gqa shim, etc.

Train trainable: ~146M / Frozen: 1.85B
Memory peak: ~62 GB on 80GB A100
Wall time: ~25 hr

v14 adds a Counterfactual Clean-Token Adapter:
  z_LQ(V-JEPA) -> z_clean_hat ~= z_HQ(V-JEPA)
The restored Wan DiT is conditioned on z_clean_hat instead of raw low-light
tokens. This targets the token-content bottleneck directly rather than only
changing the injection interface.
"""
from __future__ import annotations
import argparse, os, sys, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/vjepa_llve")

# --- shim torch.nn.RMSNorm + SDPA enable_gqa for torch 2.3 ---
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
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(self.weight.dtype) * self.weight if self.weight is not None else x
if not hasattr(nn, 'RMSNorm'):
    nn.RMSNorm = _RMSNorm

_orig_sdpa = F.scaled_dot_product_attention
def _sdpa_no_gqa(*args, enable_gqa=None, **kw):
    return _orig_sdpa(*args, **kw)
F.scaled_dot_product_attention = _sdpa_no_gqa

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from diffusers import AutoencoderKLWan, WanTransformer3DModel
from models.vjepa_backbone import build_vjepa_backbone
from tools.iqa_metrics import compute_iqa, fmt_iqa, save_compare_video
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           build_split_seqs)
from data.bvi_rlv_dataset import BVIRLVDataset

WAN_PATH = "/cache/zihaohe/checkpoint/wan21_t2v_1_3b"
BVI_RLV_PATH = "/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos"
DIT_HIDDEN = 1536          # Wan DiT inner hidden
WAN_LATENT_CH = 16
DIT_BLOCKS = 30
REPA_BLOCKS = list(range(8))        # first 8 blocks for REPA alignment


# =============== YCbCr chroma utils (post-processing) ===============
_RGB2YCBCR = torch.tensor([[ 0.299,  0.587,  0.114],
                           [-0.169, -0.331,  0.500],
                           [ 0.500, -0.419, -0.081]], dtype=torch.float32)
_YCBCR2RGB = torch.tensor([[ 1.0,  0.0,    1.402  ],
                           [ 1.0, -0.344, -0.714 ],
                           [ 1.0,  1.772,  0.0   ]], dtype=torch.float32)
def rgb_to_ycbcr(x):
    M = _RGB2YCBCR.to(x.device, x.dtype)
    return torch.einsum('cd,bdthw->bcthw', M, x)
def ycbcr_to_rgb(x):
    M = _YCBCR2RGB.to(x.device, x.dtype)
    return torch.einsum('cd,bdthw->bcthw', M, x).clamp(0, 1)
def chroma_loss_ycbcr(pred_rgb, gt_rgb):
    return F.l1_loss(rgb_to_ycbcr(pred_rgb)[:, 1:], rgb_to_ycbcr(gt_rgb)[:, 1:])
def chroma_adain_lq(pred_rgb, lq_rgb):
    pred_y = rgb_to_ycbcr(pred_rgb); ref_y = rgb_to_ycbcr(lq_rgb)
    out = pred_y.clone()
    for c in (1, 2):
        p, r = pred_y[:, c], ref_y[:, c]
        pm, ps = p.mean(), p.std() + 1e-6
        rm, rs = r.mean(), r.std() + 1e-6
        out[:, c] = (p - pm) / ps * rs + rm
    return ycbcr_to_rgb(out).clamp(0, 1)


# =============== V-JEPA projector v3: 2-layer MLP + learned PE (#4) ===============
class VJepaProjectorV3(nn.Module):
    def __init__(self, vj_dim=1024, text_dim=4096, n_tokens=2048, use_pe=False):
        super().__init__()
        # PE default OFF (shuffle ablation showed token order doesnt help)
        self.use_pe = use_pe
        if use_pe:
            self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, vj_dim))
            nn.init.normal_(self.pos_emb, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(vj_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )
    def forward(self, z_vj):
        x = z_vj + self.pos_emb if self.use_pe else z_vj
        return self.mlp(x)


class QFormerBlock(nn.Module):
    """One Q-Former block: cross-attn(q→kv) + self-attn(q→q) + FFN."""
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    def forward(self, q, kv):
        q_n = self.norm_q1(q); kv_n = self.norm_kv(kv)
        out, _ = self.cross_attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + out
        q_n = self.norm_q2(q)
        out, _ = self.self_attn(q_n, q_n, q_n, need_weights=False)
        q = q + out
        q = q + self.ffn(self.norm_ffn(q))
        return q


class QFormerProjector(nn.Module):
    """Q-Former resampler: V-JEPA (B,2048,1024) → N learnable queries → (B,N,4096).
    
    BLIP-2 / IP-Adapter style. Compresses dense V-JEPA tokens into a small set of
    informationally-rich query tokens that DiT cross-attn can use efficiently.
    """
    def __init__(self, vj_dim=1024, text_dim=4096, hidden_dim=512,
                 n_queries=128, n_heads=8, n_layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)
        self.kv_proj = nn.Linear(vj_dim, hidden_dim)
        self.layers = nn.ModuleList([
            QFormerBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, text_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
    def forward(self, z_vj):
        B = z_vj.shape[0]
        kv = self.kv_proj(z_vj)         # (B, 2048, hidden)
        q = self.queries.expand(B, -1, -1)  # (B, n_queries, hidden)
        for layer in self.layers:
            q = layer(q, kv)
        return self.out_proj(self.out_norm(q))  # (B, n_queries, text_dim)


# =============== v8 PAC: Predictor-Aligned Conditioning (warm-start) ===============
class PACProjector(nn.Module):
    """v8: Predictor-Aligned Conditioning projector.

    Refinement transformer warm-started from V-JEPA 2.1 predictor weights.
    Pipeline:
      z_lq (B, N, 1024)
        --in_proj-->     (B, N, 384)         [warm-start from predictor_embed]
        --refine_blocks->(B, N, 384)         [warm-start from predictor_blocks[0..n-1]]
        --out_proj-->    (B, N, 1024)        [random init, residual added]
        residual:        z_lq + delta
        --mlp-->         (B, N, 4096)        [text-emb, same as v5/v6]

    Novelty: first work to leverage V-JEPA 2 predictor weights downstream
    as a warm-start prior for feature refinement before cross-attention conditioning.
    """
    def __init__(self, vj_dim=1024, text_dim=4096, n_refine_blocks=2,
                 predictor_dim=384, n_heads=12,
                 vjepa2_repo="/cache/zihaohe/modelzoo/vjepa2-main",
                 predictor_state_dict=None):
        super().__init__()
        self.vj_dim = vj_dim
        self.predictor_dim = predictor_dim
        # Pre-projection: 1024 -> 384 (warm-start from predictor_embed)
        self.in_proj = nn.Linear(vj_dim, predictor_dim, bias=True)
        # Refinement blocks (warm-start from V-JEPA predictor_blocks[0..n_refine-1])
        if vjepa2_repo not in sys.path:
            sys.path.insert(0, vjepa2_repo)
        from app.vjepa_2_1.models.utils.modules import Block as _VJEPA_Block
        from functools import partial as _partial
        self.refine_blocks = nn.ModuleList([
            _VJEPA_Block(
                dim=predictor_dim, num_heads=n_heads, mlp_ratio=4.0,
                qkv_bias=True, norm_layer=_partial(nn.LayerNorm, eps=1e-6),
                use_rope=False,  # we apply on flat token sequence, no RoPE
            )
            for _ in range(n_refine_blocks)
        ])
        # Post-projection: 384 -> 1024 (random init, NOT from predictor_proj which is 384->1664)
        self.out_proj = nn.Linear(predictor_dim, vj_dim, bias=True)
        # Standard MLP projector (encoder dim -> text dim), same as VJepaProjectorV3
        self.mlp = nn.Sequential(
            nn.Linear(vj_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )
        # Warm-start
        if predictor_state_dict is not None:
            self._warmstart(predictor_state_dict, n_refine_blocks)

    def _warmstart(self, sd, n_refine_blocks):
        """Load V-JEPA 2.1 predictor weights into in_proj + first n_refine_blocks."""
        # Strip 'module.backbone.' prefix
        clean = {}
        for k, v in sd.items():
            nk = k
            for prefix in ("module.backbone.", "module.", "backbone."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break
            clean[nk] = v

        # in_proj <- predictor_embed (1024 -> 384, single Linear since hierarchical=1)
        if 'predictor_embed.weight' in clean:
            self.in_proj.weight.data.copy_(clean['predictor_embed.weight'])
            if 'predictor_embed.bias' in clean:
                self.in_proj.bias.data.copy_(clean['predictor_embed.bias'])
            print(f'[PAC] in_proj warm-started from predictor_embed')

        # refine_blocks <- predictor_blocks[0..n-1]
        loaded = 0
        for i, blk in enumerate(self.refine_blocks):
            sub_keys = [
                ('attn.qkv.weight', 'attn.qkv.weight'),
                ('attn.qkv.bias',   'attn.qkv.bias'),
                ('attn.proj.weight','attn.proj.weight'),
                ('attn.proj.bias',  'attn.proj.bias'),
                ('mlp.fc1.weight',  'mlp.fc1.weight'),
                ('mlp.fc1.bias',    'mlp.fc1.bias'),
                ('mlp.fc2.weight',  'mlp.fc2.weight'),
                ('mlp.fc2.bias',    'mlp.fc2.bias'),
                ('norm1.weight',    'norm1.weight'),
                ('norm1.bias',      'norm1.bias'),
                ('norm2.weight',    'norm2.weight'),
                ('norm2.bias',      'norm2.bias'),
            ]
            for src_sub, dst_sub in sub_keys:
                src = f'predictor_blocks.{i}.{src_sub}'
                if src in clean:
                    parts = dst_sub.split('.')
                    target = blk
                    for p in parts[:-1]:
                        target = getattr(target, p)
                    target_param = getattr(target, parts[-1])
                    if target_param.shape == clean[src].shape:
                        target_param.data.copy_(clean[src])
                        loaded += 1
        print(f'[PAC] {loaded} weights warm-started across {n_refine_blocks} refine blocks')

    def forward(self, z_lq):
        # z_lq: (B, N, vj_dim)
        x = self.in_proj(z_lq)                      # (B, N, predictor_dim)
        for blk in self.refine_blocks:
            x, _ = blk(x)                           # (B, N, predictor_dim) — Block returns (x, attn)
        delta = self.out_proj(x)                    # (B, N, vj_dim)
        z_refined = z_lq + delta                    # residual
        return self.mlp(z_refined)                  # (B, N, text_dim)


# =============== v9 TruePAC: full V-JEPA predictor (frozen, proper interface) ===============
class TruePACProjector(nn.Module):
    """v9: TRUE Predictor-Aligned Conditioning.

    Uses the FULL V-JEPA 2.1 predictor (12 blocks, RoPE, mask tokens,
    modality embedding) AS-IS, frozen. We pass z_lq as visible context with
    a tiny dummy target so the predictor runs in its trained regime, then
    use return_all_tokens=True to obtain the refined context features
    (via predictor_proj_context, output dim 1664).

    Only trainable: text_proj (1664 -> text_dim=4096). ~10M new params.
    Predictor weights stay frozen (22.97M, all from V-JEPA 2.1 ckpt).
    """
    def __init__(self, vj_dim=1024, text_dim=4096,
                 vjepa2_repo="/cache/zihaohe/modelzoo/vjepa2-main",
                 predictor_state_dict=None,
                 image_size=256, patch_size=16, num_frames=16, tubelet_size=2,
                 predictor_embed_dim=384, predictor_depth=12,
                 predictor_num_heads=12, num_mask_tokens=8,
                 teacher_embed_dim=1664):
        super().__init__()
        if vjepa2_repo not in sys.path:
            sys.path.insert(0, vjepa2_repo)
        from app.vjepa_2_1.models.predictor import vit_predictor

        # Build FULL predictor matching V-JEPA 2.1 ckpt config
        self.predictor = vit_predictor(
            img_size=(image_size, image_size),
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=vj_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            use_mask_tokens=True,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=True,
            teacher_embed_dim=teacher_embed_dim,
            n_output_distillation=1,
            modality_embedding=True,
            img_temporal_dim_size=1,        # enable modality embed (need non-None)
            use_rope=True,                  # CRITICAL: weights trained with RoPE
            return_all_tokens=True,         # get refined context, not just predicted target
        )
        # Load predictor weights (full warm-start)
        if predictor_state_dict is not None:
            cleaned = {}
            for k, v in predictor_state_dict.items():
                nk = k
                for prefix in ("module.backbone.", "module.", "backbone."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                        break
                cleaned[nk] = v
            miss, unexp = self.predictor.load_state_dict(cleaned, strict=False)
            print(f"[TruePAC] predictor loaded: missing={len(miss)} unexpected={len(unexp)}")
            if unexp:
                print(f"  unexpected first 5: {list(unexp)[:5]}")

        # Freeze predictor (it's already trained, don't break it)
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.predictor.eval()

        # Token grid (for mask construction)
        self.num_patches = (num_frames // tubelet_size) * (image_size // patch_size) ** 2
        # text projector (trainable): 1664 -> text_dim
        self.text_proj = nn.Sequential(
            nn.Linear(teacher_embed_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )

    def forward(self, z_lq):
        """z_lq: (B, N, vj_dim=1024) from V-JEPA encoder layer 23.

        Run predictor with: masks_x = all positions, masks_y = single dummy position.
        Use return_all_tokens=True to extract refined context features.
        """
        B, N, D = z_lq.shape
        assert N == self.num_patches, f"expected {self.num_patches} tokens, got {N}"
        device = z_lq.device

        # masks_x: ALL positions (everything is visible context)
        # masks_y: 1 dummy position (predictor needs at least 1 target to run)
        all_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        masks_x = all_idx                              # (B, N) — all context
        masks_y = all_idx[:, :1]                       # (B, 1) — minimal target

        # Run predictor (frozen). It returns (x_pred, x_context) since return_all_tokens=True
        # x_context: (B, N, teacher_embed_dim=1664) — refined context via predictor_proj_context
        with torch.no_grad():
            x_pred, x_context = self.predictor(z_lq, masks_x, masks_y, mod="video", mask_index=0)

        # x_context is what we want: refined version of z_lq, projected to teacher dim
        # Use it as condition (replaces raw z_lq)
        return self.text_proj(x_context)              # (B, N, text_dim)


# =============== LoRA on Wan attn linears (#3, rank 64) ===============
class SpatialVJepaAdaLN(nn.Module):
    """v13a (C1+): Spatial-aware AdaLN-Zero — γ/β vary per DiT token position.

    V-JEPA features (B, N_vj=2048, vj_dim) reshaped to (B, T_vj, H_vj, W_vj, vj_dim),
    temporally pooled 2x to match DiT grid (T_dit=4, H_dit=16, W_dit=16), then
    per-token MLP outputs (γ, β) ∈ R^{B, N_dit, 2*dit_hidden} per block.

    Replaces v12a's pooled-AdaLN which broadcast a single (γ, β) across all tokens.
    Now each DiT spatial-temporal patch gets V-JEPA-derived modulation specific
    to its location, enabling object/region-aware enhancement.

    Init: zero-init per_block last linear → γ=β=0 → v9-equivalent at iter 0.
    """
    def __init__(self, vj_dim=1024, hidden=256, n_blocks=30, dit_hidden=1536,
                 vj_grid=(8, 16, 16), dit_grid=(4, 16, 16)):
        super().__init__()
        self.vj_grid = vj_grid
        self.dit_grid = dit_grid
        self.dit_hidden = dit_hidden
        T_vj, H_vj, W_vj = vj_grid
        T_dit, H_dit, W_dit = dit_grid
        assert H_vj == H_dit and W_vj == W_dit, \
            f"spatial mismatch: V-JEPA {(H_vj,W_vj)} vs DiT {(H_dit,W_dit)}"
        assert T_vj % T_dit == 0, f"V-JEPA T={T_vj} must be divisible by DiT T={T_dit}"
        self.t_pool = T_vj // T_dit
        self.n_dit = T_dit * H_dit * W_dit
        # Shared trunk: 1024 → hidden
        self.trunk = nn.Sequential(
            nn.Linear(vj_dim, hidden),
            nn.SiLU(),
        )
        # Per-block heads (zero-init last linear)
        self.per_block = nn.ModuleList([
            nn.Linear(hidden, 2 * dit_hidden, bias=True) for _ in range(n_blocks)
        ])
        for layer in self.per_block:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, z_vj):
        """z_vj: (B, N_vj=2048, vj_dim=1024).
        Returns list of (B, N_dit, 2*dit_hidden) per block."""
        B, N_vj, D = z_vj.shape
        T_vj, H_vj, W_vj = self.vj_grid
        T_dit, H_dit, W_dit = self.dit_grid
        # Reshape to spatial-temporal grid
        z = z_vj.view(B, T_vj, H_vj, W_vj, D)
        # Temporal pool 2x: mean over t_pool consecutive frames
        z = z.view(B, T_dit, self.t_pool, H_vj, W_vj, D).mean(dim=2)
        # (B, T_dit, H_dit, W_dit, D) → flatten → (B, N_dit, D)
        z = z.reshape(B, self.n_dit, D)
        # Trunk + per-block heads
        h = self.trunk(z)
        return [layer(h) for layer in self.per_block]


class VJepaAdaLN(nn.Module):
    """v12a (C1): AdaLN-Zero modulation from V-JEPA pooled features.

    Pools V-JEPA features (B, N, vj_dim) → (B, vj_dim) via mean. Per DiT
    block, MLP outputs (γ, β) ∈ R^{2 * dit_hidden}. Applied in forward hook
    AFTER DiT block output: x = x * (1 + γ.unsqueeze(1)) + β.unsqueeze(1).

    Init: zero-init last linear of each per-block head → γ=β=0 → no
    modulation → v9-equivalent at iter 0.
    """
    def __init__(self, vj_dim=1024, hidden=256, n_blocks=30, dit_hidden=1536):
        super().__init__()
        self.n_blocks = n_blocks
        self.dit_hidden = dit_hidden
        self.trunk = nn.Sequential(
            nn.Linear(vj_dim, hidden),
            nn.SiLU(),
        )
        self.per_block = nn.ModuleList([
            nn.Linear(hidden, 2 * dit_hidden, bias=True) for _ in range(n_blocks)
        ])
        # Zero-init last linear of each per-block head → γ=β=0 init
        for layer in self.per_block:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, z_vj):
        """z_vj: (B, N, vj_dim) or (B, vj_dim). Pools internally if 3D.
        Returns list of (B, 2*dit_hidden) per block."""
        if z_vj.dim() == 3:
            z_vj = z_vj.mean(dim=1)  # pool to (B, vj_dim)
        h = self.trunk(z_vj)
        return [layer(h) for layer in self.per_block]


class LQCondResBlock3D(nn.Module):
    """v11 (A2): Active LQ feature extractor that produces an additive residual
    to inject into DiT input. Replaces the implicit-only LQ pathway (z_t blend)
    with a learned channel-mixed residual.

    Architecture: 2 ResBlock3D in latent (16-ch), zero-init final 1x1 Conv3D.
    Init equivalent to v9 (output = 0).

    Forward signature: lq_cond(z_lq) → residual same shape.
    """
    def __init__(self, ch=16, hidden=64):
        super().__init__()
        self.in_proj = nn.Conv3d(ch, hidden, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.out_proj = nn.Conv3d(hidden, ch, kernel_size=1)
        # Zero-init last layer → initial output = 0 → v9 equivalent
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_lq):
        # z_lq: (B, C=16, T_lat, H_lat, W_lat) in fp32
        h = self.in_proj(z_lq)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_proj(h)


class MultiLayerReducer(nn.Module):
    """v10: Channel-concat reducer fusing multiple V-JEPA layers into single
    V-JEPA-equivalent feature for downstream TruePAC predictor.

    Init: zero-init reducer with identity slot for the deep layer position so
    output equals deep layer at iter 0 (preserves v9 starting point exactly).
    """
    def __init__(self, vj_dim=1024, n_layers=3, deep_pos=2):
        super().__init__()
        self.vj_dim = vj_dim
        self.n_layers = n_layers
        self.deep_pos = deep_pos
        self.reducer = nn.Linear(vj_dim * n_layers, vj_dim, bias=True)
        with torch.no_grad():
            self.reducer.weight.zero_()
            self.reducer.bias.zero_()
            i_start = deep_pos * vj_dim
            self.reducer.weight[:, i_start:i_start + vj_dim] = torch.eye(vj_dim)

    def forward(self, feats_list):
        x = torch.cat(feats_list, dim=-1)
        return self.reducer(x)


class CleanTokenAdapter(nn.Module):
    """v14: predict a counterfactual clean-scene V-JEPA token field.

    Input tokens still come from the degraded low-light clip. The adapter learns
    a residual delta toward the HQ/normal-light V-JEPA tokens, plus a demand map
    that localizes where strong semantic correction is needed.

    Init is conservative: delta projection is zero, so z_clean_hat == z_lq at
    iter 0. The adapter then learns from an explicit z_clean_hat -> z_HQ token
    loss while the Wan DiT adapts to the cleaned prior.
    """
    def __init__(self, vj_dim=1024, hidden=256, depth=2,
                 vj_grid=(8, 16, 16), init_gate=-2.0):
        super().__init__()
        self.vj_grid = vj_grid
        self.hidden = hidden
        self.norm = nn.LayerNorm(vj_dim)
        self.down = nn.Linear(vj_dim, hidden)
        self.mix_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.temporal_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.spatial_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Linear(hidden * 4, hidden),
            )
            for _ in range(depth)
        ])
        self.delta = nn.Linear(hidden, vj_dim)
        self.demand = nn.Linear(hidden, 1)
        self.gate_logit = nn.Parameter(torch.tensor(float(init_gate)))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.demand.weight)
        nn.init.zeros_(self.demand.bias)

    def forward(self, z_lq, return_aux=False):
        B, N, D = z_lq.shape
        T, H, W = self.vj_grid
        assert N == T * H * W, f"expected {T*H*W} tokens, got {N}"
        h = self.down(self.norm(z_lq))
        for mix_norm, temporal_proj, spatial_proj, ffn in zip(
                self.mix_norms, self.temporal_proj, self.spatial_proj, self.ffns):
            hg = h.view(B, T, H, W, self.hidden)
            # Pure tensor/linear token mixer: avoids cuDNN Conv3d while still
            # injecting global temporal and per-location spatial context.
            t_ctx = hg.mean(dim=(2, 3), keepdim=True).expand_as(hg).reshape(B, N, self.hidden)
            s_ctx = hg.mean(dim=1, keepdim=True).expand_as(hg).reshape(B, N, self.hidden)
            h = h + temporal_proj(mix_norm(t_ctx)) + spatial_proj(mix_norm(s_ctx))
            h = h + ffn(h)
        demand = torch.sigmoid(self.demand(h))
        gate = torch.sigmoid(self.gate_logit)
        delta = self.delta(h)
        z_clean = z_lq + gate * demand * delta
        if return_aux:
            return z_clean, {"demand": demand, "gate": gate, "delta": delta}
        return z_clean


def clean_token_losses(z_clean, z_target):
    """Token-level counterfactual prior losses.

    Cosine is the primary objective because V-JEPA token scale can drift across
    clips. Smooth L1 is a weak stabilizer controlled by a separate lambda.
    """
    target = z_target.detach()
    zc = F.normalize(z_clean.float(), dim=-1)
    zt = F.normalize(target.float(), dim=-1)
    l_cos = 1.0 - (zc * zt).sum(dim=-1).mean()
    l_l1 = F.smooth_l1_loss(z_clean.float(), target.float())
    return l_cos, l_l1


class LoRALinear(nn.Module):
    def __init__(self, base, rank=64, alpha=64.0):
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


class DoRALinear(nn.Module):
    """DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al. ICML 2024).
    
    W_eff = m * (W_0 + scale * BA) / ||W_0 + scale * BA||_c
    where m is learnable per-output-dim magnitude vector, init to ||W_0||_c.
    """
    def __init__(self, base, rank=64, alpha=64.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # DoRA magnitude vector init = ||W_0||_c (column-wise L2 norm)
        with torch.no_grad():
            base_w_norm = self.base.weight.float().norm(dim=1, keepdim=False)  # (out_f,)
        self.m = nn.Parameter(base_w_norm.clone())
    
    def forward(self, x):
        # Compute combined weight = W_0 + scale * BA
        delta = (self.B @ self.A) * self.scaling  # (out_f, in_f)
        W_combined = self.base.weight + delta.to(self.base.weight.dtype)
        # Column-wise L2 norm + magnitude rescale
        W_norm = W_combined.float().norm(dim=1, keepdim=True) + 1e-8  # (out_f, 1)
        W_eff = W_combined * (self.m.unsqueeze(1) / W_norm).to(W_combined.dtype)
        # Linear forward
        out = F.linear(x, W_eff, self.base.bias)
        return out


def inject_lora_on_dit(dit, rank=64, use_dora=False):
    Cls = DoRALinear if use_dora else LoRALinear
    n = 0
    for blk in dit.blocks:
        for attn_name in ("attn1", "attn2"):
            attn = getattr(blk, attn_name, None)
            if attn is None: continue
            for proj in ("to_q", "to_k", "to_v"):
                lin = getattr(attn, proj, None)
                if isinstance(lin, nn.Linear):
                    setattr(attn, proj, Cls(lin, rank=rank))
                    n += 1
            # to_out is a ModuleList of [Linear, Dropout]
            if hasattr(attn, "to_out") and isinstance(attn.to_out[0], nn.Linear):
                attn.to_out[0] = Cls(attn.to_out[0], rank=rank)
                n += 1
        # FFN
        ffn = getattr(blk, "ffn", None)
        if ffn is not None:
            if isinstance(ffn.net[0], nn.Linear):
                ffn.net[0] = Cls(ffn.net[0], rank=rank); n += 1
            if isinstance(ffn.net[2], nn.Linear):
                ffn.net[2] = Cls(ffn.net[2], rank=rank); n += 1
    return n


def unfreeze_dit_norm(dit):
    """#2: unfreeze LayerNorm/RMSNorm gamma/beta for fine-tuning."""
    n = 0
    for name, p in dit.named_parameters():
        if any(s in name.lower() for s in ['norm']) and ('weight' in name or 'bias' in name):
            p.requires_grad_(True)
            n += p.numel()
    return n


# =============== Wan VAE helpers ===============
def get_wan_latents_norm(vae, device):
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, 16, 1, 1, 1).float()
    std = (1.0 / torch.tensor(vae.config.latents_std, device=device).view(1, 16, 1, 1, 1).float())
    return mean, std

def wan_encode(vae, x_rgb, mean, std):
    x = x_rgb * 2 - 1
    z = vae.encode(x).latent_dist.mode()
    z = (z - mean) * std
    return z

def wan_decode(vae, z_norm, mean, std):
    z = z_norm / std + mean
    x = vae.decode(z).sample
    return ((x + 1) / 2).clamp(0, 1)


# =============== Loss helpers ===============
def temporal_loss(pred, gt):
    """MSE on consecutive frame differences (preserves temporal stability)."""
    if pred.shape[2] < 2:
        return torch.tensor(0.0, device=pred.device)
    pd = pred[:, :, 1:] - pred[:, :, :-1]
    gd = gt[:, :, 1:] - gt[:, :, :-1]
    return F.mse_loss(pd, gd)

def build_lpips(device):
    import pyiqa
    m = pyiqa.create_metric("lpips", as_loss=True, device=device)
    for p in m.parameters(): p.requires_grad_(False)
    return m

def lpips_video_sample(lpips_m, pred, gt, n_frames=4):
    T = pred.shape[2]
    idx = torch.linspace(0, T - 1, n_frames).long()
    p = pred[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, 3, pred.shape[3], pred.shape[4])
    g = gt[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, 3, gt.shape[3], gt.shape[4])
    return lpips_m(p, g).mean()


# =============== Inference (DOVE t=399 + AdaIN) (#6) ===============
@torch.no_grad()
def wanvj_inference(vae, dit, vj2text, backbone, lq, mean, std,
                    ref_layer=3, dove_t=399, total_t=1000, apply_adain=True,
                    n_steps=1, multilayer_reducer=None, ml_indices=None,
                    no_vjepa=False, lq_cond=None,
                    vjepa_adaln=None, adaln_holder=None,
                    clean_token_adapter=None):
    """1-NFE (default) or N-step Euler ODE inference along rectified flow.

    v12a: if vjepa_adaln is set, also computes pooled V-JEPA → AdaLN params
    and applies in DiT block hooks via adaln_holder.
    """
    z_lq = wan_encode(vae, lq, mean, std)
    B = z_lq.shape[0]
    lq_resid = lq_cond(z_lq) if lq_cond is not None else None
    if no_vjepa:
        text_emb = torch.zeros(B, 2048, 4096, device=lq.device)
        z_vj = None
    else:
        if multilayer_reducer is not None:
            feats_all = encode(backbone, lq)
            feats_sel = [feats_all[i].float() for i in ml_indices]
            z_vj = multilayer_reducer(feats_sel)
        else:
            z_vj = encode(backbone, lq)[ref_layer].float()
        if clean_token_adapter is not None:
            z_vj = clean_token_adapter(z_vj)
        text_emb = vj2text(z_vj)
    # v12a: compute AdaLN params from pooled V-JEPA
    if vjepa_adaln is not None and z_vj is not None:
        # Pass z_vj directly; VJepaAdaLN pools, SpatialVJepaAdaLN keeps spatial
        adaln_params = vjepa_adaln(z_vj)  # list of params per block
        adaln_holder['params'] = adaln_params
        adaln_holder['active'] = True
    handles = []
    try:
        if n_steps == 1:
            t = torch.zeros(B, device=lq.device).long()
            dit_in = z_lq + lq_resid if lq_resid is not None else z_lq
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_pred = dit(dit_in.to(torch.bfloat16), t,
                             encoder_hidden_states=text_emb.to(torch.bfloat16),
                             return_dict=False)[0].float()
            z_hq = z_lq + v_pred
        else:
            dt = 1.0 / n_steps
            z = z_lq.clone()
            for step in range(n_steps):
                t_val = step * dt
                t = torch.full((B,), int(round(t_val * total_t)), device=lq.device).long()
                dit_in = z + lq_resid if lq_resid is not None else z
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v = dit(dit_in.to(torch.bfloat16), t,
                            encoder_hidden_states=text_emb.to(torch.bfloat16),
                            return_dict=False)[0].float()
                z = z + dt * v
            z_hq = z
    finally:
        for h in handles: h.remove()
        if adaln_holder is not None:
            adaln_holder['active'] = False
    pred = wan_decode(vae, z_hq, mean, std).float()
    if apply_adain:
        T_out = pred.shape[2]
        T_drop = (lq.shape[2] - T_out) // 2
        lq_trim = lq[:, :, T_drop:T_drop + T_out]
        pred = chroma_adain_lq(pred, lq_trim).clamp(0, 1)
    return pred


# =============== AdaLN-Zero V-JEPA hooks (v12a / v13a) ===============
def _attach_adaln_hooks(dit, holder):
    """Forward-hook each DiT block to apply v_jepa AdaLN-Zero modulation.

    holder = {'active': bool, 'params': list of tensors per block}
      v12a pooled: each is (B, 2*hidden) — broadcast across all tokens
      v13a spatial: each is (B, N_dit, 2*hidden) — per-token modulation
    When active=True, modifies output: out = out * (1 + γ) + β
    """
    handles = []
    for blk_idx, block in enumerate(dit.blocks):
        def make_hook(idx):
            def hook(module, inputs, output):
                if not holder['active']: return output
                gb = holder['params'][idx]
                gamma, beta = gb.chunk(2, dim=-1)
                out_dtype = output[0].dtype if isinstance(output, tuple) else output.dtype
                # If pooled (B, hidden), add token dim; if spatial (B, N, hidden), keep
                if gamma.dim() == 2:
                    gamma = gamma.unsqueeze(1)
                    beta = beta.unsqueeze(1)
                gamma = gamma.to(out_dtype)
                beta = beta.to(out_dtype)
                if isinstance(output, tuple):
                    h = output[0]
                    new_h = h * (1 + gamma) + beta
                    return (new_h,) + output[1:]
                else:
                    return output * (1 + gamma) + beta
            return hook
        handles.append(block.register_forward_hook(make_hook(blk_idx)))
    return handles


# =============== EMA ===============
class EMAState:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Only shadow trainable params (saves ~5GB on Wan DiT 1.3B)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.shadow = {k: v.detach().float().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point and k in trainable}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().to(self.shadow[k].dtype), alpha=1 - self.decay)


# =============== In-training eval (#X) ===============
@torch.no_grad()
def in_training_eval(it, vae, dit, vj2text, backbone, mean, std,
                     test_seqs_subset, csv_writer, viz_root, args,
                     multilayer_reducer=None, lq_cond=None,
                     vjepa_adaln=None, adaln_holder=None,
                     clean_token_adapter=None):
    dit.eval()
    if vj2text is not None: vj2text.eval()
    if multilayer_reducer is not None: multilayer_reducer.eval()
    if lq_cond is not None: lq_cond.eval()
    if vjepa_adaln is not None: vjepa_adaln.eval()
    if clean_token_adapter is not None: clean_token_adapter.eval()
    ml_indices = ([int(i) for i in args.ml_indices.split(",")]
                  if multilayer_reducer is not None else None)
    rows = []
    for name, lq_p, gt_p in test_seqs_subset:
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(args.device), gt.to(args.device)
        pred = wanvj_inference(vae, dit, vj2text, backbone, lq, mean, std,
                               ref_layer=args.ref_layer, dove_t=args.dove_t,
                               apply_adain=False, n_steps=args.inference_steps,
                               multilayer_reducer=multilayer_reducer,
                               ml_indices=ml_indices, no_vjepa=args.no_vjepa,
                               lq_cond=lq_cond,
                               vjepa_adaln=vjepa_adaln, adaln_holder=adaln_holder,
                               clean_token_adapter=clean_token_adapter)
        T_out = pred.shape[2]
        T_drop = (gt.shape[2] - T_out) // 2
        gt_trim = gt[:, :, T_drop:T_drop + T_out]
        lq_trim = lq[:, :, T_drop:T_drop + T_out]
        psnr = mean_psnr(to_u8(pred), to_u8(gt_trim))
        rows.append((name, psnr))
        # Save viz
        viz_dir = os.path.join(viz_root, f"iter_{it:06d}")
        os.makedirs(viz_dir, exist_ok=True)
        save_compare_video(os.path.join(viz_dir, f"{name}.mp4"),
                           lq_trim, pred, gt_trim, fps=10)
    mean_psnr_v = float(np.mean([p for _, p in rows]))
    print(f"[in-train eval iter {it}] mean PSNR={mean_psnr_v:.3f} per-seq: " +
          " ".join(f"{n}={p:.2f}" for n, p in rows))
    csv_writer.writerow([it, mean_psnr_v] + [p for _, p in rows])
    dit.train()
    if vj2text is not None: vj2text.train()
    if multilayer_reducer is not None: multilayer_reducer.train()
    if lq_cond is not None: lq_cond.train()
    if vjepa_adaln is not None: vjepa_adaln.train()
    if clean_token_adapter is not None: clean_token_adapter.train()
    return mean_psnr_v


# =============== Main training loop ===============
def main(args):
    device = torch.device("cuda")
    args.device = device
    os.makedirs(args.out_dir, exist_ok=True)
    viz_root = args.viz_dir or f"/home/ma-user/workspace/zihaohe/visualizations/{os.path.basename(args.out_dir)}"
    os.makedirs(viz_root, exist_ok=True)

    print("[v3] loading frozen backbones ...")
    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae",
                                           torch_dtype=torch.float32).to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    dit = WanTransformer3DModel.from_pretrained(WAN_PATH, subfolder="transformer",
                                                torch_dtype=torch.bfloat16).to(device).eval()
    for p in dit.parameters(): p.requires_grad_(False)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    # === Trainable adapters (all from-scratch init) ===
    if args.no_lora:
        # Wan-Full baseline: full DiT FT, no LoRA injection
        n_lora = 0
        for p in dit.parameters(): p.requires_grad_(True)
        n_full = sum(p.numel() for p in dit.parameters() if p.requires_grad)
        print(f"[wanfull] Full DiT FT mode: {n_full/1e6:.1f}M params unfrozen, no LoRA")
    else:
        n_lora = inject_lora_on_dit(dit, rank=args.lora_rank, use_dora=args.use_dora)
        dit.to(device)
        if args.freeze_norm:
            n_norm = 0
            print(f"[v5] DiT norm γ/β REFROZEN (Phase 1 style)")
        else:
            n_norm = unfreeze_dit_norm(dit)
            print(f"[v3] DiT norm gamma/beta unfrozen: {n_norm/1e6:.2f}M params")
        _adapt_name = "DoRA" if args.use_dora else "LoRA"
        print(f"[v3] {_adapt_name} modules: {n_lora}, rank={args.lora_rank}")

    if args.no_vjepa:
        # Track B fallback: zero text emb, no V-JEPA
        vj2text = None
        print("[v3-NoVJEPA] V-JEPA path DISABLED (zero encoder_hidden_states)")
    else:
        if args.use_truepac:
            _vj_full = torch.load(args.vjepa2_ckpt, map_location="cpu", weights_only=False)
            _pred_sd = _vj_full.get("predictor", {})
            vj2text = TruePACProjector(vj_dim=1024, text_dim=4096,
                                        vjepa2_repo=args.vjepa2_repo,
                                        predictor_state_dict=_pred_sd).to(device)
            print(f"[v9] V-JEPA projector: TruePAC (full 12-block predictor frozen + RoPE + mask)")
            del _vj_full, _pred_sd
        elif args.use_pac:
            # Load V-JEPA 2.1 predictor state_dict for warm-start
            _vj_full = torch.load(args.vjepa2_ckpt, map_location="cpu", weights_only=False)
            _pred_sd = _vj_full.get("predictor", {})
            vj2text = PACProjector(vj_dim=1024, text_dim=4096,
                                    n_refine_blocks=args.pac_n_refine,
                                    predictor_dim=384, n_heads=12,
                                    vjepa2_repo=args.vjepa2_repo,
                                    predictor_state_dict=_pred_sd).to(device)
            print(f"[v8] V-JEPA projector: PAC (n_refine={args.pac_n_refine} blocks "
                  f"warm-started from V-JEPA 2.1 predictor)")
            del _vj_full, _pred_sd
        elif args.use_qformer:
            vj2text = QFormerProjector(vj_dim=1024, text_dim=4096, hidden_dim=512,
                                       n_queries=args.qformer_n_queries,
                                       n_heads=8, n_layers=args.qformer_n_layers).to(device)
            print(f"[v7] V-JEPA projector: Q-Former (n_queries={args.qformer_n_queries}, "
                  f"n_layers={args.qformer_n_layers}, hidden=512)")
        else:
            vj2text = VJepaProjectorV3(vj_dim=1024, text_dim=4096, n_tokens=2048,
                                        use_pe=args.use_pe).to(device)
            print(f"[v3] V-JEPA projector v3: 2-layer MLP" +
                  (" + learned PE" if args.use_pe else " (no PE per shuffle finding)"))

    # v10: optional multi-layer V-JEPA fusion (channel-concat reducer)
    multilayer_reducer = None
    ml_indices = None
    if args.use_multilayer_vjepa and not args.no_vjepa:
        ml_indices = [int(i) for i in args.ml_indices.split(",")]
        if args.ref_layer not in ml_indices:
            raise ValueError(f"ref_layer={args.ref_layer} must be one of ml_indices={ml_indices}")
        deep_pos = ml_indices.index(args.ref_layer)
        multilayer_reducer = MultiLayerReducer(
            vj_dim=1024, n_layers=len(ml_indices), deep_pos=deep_pos,
        ).to(device)
        print(f"[v10] MultiLayerReducer: layer indices={ml_indices}, "
              f"deep_pos={deep_pos} (ref_layer={args.ref_layer}); "
              f"init=v9-equivalent (zero non-deep, identity deep)")

    # v11 (A2): optional active LQ conditioner (ResBlock3D + zero-init residual)
    lq_cond_module = None
    if args.use_lq_cond:
        lq_cond_module = LQCondResBlock3D(ch=WAN_LATENT_CH, hidden=args.lq_cond_hidden).to(device)
        n_lqc = sum(p.numel() for p in lq_cond_module.parameters())
        print(f"[v11-A2] LQCondResBlock3D: hidden={args.lq_cond_hidden}, "
              f"params={n_lqc/1e3:.1f}K (zero-init out_proj → v9-equivalent at init)")

    # v12a / v13a (C1): optional V-JEPA AdaLN-Zero modulation (pooled or spatial)
    vjepa_adaln = None
    adaln_holder = None
    adaln_handles = []
    if (args.use_vjepa_adaln or args.use_spatial_adaln) and not args.no_vjepa:
        n_blocks = len(dit.blocks)
        if args.use_spatial_adaln:
            vjepa_adaln = SpatialVJepaAdaLN(
                vj_dim=1024, hidden=args.adaln_hidden,
                n_blocks=n_blocks, dit_hidden=DIT_HIDDEN,
                vj_grid=(8, 16, 16), dit_grid=(4, 16, 16),
            ).to(device)
            tag = "v13a-C1+ SpatialVJepaAdaLN"
        else:
            vjepa_adaln = VJepaAdaLN(vj_dim=1024, hidden=args.adaln_hidden,
                                      n_blocks=n_blocks, dit_hidden=DIT_HIDDEN).to(device)
            tag = "v12a-C1 VJepaAdaLN (pooled)"
        n_aln = sum(p.numel() for p in vjepa_adaln.parameters())
        adaln_holder = {'active': False, 'params': None}
        adaln_handles = _attach_adaln_hooks(dit, adaln_holder)
        print(f"[{tag}] hidden={args.adaln_hidden}, n_blocks={n_blocks}, "
              f"params={n_aln/1e6:.2f}M (zero-init per_block → γ=β=0 → v9-equivalent)")

    # v14: Counterfactual Clean-Token Adapter, z_LQ(V-JEPA) -> z_clean_hat ~= z_HQ(V-JEPA)
    clean_token_adapter = None
    if args.use_clean_token_adapter and not args.no_vjepa:
        clean_token_adapter = CleanTokenAdapter(
            vj_dim=1024, hidden=args.clean_hidden, depth=args.clean_depth,
            vj_grid=(8, 16, 16), init_gate=args.clean_init_gate,
        ).to(device)
        n_clean = sum(p.numel() for p in clean_token_adapter.parameters())
        print(f"[v14] CleanTokenAdapter: hidden={args.clean_hidden}, "
              f"depth={args.clean_depth}, params={n_clean/1e6:.2f}M, "
              f"init_gate=sigmoid({args.clean_init_gate})")

    train_params = [p for p in dit.parameters() if p.requires_grad]
    if vj2text is not None:
        train_params += list(vj2text.parameters())
    if multilayer_reducer is not None:
        train_params += list(multilayer_reducer.parameters())
    if lq_cond_module is not None:
        train_params += list(lq_cond_module.parameters())
    if vjepa_adaln is not None:
        train_params += list(vjepa_adaln.parameters())
    if clean_token_adapter is not None:
        train_params += list(clean_token_adapter.parameters())
    n_train = sum(p.numel() for p in train_params)
    print(f"[v3] TOTAL trainable: {n_train/1e6:.2f}M")

    # === REPA proj (DiT block 7 hidden -> V-JEPA dim) ===
    repa_proj = nn.Linear(DIT_HIDDEN, 1024).to(device) if not args.no_vjepa else None
    if repa_proj is not None:
        train_params += list(repa_proj.parameters())
    # Hook on first 8 blocks for REPA
    _bottleneck = {}
    def _make_hook(idx):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            _bottleneck[idx] = h
        return hook
    repa_handles = []
    for bi in REPA_BLOCKS:
        repa_handles.append(dit.blocks[bi].register_forward_hook(_make_hook(bi)))

    # === LPIPS ===
    print("[v3] loading LPIPS ...")
    lpips_m = build_lpips(device)

    # === Optim + sched ===
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr * 0.05)

    # === EMA on dit + vj2text + multilayer_reducer + lq_cond + vjepa_adaln + clean adapter ===
    ema_dit = EMAState(dit, args.ema)
    ema_vj = EMAState(vj2text, args.ema) if vj2text is not None else None
    ema_ml = EMAState(multilayer_reducer, args.ema) if multilayer_reducer is not None else None
    ema_lqc = EMAState(lq_cond_module, args.ema) if lq_cond_module is not None else None
    ema_aln = EMAState(vjepa_adaln, args.ema) if vjepa_adaln is not None else None
    ema_clean = EMAState(clean_token_adapter, args.ema) if clean_token_adapter is not None else None

    # === Data ===
    bvi = BVIRLVDataset(BVI_RLV_PATH, split="train", clip_len=16,
                        crop_size=args.crop, is_train=True)
    sdsd_train_seqs, sdsd_test_seqs = build_split_seqs()
    # In-train eval subset: pair31 (best), pair52 (worst typical), pair70 (median)
    eval_subset = [s for s in sdsd_test_seqs if s[0] in {"pair31", "pair52", "pair70"}]
    print(f"[data] BVI: {len(bvi)} | SDSD train: {len(sdsd_train_seqs)} | "
          f"held-out: {len(sdsd_test_seqs)} | in-train eval subset: {len(eval_subset)}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    mean, std = get_wan_latents_norm(vae, device)

    # CSV for in-train eval
    eval_csv_path = os.path.join(args.out_dir, "eval_progress.csv")
    eval_csv = open(eval_csv_path, "w", newline="")
    csv_writer = csv.writer(eval_csv)
    csv_writer.writerow(["iter", "mean_psnr", "pair31", "pair52", "pair70"])
    eval_csv.flush()

    log = []
    _best_psnr = [0.0]  # mutable holder for closure-style update
    t_start = time.time()

    for it in range(args.iters):
        # 1:1 BVI:SDSD alternation
        source = "bvi" if (it % 2 == 0) else "sdsd"
        try:
            if source == "bvi":
                s = bvi[int(rng.integers(0, len(bvi)))]
                lq = s["lq"].unsqueeze(0).to(device)
                gt = s["hq"].unsqueeze(0).to(device)
            else:
                _, lq_p, gt_p = sdsd_train_seqs[int(rng.integers(0, len(sdsd_train_seqs)))]
                lq_t, gt_t = load_clip(lq_p, gt_p, crop_size=args.crop)
                lq = lq_t.to(device); gt = gt_t.to(device)
        except Exception as e:
            # fallback to bvi
            s = bvi[int(rng.integers(0, len(bvi)))]
            lq = s["lq"].unsqueeze(0).to(device)
            gt = s["hq"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_lq = wan_encode(vae, lq, mean, std)
            z_hq = wan_encode(vae, gt, mean, std)
            if not args.no_vjepa:
                feats_lq_all = encode(backbone, lq)
                feats_gt_all = encode(backbone, gt)
                if multilayer_reducer is not None:
                    feats_lq_sel = [feats_lq_all[i].float() for i in ml_indices]
                    feats_gt_sel = [feats_gt_all[i].float() for i in ml_indices]
                else:
                    feats_lq_sel = None
                    feats_gt_sel = None
                z_vj_single = feats_lq_all[args.ref_layer].float()
                z_vj_gt = feats_gt_all[args.ref_layer].float()

        B = z_lq.shape[0]
        # DOVE training mixin: with prob dove_prob, z_t = z_lq (claim t=0.399)
        # so model learns "given z_lq @ t=399, predict velocity v -> z_hq"
        # both inference modes (t=0 and t=399) become viable
        if args.dove_prob > 0 and rng.random() < args.dove_prob:
            t = torch.full((B,), args.dove_t / 1000.0, device=device)
            z_t = z_lq.clone()
            dove_active = True
        else:
            # v11 (A1): logit-normal t-sampling (SD3-style mid-noise weighting)
            if args.use_logit_normal_t:
                t = torch.sigmoid(torch.randn(B, device=device))
            else:
                t = torch.rand(B, device=device)
            tv = t.view(B, 1, 1, 1, 1)
            z_t = (1.0 - tv) * z_lq + tv * z_hq
            dove_active = False
        v_target = z_hq - z_lq

        if not args.no_vjepa:
            if multilayer_reducer is not None:
                z_vj = multilayer_reducer(feats_lq_sel)
            else:
                z_vj = z_vj_single
            L_clean_cos = torch.tensor(0.0, device=device)
            L_clean_l1 = torch.tensor(0.0, device=device)
            clean_demand_mean = torch.tensor(0.0, device=device)
            clean_gate = torch.tensor(0.0, device=device)
            if clean_token_adapter is not None:
                z_vj, clean_aux = clean_token_adapter(z_vj, return_aux=True)
                L_clean_cos, L_clean_l1 = clean_token_losses(z_vj, z_vj_gt)
                clean_demand_mean = clean_aux["demand"].mean()
                clean_gate = clean_aux["gate"].detach()
            text_emb = vj2text(z_vj)
        else:
            text_emb = torch.zeros(B, 2048, 4096, device=device)
            z_vj = None
            L_clean_cos = torch.tensor(0.0, device=device)
            L_clean_l1 = torch.tensor(0.0, device=device)
            clean_demand_mean = torch.tensor(0.0, device=device)
            clean_gate = torch.tensor(0.0, device=device)
        timestep = (t * 1000).long()

        # v12a/v13a (C1): compute V-JEPA AdaLN params and activate hooks
        if vjepa_adaln is not None and z_vj is not None:
            adaln_holder['params'] = vjepa_adaln(z_vj)
            adaln_holder['active'] = True
        try:
            # v11 (A2): add learned LQ residual to DiT input (zero-init at start)
            dit_in = z_t + lq_cond_module(z_lq) if lq_cond_module is not None else z_t
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_pred = dit(dit_in.to(torch.bfloat16),
                             timestep,
                             encoder_hidden_states=text_emb.to(torch.bfloat16),
                             return_dict=False)[0].float()
        finally:
            if adaln_holder is not None:
                adaln_holder['active'] = False

        L_flow = F.mse_loss(v_pred, v_target)

        # Decode pred for pixel/perceptual losses
        z_hq_pred = z_lq + v_pred
        pred_rgb = wan_decode(vae, z_hq_pred, mean, std).clamp(0, 1)
        T_out = pred_rgb.shape[2]
        T_drop = (gt.shape[2] - T_out) // 2
        gt_trim = gt[:, :, T_drop:T_drop + T_out]

        L_l1 = F.l1_loss(pred_rgb, gt_trim)
        L_chroma = chroma_loss_ycbcr(pred_rgb, gt_trim)
        L_temporal = temporal_loss(pred_rgb, gt_trim)
        L_lpips = torch.tensor(0.0, device=device)
        if (it + 1) % args.lpips_every == 0:
            L_lpips = lpips_video_sample(lpips_m, pred_rgb, gt_trim, n_frames=4)

        # REPA on first 8 blocks (mean over blocks)
        L_repa = torch.tensor(0.0, device=device)
        haste_active = (it < args.haste_iter) and (not args.no_vjepa)
        if haste_active:
            block_losses = []
            for bi in REPA_BLOCKS:
                if bi in _bottleneck:
                    h_mid = _bottleneck[bi].float()
                    h_proj = repa_proj(h_mid)
                    h_p = F.normalize(h_proj.mean(dim=1), dim=-1)
                    v_p = F.normalize(z_vj_gt.mean(dim=1), dim=-1)
                    block_losses.append(-(h_p * v_p).sum(dim=-1).mean())
            if block_losses:
                L_repa = torch.stack(block_losses).mean()

        # v12b (C2): MeanFlow consistency loss — sample second t', enforce velocity invariance
        L_mf = torch.tensor(0.0, device=device)
        if args.lambda_meanflow > 0:
            with torch.no_grad():
                t2 = (torch.sigmoid(torch.randn(B, device=device))
                      if args.use_logit_normal_t else torch.rand(B, device=device))
                tv2 = t2.view(B, 1, 1, 1, 1)
                z_t2 = (1.0 - tv2) * z_lq + tv2 * z_hq
                ts2 = (t2 * 1000).long()
            try:
                if vjepa_adaln is not None and z_vj is not None:
                    adaln_holder['active'] = True
                dit_in2 = z_t2 + lq_cond_module(z_lq) if lq_cond_module is not None else z_t2
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v_pred2 = dit(dit_in2.to(torch.bfloat16), ts2,
                                  encoder_hidden_states=text_emb.to(torch.bfloat16),
                                  return_dict=False)[0].float()
            finally:
                if adaln_holder is not None:
                    adaln_holder['active'] = False
            # MeanFlow: both v_pred and v_pred2 should equal z_hq - z_lq
            L_mf = F.mse_loss(v_pred, v_pred2.detach())

        # Ramp NEW losses 0->1 over first 1000 iter
        ramp = min(1.0, (it + 1) / max(1, args.ramp_iters))
        L = (L_flow
             + ramp * args.lambda_l1       * L_l1
             + ramp * args.lambda_chroma   * L_chroma
             + ramp * args.lambda_lpips    * L_lpips
             + ramp * args.lambda_repa     * L_repa
             + ramp * args.lambda_temporal * L_temporal
             + ramp * args.lambda_meanflow * L_mf
             + ramp * args.lambda_clean_token * L_clean_cos
             + ramp * args.lambda_clean_l1 * L_clean_l1)

        optim.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optim.step(); sched.step()

        # EMA every 10 iter
        if (it + 1) % 10 == 0:
            ema_dit.update(dit)
            if ema_vj is not None: ema_vj.update(vj2text)
            if ema_ml is not None: ema_ml.update(multilayer_reducer)
            if ema_lqc is not None: ema_lqc.update(lq_cond_module)
            if ema_aln is not None: ema_aln.update(vjepa_adaln)
            if ema_clean is not None: ema_clean.update(clean_token_adapter)

        if (it + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ips = (it + 1) / elapsed
            eta_h = (args.iters - it - 1) / ips / 3600
            print(f"  it {it+1:>6d}/{args.iters} src={source} dove={int(dove_active)} ramp={ramp:.2f} "
                  f"L={L.item():.4f} flow={L_flow.item():.4f} "
                  f"l1={L_l1.item():.4f} ch={L_chroma.item():.4f} "
                  f"lp={L_lpips.item():.4f} rp={L_repa.item():.4f} "
                  f"tp={L_temporal.item():.4f} "
                  f"ct={L_clean_cos.item():.4f}/{L_clean_l1.item():.4f} "
                  f"dm={clean_demand_mean.item():.3f} cg={clean_gate.item():.3f} "
                  f"haste={int(haste_active)} eta={eta_h:.1f}h")
            log.append({"iter": it+1, "L": L.item(), "L_flow": L_flow.item(),
                        "L_l1": L_l1.item(), "L_chroma": L_chroma.item(),
                        "L_lpips": L_lpips.item(), "L_repa": L_repa.item(),
                        "L_temporal": L_temporal.item(),
                        "L_clean_cos": L_clean_cos.item(),
                        "L_clean_l1": L_clean_l1.item(),
                        "clean_demand_mean": clean_demand_mean.item(),
                        "clean_gate": clean_gate.item()})

        # In-training eval every args.eval_every iter
        if (not args.skip_final_eval) and ((it + 1) % args.eval_every == 0 or (it + 1) == args.iters):
            cur_psnr = in_training_eval(it + 1, vae, dit, vj2text, backbone,
                                        mean, std, eval_subset, csv_writer, viz_root, args,
                                        multilayer_reducer=multilayer_reducer,
                                        lq_cond=lq_cond_module,
                                        vjepa_adaln=vjepa_adaln, adaln_holder=adaln_holder,
                                        clean_token_adapter=clean_token_adapter)
            eval_csv.flush()
            # Save intermediate ckpt
            ckpt_path = os.path.join(args.out_dir, f"ckpt_iter{it+1:06d}.pt")
            _save_ckpt(ckpt_path, dit, vj2text, repa_proj,
                       ema_dit, ema_vj, args, log,
                       multilayer_reducer=multilayer_reducer, ema_ml=ema_ml,
                       lq_cond=lq_cond_module, ema_lqc=ema_lqc,
                       vjepa_adaln=vjepa_adaln, ema_aln=ema_aln,
                       clean_token_adapter=clean_token_adapter, ema_clean=ema_clean)
            # v3.1: save best.pt if mean PSNR improved
            if cur_psnr > _best_psnr[0]:
                _best_psnr[0] = cur_psnr
                best_path = os.path.join(args.out_dir, "best.pt")
                _save_ckpt(best_path, dit, vj2text, repa_proj,
                           ema_dit, ema_vj, args, log,
                           multilayer_reducer=multilayer_reducer, ema_ml=ema_ml,
                           lq_cond=lq_cond_module, ema_lqc=ema_lqc,
                           vjepa_adaln=vjepa_adaln, ema_aln=ema_aln,
                           clean_token_adapter=clean_token_adapter, ema_clean=ema_clean)
                print(f"  [best.pt] new best PSNR={cur_psnr:.3f} saved")
            _prune_ckpts(args.out_dir, keep_last=2)

    eval_csv.close()
    # Final ckpt
    final_path = os.path.join(args.out_dir, "ckpt.pt")
    _save_ckpt(final_path, dit, vj2text, repa_proj,
               ema_dit, ema_vj, args, log,
               multilayer_reducer=multilayer_reducer, ema_ml=ema_ml,
               lq_cond=lq_cond_module, ema_lqc=ema_lqc,
               vjepa_adaln=vjepa_adaln, ema_aln=ema_aln,
               clean_token_adapter=clean_token_adapter, ema_clean=ema_clean)
    print(f"[ckpt] final saved -> {final_path}")
    if args.skip_final_eval:
        print("[skip_final_eval] final full eval skipped")
        return

    # Final full eval (11 seqs + viz)
    print("\n=== FINAL EVAL: 11 SDSD held-out seqs ===")
    dit.eval()
    if vj2text is not None: vj2text.eval()
    if vjepa_adaln is not None: vjepa_adaln.eval()
    if clean_token_adapter is not None: clean_token_adapter.eval()
    final_viz_dir = os.path.join(viz_root, "final")
    os.makedirs(final_viz_dir, exist_ok=True)
    psnrs = []; iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    with torch.no_grad():
        for name, lq_p, gt_p in sdsd_test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            pred = wanvj_inference(vae, dit, vj2text, backbone, lq, mean, std,
                                   ref_layer=args.ref_layer, dove_t=args.dove_t,
                                   apply_adain=False, n_steps=args.inference_steps,
                                   multilayer_reducer=multilayer_reducer,
                                   ml_indices=ml_indices, no_vjepa=args.no_vjepa,
                                   lq_cond=lq_cond_module,
                                   vjepa_adaln=vjepa_adaln, adaln_holder=adaln_holder,
                                   clean_token_adapter=clean_token_adapter)
            T_out = pred.shape[2]
            T_drop = (gt.shape[2] - T_out) // 2
            gt_trim = gt[:, :, T_drop:T_drop + T_out]
            lq_trim = lq[:, :, T_drop:T_drop + T_out]
            p = mean_psnr(to_u8(pred), to_u8(gt_trim))
            iqa = compute_iqa(pred, gt_trim, device=device)
            psnrs.append(p)
            for k in iqa_acc: iqa_acc[k].append(iqa[k])
            print(f"    [{name}] PSNR={p:.3f} {fmt_iqa(iqa)}")
            save_compare_video(os.path.join(final_viz_dir, f"{name}.mp4"),
                               lq_trim, pred, gt_trim, fps=10)

    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "WANVJ_PHASE2_V3_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# WanVJ Phase 2 (no ControlNet, LoRA + V-JEPA TruePAC)\n\n")
        f.write(f"Iters: {args.iters}; LR: {args.lr}; LoRA rank: {args.lora_rank}\n")
        f.write(f"DOVE t={args.dove_t}\n")
        f.write(f"Loss: l1={args.lambda_l1} chroma={args.lambda_chroma} "
                f"lpips={args.lambda_lpips} repa={args.lambda_repa} "
                f"temporal={args.lambda_temporal} "
                f"clean_token={args.lambda_clean_token} clean_l1={args.lambda_clean_l1}\n")
        if args.use_clean_token_adapter:
            f.write(f"v14 CleanTokenAdapter: hidden={args.clean_hidden} "
                    f"depth={args.clean_depth} init_gate={args.clean_init_gate}\n")
        f.write(f"V-JEPA: {'DISABLED' if args.no_vjepa else 'ENABLED'}\n\n")
        f.write(f"## SDSD held-out 11 seqs (1-NFE, NO AdaIN)\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}\n\n")
        f.write(f"### Per-seq\n\n")
        for name, _, _ in sdsd_test_seqs:
            i = [s[0] for s in sdsd_test_seqs].index(name)
            f.write(f"- {name}: PSNR={psnrs[i]:.3f}\n")
    print(f"\n[FINAL] PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")


def _save_ckpt(path, dit, vj2text, repa_proj,
               ema_dit, ema_vj, args, log,
               multilayer_reducer=None, ema_ml=None,
               lq_cond=None, ema_lqc=None,
               vjepa_adaln=None, ema_aln=None,
               clean_token_adapter=None, ema_clean=None):
    trainable_names = {n for n, p in dit.named_parameters() if p.requires_grad}
    sd = {
        "dit_lora": {k: v.cpu() for k, v in dit.state_dict().items()
                     if k in trainable_names},
        "vj2text": vj2text.state_dict() if vj2text is not None else None,
        "repa_proj": repa_proj.state_dict() if repa_proj is not None else None,
        "multilayer_reducer": (multilayer_reducer.state_dict()
                               if multilayer_reducer is not None else None),
        "lq_cond": lq_cond.state_dict() if lq_cond is not None else None,
        "vjepa_adaln": vjepa_adaln.state_dict() if vjepa_adaln is not None else None,
        "clean_token_adapter": (clean_token_adapter.state_dict()
                                if clean_token_adapter is not None else None),
        "ema_dit": ema_dit.shadow,
        "ema_vj": ema_vj.shadow if ema_vj is not None else None,
        "ema_ml": ema_ml.shadow if ema_ml is not None else None,
        "ema_lqc": ema_lqc.shadow if ema_lqc is not None else None,
        "ema_aln": ema_aln.shadow if ema_aln is not None else None,
        "ema_clean": ema_clean.shadow if ema_clean is not None else None,
        "args": vars(args), "log": log, "stage": "wanvj_phase2_v3",
    }
    torch.save(sd, path)


def _prune_ckpts(out_dir, keep_last=2):
    """Disk hygiene: keep only N latest intermediate ckpts."""
    ckpts = sorted([f for f in os.listdir(out_dir)
                    if f.startswith("ckpt_iter") and f.endswith(".pt")])
    for f in ckpts[:-keep_last]:
        try: os.remove(os.path.join(out_dir, f))
        except OSError: pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--viz_dir", default=None)
    ap.add_argument("--iters", type=int, default=25000)
    ap.add_argument("--ramp_iters", type=int, default=1000)
    ap.add_argument("--haste_iter", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--lpips_every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_final_eval", action="store_true",
                    help="Debug only: skip in-training eval and final 11-seq eval")
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--dove_t", type=int, default=399,
                    help="DOVE inference start timestep (0-1000)")
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_chroma", type=float, default=0.1)
    ap.add_argument("--lambda_lpips", type=float, default=0.1)
    ap.add_argument("--lambda_repa", type=float, default=0.05)
    ap.add_argument("--lambda_temporal", type=float, default=0.1)
    ap.add_argument("--no_vjepa", action="store_true",
                    help="Track B: zero V-JEPA encoder hidden states (ablation)")
    ap.add_argument("--use_pe", action="store_true",
                    help="Add learned PE to V-JEPA tokens (default OFF per shuffle ablation)")
    ap.add_argument("--freeze_norm", action="store_true",
                    help="v5: keep DiT LayerNorm γ/β frozen (Phase 1 style)")
    ap.add_argument("--use_dora", action="store_true",
                    help="v6: use DoRA (Liu et al. ICML 2024) instead of LoRA")
    ap.add_argument("--use_pac", action="store_true",
                    help="v8: use PACProjector (warm-start from V-JEPA predictor) instead of MLP")
    ap.add_argument("--use_truepac", action="store_true",
                    help="v9: use TruePACProjector (full V-JEPA predictor frozen, proper RoPE+mask)")
    ap.add_argument("--pac_n_refine", type=int, default=2,
                    help="PAC: number of refinement transformer blocks (warm-start from predictor_blocks[0..n-1])")
    ap.add_argument("--vjepa2_repo", type=str,
                    default="/cache/zihaohe/modelzoo/vjepa2-main",
                    help="Path to local V-JEPA 2 repo (for predictor module)")
    ap.add_argument("--vjepa2_ckpt", type=str,
                    default="/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt",
                    help="Path to V-JEPA 2.1 ckpt (provides predictor warm-start weights)")
    ap.add_argument("--use_qformer", action="store_true",
                    help="v7: replace MLP projector with Q-Former (128 queries, 2 layers)")
    ap.add_argument("--qformer_n_queries", type=int, default=128,
                    help="Q-Former: number of learnable query tokens")
    ap.add_argument("--qformer_n_layers", type=int, default=2,
                    help="Q-Former: number of cross+self attn layers")
    ap.add_argument("--dove_prob", type=float, default=0.0,
                    help="DOVE training mixin disabled in v3.1 (was buggy in v3)")
    # Wan-Full baseline
    ap.add_argument("--no_lora", action="store_true",
                    help="Wan-Full baseline: skip LoRA injection, full DiT fine-tune")
    # v10: multi-layer V-JEPA via channel-concat reducer
    ap.add_argument("--use_multilayer_vjepa", action="store_true",
                    help="v10: fuse multiple V-JEPA layers via channel-concat reducer")
    ap.add_argument("--ml_indices", type=str, default="0,2,3",
                    help="v10: comma-separated indices into backbone feature_layers "
                         "(default 0,2,3 = ViT L5,L17,L23 from default [5,11,17,23])")
    # Inference: 1-NFE (1) or N-step Euler ODE
    ap.add_argument("--inference_steps", type=int, default=1,
                    help="v10: 4 for 4-step Euler ODE; 1 (default) keeps 1-NFE")
    # v11 (A1): SD3-style logit-normal t-sampling
    ap.add_argument("--use_logit_normal_t", action="store_true",
                    help="v11-A1: t = sigmoid(randn) instead of U(0,1) — SD3/FLUX style mid-noise weighting")
    # v11 (A2): active LQ conditioner
    ap.add_argument("--use_lq_cond", action="store_true",
                    help="v11-A2: add learned LQ residual via 2-block ResNet3D, zero-init (v9-equivalent at start)")
    ap.add_argument("--lq_cond_hidden", type=int, default=64,
                    help="v11-A2: lq_cond hidden channel dim")
    # v12a (C1): V-JEPA AdaLN-Zero modulation
    ap.add_argument("--use_vjepa_adaln", action="store_true",
                    help="v12a-C1: V-JEPA pooled → per-block AdaLN-Zero modulation; init γ=β=0 → v9-equivalent")
    ap.add_argument("--use_spatial_adaln", action="store_true",
                    help="v13a-C1+: V-JEPA per-token spatial-aware AdaLN-Zero (overrides --use_vjepa_adaln)")
    ap.add_argument("--adaln_hidden", type=int, default=256,
                    help="v12a/v13a-C1: AdaLN trunk hidden dim")
    # v14: counterfactual clean-token prior
    ap.add_argument("--use_clean_token_adapter", action="store_true",
                    help="v14: predict clean/HQ V-JEPA tokens from LQ tokens before TruePAC conditioning")
    ap.add_argument("--clean_hidden", type=int, default=256,
                    help="v14: hidden dim for CleanTokenAdapter")
    ap.add_argument("--clean_depth", type=int, default=2,
                    help="v14: number of spatiotemporal token-mixing blocks")
    ap.add_argument("--clean_init_gate", type=float, default=-2.0,
                    help="v14: initial gate logit; -2 gives a conservative ~0.12 residual scale")
    ap.add_argument("--lambda_clean_token", type=float, default=0.1,
                    help="v14: weight for cosine z_clean_hat -> z_HQ token loss")
    ap.add_argument("--lambda_clean_l1", type=float, default=0.01,
                    help="v14: weak SmoothL1 stabilizer for z_clean_hat -> z_HQ token loss")
    # v12b (C2): MeanFlow consistency loss
    ap.add_argument("--lambda_meanflow", type=float, default=0.0,
                    help="v12b-C2: MeanFlow consistency loss weight; 0 disables (default)")
    main(ap.parse_args())
