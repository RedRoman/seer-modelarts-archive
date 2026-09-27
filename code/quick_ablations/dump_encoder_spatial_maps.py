"""Dump SPATIAL (patch-token) feature maps from V-JEPA / CLIP / DINOv2 on ONE
SDSD-indoor pair (LQ and HQ), and compute per-patch normalised L2 shift and
cosine-similarity maps between LQ and HQ.

Output: /tmp/encoder_spatial_maps.npz
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/cache/zihaohe/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/zihaohe/hf_cache")

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/vjepa_llve")
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa

DEVICE = torch.device("cuda")

SDSD_ROOT   = "/cache/zihaohe/datasets/SDSD_indoor/indoor_static_np"
VJEPA_REPO  = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT  = "/cache/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"
CLIP_ID     = "openai/clip-vit-large-patch14"
DINO_ID     = "facebook/dinov2-large"
CACHE_DIR   = "/cache/zihaohe/hf_cache"

SEQ = "pair52"
CLIP_LEN = 16
CROP     = 512


def load_pair_frames(seq: str, side: str) -> np.ndarray:
    d = os.path.join(SDSD_ROOT, side, seq)
    fns = sorted(os.listdir(d))
    T = len(fns)
    assert T >= CLIP_LEN, f"{seq}/{side}: only {T} frames"
    start = (T - CLIP_LEN) // 2
    fns = fns[start:start + CLIP_LEN]
    arrs = []
    for fn in fns:
        a = np.load(os.path.join(d, fn))
        H, W, _ = a.shape
        y0 = (H - CROP) // 2
        x0 = (W - CROP) // 2
        arrs.append(a[y0:y0+CROP, x0:x0+CROP, :])
    return np.stack(arrs, axis=0)  # (T, 512, 512, 3) uint8


# --- V-JEPA ---
def build_vjepa():
    return VJepaBackboneLocal21(
        local_repo=VJEPA_REPO,
        local_ckpt=VJEPA_CKPT,
        feature_layers=(23,),
        image_size=CROP,
        frames_per_clip=CLIP_LEN,
        patch_size=16,
        tubelet_size=2,
        freeze=True,
    ).to(DEVICE).eval()


@torch.no_grad()
def vjepa_patch_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    """Return (H', W', D) mean-over-time patch feature tensor.
    Patch grid is 32x32 for 512/16."""
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1,3,T,H,W)
    out = model(x)
    feats = out["feats"][0]                              # (1, N, 1024)
    Tp, Hp, Wp = out["token_grid"]                       # (8, 32, 32) for 512/16
    N = Tp * Hp * Wp
    assert feats.shape[1] == N, f"N mismatch: got {feats.shape[1]} expected {N}"
    D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D)                     # (1,T',H',W',D)
    f = f.mean(dim=1).squeeze(0)                         # (H', W', D)
    return f.detach().float().cpu().numpy()


# --- CLIP / DINOv2 shared preprocessing ---
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN     = torch.tensor([0.48145466, 0.4578275,  0.40821073])
CLIP_STD      = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def _to_bchw01(frames_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


def _resize_norm(x_bchw: torch.Tensor, size: int, mean: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x_bchw, size=(size, size), mode="bilinear",
                      align_corners=False)
    m = mean.to(x.device).view(1, 3, 1, 1)
    s = std.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def clip_patch_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    """CLIP ViT-L/14 @ 224 -> 16x16 patch grid. Return (16, 16, D) mean-over-T."""
    x = _to_bchw01(frames_uint8)
    x = _resize_norm(x, 224, CLIP_MEAN, CLIP_STD)
    out = model(pixel_values=x, output_hidden_states=False)
    tokens = out.last_hidden_state[:, 1:, :]             # (T, 256, 1024)
    T = tokens.shape[0]
    D = tokens.shape[-1]
    grid = int(round(tokens.shape[1] ** 0.5))            # 16
    assert grid * grid == tokens.shape[1], f"unexpected patch count {tokens.shape[1]}"
    tokens = tokens.view(T, grid, grid, D)
    feats = tokens.mean(dim=0)                           # (grid, grid, D)
    return feats.detach().float().cpu().numpy()


@torch.no_grad()
def dino_patch_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    """DINOv2 ViT-L/14 @ 224 -> 16x16 patch grid (drop CLS)."""
    x = _to_bchw01(frames_uint8)
    x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    out = model(pixel_values=x, output_hidden_states=False)
    tokens = out.last_hidden_state[:, 1:, :]             # (T, 256, 1024)
    T, N, D = tokens.shape
    grid = int(round(N ** 0.5))
    assert grid * grid == N, f"unexpected patch count {N}"
    tokens = tokens.view(T, grid, grid, D)
    feats = tokens.mean(dim=0)                           # (grid, grid, D)
    return feats.detach().float().cpu().numpy()


def per_patch_shift_and_cos(lq_feat: np.ndarray, hq_feat: np.ndarray):
    """lq_feat, hq_feat: (h, w, D) float32.
    Returns (shift_map, cos_map) each (h, w) float32.
      shift_map[i,j] = ||lq - hq||_2 / (||hq||_2 + 1e-6)
      cos_map[i,j]   = <lq,hq> / (||lq|| * ||hq|| + 1e-6)
    """
    lq = lq_feat.astype(np.float64)
    hq = hq_feat.astype(np.float64)
    diff = lq - hq
    diff_norm = np.linalg.norm(diff, axis=-1)             # (h, w)
    hq_norm   = np.linalg.norm(hq,   axis=-1)             # (h, w)
    lq_norm   = np.linalg.norm(lq,   axis=-1)             # (h, w)
    shift = diff_norm / (hq_norm + 1e-6)
    dot   = (lq * hq).sum(axis=-1)
    cos   = dot / (lq_norm * hq_norm + 1e-6)
    return shift.astype(np.float32), cos.astype(np.float32)


def main():
    from transformers import CLIPVisionModel, Dinov2Model

    print(f"[*] SEQ = {SEQ}", flush=True)
    print("[*] loading frames ...", flush=True)
    lq = load_pair_frames(SEQ, "input")   # (16, 512, 512, 3) uint8
    hq = load_pair_frames(SEQ, "GT")
    print(f"    LQ {lq.shape}  HQ {hq.shape}", flush=True)

    center = CLIP_LEN // 2
    lq_thumb = lq[center].copy()
    hq_thumb = hq[center].copy()

    print("[*] loading V-JEPA ...", flush=True)
    vjepa = build_vjepa()
    v_lq_feat = vjepa_patch_feats(vjepa, lq)   # (32, 32, D)
    v_hq_feat = vjepa_patch_feats(vjepa, hq)
    del vjepa; torch.cuda.empty_cache()

    print("[*] loading CLIP ...", flush=True)
    clip = CLIPVisionModel.from_pretrained(CLIP_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    c_lq_feat = clip_patch_feats(clip, lq)
    c_hq_feat = clip_patch_feats(clip, hq)
    del clip; torch.cuda.empty_cache()

    print("[*] loading DINOv2 ...", flush=True)
    dino = Dinov2Model.from_pretrained(DINO_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    d_lq_feat = dino_patch_feats(dino, lq)
    d_hq_feat = dino_patch_feats(dino, hq)
    del dino; torch.cuda.empty_cache()

    # per-patch L2 shift and cosine
    v_shift, v_cos = per_patch_shift_and_cos(v_lq_feat, v_hq_feat)
    c_shift, c_cos = per_patch_shift_and_cos(c_lq_feat, c_hq_feat)
    d_shift, d_cos = per_patch_shift_and_cos(d_lq_feat, d_hq_feat)

    # keep old fields (channel-mean maps) for backward-compat, ignored by local script
    def _mean_channel_map(feat_hw_d):
        return feat_hw_d.mean(axis=-1).astype(np.float32)

    v_lq_map = _mean_channel_map(v_lq_feat)
    v_hq_map = _mean_channel_map(v_hq_feat)
    c_lq_map = _mean_channel_map(c_lq_feat)
    c_hq_map = _mean_channel_map(c_hq_feat)
    d_lq_map = _mean_channel_map(d_lq_feat)
    d_hq_map = _mean_channel_map(d_hq_feat)

    np.savez(
        "/tmp/encoder_spatial_maps.npz",
        seq_name=np.array(SEQ),
        # new per-patch shift maps
        vjepa_shift=v_shift, clip_shift=c_shift, dino_shift=d_shift,
        # new per-patch cosine maps
        vjepa_cos=v_cos, clip_cos=c_cos, dino_cos=d_cos,
        # legacy channel-mean maps (kept for backward-compat)
        vjepa_lq_map=v_lq_map, vjepa_hq_map=v_hq_map,
        clip_lq_map=c_lq_map,  clip_hq_map=c_hq_map,
        dino_lq_map=d_lq_map,  dino_hq_map=d_hq_map,
        lq_thumb=lq_thumb.astype(np.uint8),
        hq_thumb=hq_thumb.astype(np.uint8),
    )
    print("[*] saved /tmp/encoder_spatial_maps.npz", flush=True)

    # --- sanity print ---
    print("\n=== per-patch shift-map means (should approx paper's rho_L2) ===")
    print(f"    V-JEPA  shift.mean = {v_shift.mean():.4f}   (target 0.267)")
    print(f"    CLIP    shift.mean = {c_shift.mean():.4f}   (target 0.507)")
    print(f"    DINOv2  shift.mean = {d_shift.mean():.4f}   (target 0.547)")

    print("\n=== per-patch cosine-map means (should approx paper's rho_cos) ===")
    print(f"    V-JEPA  cos.mean   = {v_cos.mean():.4f}   (target 0.965)")
    print(f"    CLIP    cos.mean   = {c_cos.mean():.4f}   (target 0.868)")
    print(f"    DINOv2  cos.mean   = {d_cos.mean():.4f}   (target 0.831)")

    print("\n=== map shapes / ranges ===")
    for name, m in [("vjepa_shift", v_shift), ("clip_shift", c_shift),
                    ("dino_shift", d_shift), ("vjepa_cos", v_cos),
                    ("clip_cos", c_cos), ("dino_cos", d_cos)]:
        print(f"    {name:12s}  shape={m.shape}  min={m.min():+.4f}  max={m.max():+.4f}  mean={m.mean():+.4f}")


if __name__ == "__main__":
    main()
