"""Dump per-frame patch-feature L2 norm maps for V-JEPA vs SigLIP 2 on a
moving BVI sequence (S37_Mario3__10fps). Also compute frame-to-frame
consistency: mean pairwise cosine similarity of adjacent-frame norm-map
vectors.

- 8 evenly-spaced frames from the 370-frame LQ sequence.
- V-JEPA at 512 px, image encoder set to 16 frames (each of the 8 sample
  frames repeated with its immediate next frame -> V-JEPA sees a T=16 clip
  built from 8 index positions, tubelet=2 => 8 output time-groups aligned
  with the 8 sample frames).
- SigLIP 2 @ 256 px, per-frame independent forward pass (image encoder).

Output: /tmp/temporal_feature_norms.npz
  frame_thumbs      (8, 512, 512, 3) uint8
  vjepa_norm_frames (8, 32, 32)      float32
  siglip_norm_frames(8, 16, 16)      float32
  dino3_norm_frames (8, 16, 16)      float32
  seq_name          "S37_Mario3__10fps"
"""
from __future__ import annotations
import os, sys, json, tarfile
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/cache/zihaohe/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/zihaohe/hf_cache")

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/vjepa_llve")
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa

DEVICE       = torch.device("cuda")
BVI_ROOT     = "/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos"
VJEPA_REPO   = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT   = "/cache/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"
SIGLIP2_ID   = "google/siglip2-large-patch16-256"
DINO3_ID     = "vit_large_patch16_dinov3.lvd1689m"
CACHE_DIR    = "/cache/zihaohe/hf_cache"

SEQ_NAME = "S37_Mario3__10fps"
LQ_DIR   = os.path.join(BVI_ROOT, "S37_Mario3", "low_light_10")
CROP     = 512
N_SAMPLE = 8
CLIP_LEN = 16
TUBELET  = 2

SIGLIP_MEAN = torch.tensor([0.5, 0.5, 0.5])
SIGLIP_STD  = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    H, W = arr.shape[:2]
    y0 = (H - size) // 2; x0 = (W - size) // 2
    return arr[y0:y0+size, x0:x0+size]


def load_sampled_frames() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return
      - thumbs_u8 : (N_SAMPLE, CROP, CROP, 3) uint8, the 8 sample frames
      - clip_u8   : (CLIP_LEN=16, CROP, CROP, 3) uint8, V-JEPA input clip
                    built as 8 pairs (sample_i, sample_i+step_frac) so
                    tubelet=2 aligns 8 output time-groups to our 8 samples.
      - indices   : which frame indices (in the raw LQ dir) were used
    """
    files = sorted(f for f in os.listdir(LQ_DIR) if f.endswith(".png"))
    T = len(files)
    # 8 evenly spaced frames across the whole sequence.
    idx = np.linspace(0, T - 1, N_SAMPLE, dtype=int).tolist()

    def _load(i):
        return center_crop(imageio.imread(os.path.join(LQ_DIR, files[i])), CROP)

    thumbs = np.stack([_load(i) for i in idx], axis=0)   # (8, 512, 512, 3)

    # Build a 16-frame V-JEPA clip: for each sample i, take (i, i+delta) where
    # delta is a small offset to a nearby frame; tubelet=2 will collapse each
    # pair into 1 time-group aligned with our sample. This means the 8
    # V-JEPA output time-groups correspond exactly to our 8 sample frames.
    # Use next frame if available, else prev.
    pair_indices = []
    for i in idx:
        j = i + 1 if i + 1 < T else i - 1
        pair_indices.extend([i, j])                       # length 16
    clip_frames = np.stack([_load(k) for k in pair_indices], axis=0)  # (16, ...)
    return thumbs, clip_frames, idx


def build_vjepa():
    return VJepaBackboneLocal21(
        local_repo=VJEPA_REPO,
        local_ckpt=VJEPA_CKPT,
        feature_layers=(23,),
        image_size=CROP,
        frames_per_clip=CLIP_LEN,
        patch_size=16,
        tubelet_size=TUBELET,
        freeze=True,
    ).to(DEVICE).eval()


@torch.no_grad()
def vjepa_per_frame_norm(model, clip_u8: np.ndarray) -> np.ndarray:
    """clip_u8: (16, 512, 512, 3) uint8. Returns (8, Hp, Wp) L2 norm per patch
    for each of the 8 tubelet-collapsed time-groups."""
    x = torch.from_numpy(clip_u8).to(DEVICE).float().div_(255.0)
    # (T, H, W, 3) -> (1, 3, T, H, W)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    out = model(x)
    feats = out["feats"][0]                          # (1, N, D)
    Tp, Hp, Wp = out["token_grid"]                   # (8, 32, 32)
    assert feats.shape[1] == Tp * Hp * Wp
    D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D).squeeze(0)      # (Tp, Hp, Wp, D)
    norm = torch.linalg.norm(f, dim=-1)              # (Tp, Hp, Wp)
    return norm.detach().float().cpu().numpy()


def _resize_norm_2d(x_bchw: torch.Tensor, size: int, mean: torch.Tensor,
                    std: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x_bchw, size=(size, size), mode="bilinear",
                      align_corners=False)
    m = mean.to(x.device).view(1, 3, 1, 1)
    s = std.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def siglip_per_frame_norm(model, thumbs_u8: np.ndarray) -> np.ndarray:
    """thumbs_u8: (8, 512, 512, 3) uint8. Independent per-frame forward at
    SigLIP native 256. Returns (8, 16, 16) L2 norm per patch."""
    x = torch.from_numpy(thumbs_u8).to(DEVICE).float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()               # (8, 3, 512, 512)
    x = _resize_norm_2d(x, 256, SIGLIP_MEAN, SIGLIP_STD)  # (8, 3, 256, 256)
    out = model(pixel_values=x, output_hidden_states=False)
    tokens = out.last_hidden_state                        # (8, 256, D) no CLS
    B, N, D = tokens.shape
    grid = int(round(N ** 0.5)); assert grid * grid == N
    tokens = tokens.view(B, grid, grid, D)                # (8, 16, 16, D)
    norm = torch.linalg.norm(tokens, dim=-1)              # (8, 16, 16)
    return norm.detach().float().cpu().numpy()


@torch.no_grad()
def dino3_per_frame_norm(model, thumbs_u8: np.ndarray, n_prefix: int) -> np.ndarray:
    """thumbs_u8: (8, 512, 512, 3) uint8. Independent per-frame forward at
    DINOv3 native 256. Drops CLS + register tokens (n_prefix total).
    Returns (8, 16, 16) L2 norm per patch."""
    x = torch.from_numpy(thumbs_u8).to(DEVICE).float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()               # (8, 3, 512, 512)
    x = _resize_norm_2d(x, 256, IMAGENET_MEAN, IMAGENET_STD)  # (8, 3, 256, 256)
    tokens = model.forward_features(x)                    # (8, 5+256, D)
    tokens = tokens[:, n_prefix:, :]                      # (8, 256, D)
    B, N, D = tokens.shape
    grid = int(round(N ** 0.5)); assert grid * grid == N, f"unexpected {N}"
    tokens = tokens.view(B, grid, grid, D)                # (8, 16, 16, D)
    norm = torch.linalg.norm(tokens, dim=-1)              # (8, 16, 16)
    return norm.detach().float().cpu().numpy()


def adjacent_cos_mean(norm_frames: np.ndarray) -> float:
    """norm_frames: (N, h, w) -> mean cosine similarity between adjacent
    frame norm-maps flattened as vectors."""
    N = norm_frames.shape[0]
    vecs = norm_frames.reshape(N, -1).astype(np.float64)
    sims = []
    for i in range(N - 1):
        a, b = vecs[i], vecs[i + 1]
        num = float((a * b).sum())
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        sims.append(num / den)
    return float(np.mean(sims))


def main():
    print(f"[*] seq = {SEQ_NAME}   dir = {LQ_DIR}", flush=True)
    thumbs, clip16, idx = load_sampled_frames()
    print(f"[*] sample indices = {idx}   thumbs {thumbs.shape}   clip16 {clip16.shape}",
          flush=True)

    print("[*] loading V-JEPA ...", flush=True)
    vjepa = build_vjepa()
    v_norm = vjepa_per_frame_norm(vjepa, clip16)          # (8, 32, 32)
    print(f"    v_norm {v_norm.shape}  mean={v_norm.mean():.3f}", flush=True)
    del vjepa; torch.cuda.empty_cache()

    print("[*] loading SigLIP 2 ...", flush=True)
    from transformers import SiglipVisionModel
    siglip = SiglipVisionModel.from_pretrained(SIGLIP2_ID,
                                               cache_dir=CACHE_DIR).to(DEVICE).eval()
    s_norm = siglip_per_frame_norm(siglip, thumbs)        # (8, 16, 16)
    print(f"    s_norm {s_norm.shape}  mean={s_norm.mean():.3f}", flush=True)
    del siglip; torch.cuda.empty_cache()

    print("[*] loading DINOv3 ViT-L/16 (timm) ...", flush=True)
    import timm
    dino3 = timm.create_model(DINO3_ID, pretrained=True, num_classes=0).to(DEVICE).eval()
    n_prefix = int(getattr(dino3, "num_prefix_tokens", 5))
    n_reg = int(dino3.reg_token.shape[1]) if hasattr(dino3, "reg_token") else 0
    print(f"    dino3 num_prefix_tokens = {n_prefix}  (1 CLS + {n_reg} reg)", flush=True)
    d_norm = dino3_per_frame_norm(dino3, thumbs, n_prefix=n_prefix)   # (8, 16, 16)
    print(f"    d_norm {d_norm.shape}  mean={d_norm.mean():.3f}", flush=True)
    del dino3; torch.cuda.empty_cache()

    v_adj = adjacent_cos_mean(v_norm)
    s_adj = adjacent_cos_mean(s_norm)
    d_adj = adjacent_cos_mean(d_norm)
    print(f"\nvjepa_adj_cos_mean  = {v_adj:.6f}")
    print(f"siglip_adj_cos_mean = {s_adj:.6f}")
    print(f"dino3_adj_cos_mean  = {d_adj:.6f}")
    # expected: V-JEPA > DINOv3 > SigLIP
    order_ok = (v_adj > d_adj) and (d_adj > s_adj)
    if order_ok:
        print("[OK] ordering V-JEPA > DINOv3 > SigLIP holds")
    else:
        print("[SURPRISE] expected V-JEPA > DINOv3 > SigLIP but got "
              f"V={v_adj:.4f}, D={d_adj:.4f}, S={s_adj:.4f}")
        if d_adj > v_adj:
            print("[SURPRISE] DINOv3 > V-JEPA on adjacent-frame cosine "
                  "(NOT altering pooling per spec)")

    out_path = "/tmp/temporal_feature_norms.npz"
    np.savez(
        out_path,
        frame_thumbs=thumbs.astype(np.uint8),
        vjepa_norm_frames=v_norm.astype(np.float32),
        siglip_norm_frames=s_norm.astype(np.float32),
        dino3_norm_frames=d_norm.astype(np.float32),
        seq_name=np.array(SEQ_NAME),
        sample_indices=np.array(idx, dtype=np.int32),
        vjepa_adj_cos_mean=np.float32(v_adj),
        siglip_adj_cos_mean=np.float32(s_adj),
        dino3_adj_cos_mean=np.float32(d_adj),
    )
    print(f"\n[SAVE] {out_path}", flush=True)

    # summary json (create/overwrite alongside npz)
    summary_path = "/tmp/temporal_feature_norms_summary.json"
    summary = {
        "seq_name": SEQ_NAME,
        "sample_indices": idx,
        "adj_cos_mean": {
            "vjepa": float(v_adj),
            "dino3": float(d_adj),
            "siglip": float(s_adj),
        },
        "expected_ordering": "V-JEPA > DINOv3 > SigLIP",
        "ordering_holds": bool(order_ok),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] {summary_path}", flush=True)

    # tar for download
    tar_path = "/tmp/temporal_feature_norms.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(out_path, arcname=os.path.basename(out_path))
        tf.add(summary_path, arcname=os.path.basename(summary_path))
    print(f"[SAVE] {tar_path}", flush=True)


if __name__ == "__main__":
    main()
