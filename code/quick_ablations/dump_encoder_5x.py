"""Dump per-patch feature maps (LQ and HQ separately + relative L2 shift)
for FIVE encoders across the 15 SDSD-indoor pairs.

Encoders:
  clip   -> CLIP ViT-L/14         (openai/clip-vit-large-patch14),  224px, 16x16
  dino   -> DINOv2 ViT-L/14       (facebook/dinov2-large),          224px, 16x16
  dino3  -> DINOv3 ViT-L/16       (timm vit_large_patch16_dinov3.lvd1689m,
                                    hosted as public timm weights of the
                                    facebook DINOv3 LVD-1689M ViT-L/16),  256px, 16x16
  siglip -> SigLIP 2 ViT-L/16     (google/siglip2-large-patch16-256), 256px, 16x16
  vjepa  -> V-JEPA 2.1 ViT-L      (local ckpt), 512px, 32x32 (T-mean of block 23)

For each encoder and each of (LQ, HQ) we grab last-block patch tokens, reshape
to (h, w, D), and compute:
  shift_map[i,j]   = ||feat_LQ[i,j] - feat_HQ[i,j]||_2 / (||feat_HQ[i,j]||_2 + 1e-6)
  lq_norm_map[i,j] = ||feat_LQ[i,j]||_2
  hq_norm_map[i,j] = ||feat_HQ[i,j]||_2

We average across time (16 frames), giving per-encoder (h, w) maps.

Output:
  /tmp/encoder_features_5x.npz
  /tmp/encoder_features_5x_summary.json
  /tmp/encoder_features_5x.tar.gz
"""
from __future__ import annotations
import os, sys, json, tarfile, time
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
DINO3_ID    = "vit_large_patch16_dinov3.lvd1689m"          # timm.create_model id
SIGLIP2_ID  = "google/siglip2-large-patch16-256"
CACHE_DIR   = "/cache/zihaohe/hf_cache"

SEQ_NAMES = [
    "pair50","pair52","pair58","pair5_2","pair6","pair60","pair68","pair6_2",
    "pair7","pair70","pair7_2","pair8","pair8_2","pair9","pair9_2",
]

CLIP_LEN = 16
CROP     = 512


# ================================================================= data
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
    return np.stack(arrs, axis=0)   # (T, 512, 512, 3) uint8


# ================================================================= preproc
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN     = torch.tensor([0.48145466, 0.4578275,  0.40821073])
CLIP_STD      = torch.tensor([0.26862954, 0.26130258, 0.27577711])
# SigLIP 2 uses [0.5, 0.5, 0.5] mean and std
SIGLIP_MEAN   = torch.tensor([0.5, 0.5, 0.5])
SIGLIP_STD    = torch.tensor([0.5, 0.5, 0.5])


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


# ================================================================= V-JEPA
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
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()   # (1,3,T,H,W)
    out = model(x)
    feats = out["feats"][0]                               # (1, N, 1024)
    Tp, Hp, Wp = out["token_grid"]
    D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D).mean(dim=1).squeeze(0)   # (Hp, Wp, D)
    return f.detach().float().cpu().numpy()


# ================================================================= CLIP
@torch.no_grad()
def clip_patch_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames_uint8)
    x = _resize_norm(x, 224, CLIP_MEAN, CLIP_STD)
    out = model(pixel_values=x, output_hidden_states=False)
    tokens = out.last_hidden_state[:, 1:, :]              # drop CLS -> (T,256,1024)
    T, N, D = tokens.shape
    grid = int(round(N ** 0.5))
    assert grid * grid == N
    feats = tokens.view(T, grid, grid, D).mean(dim=0)     # (grid,grid,D)
    return feats.detach().float().cpu().numpy()


# ================================================================= DINOv2
@torch.no_grad()
def dino_patch_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames_uint8)
    x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    out = model(pixel_values=x, output_hidden_states=False)
    tokens = out.last_hidden_state[:, 1:, :]              # drop CLS
    T, N, D = tokens.shape
    grid = int(round(N ** 0.5))
    assert grid * grid == N
    feats = tokens.view(T, grid, grid, D).mean(dim=0)
    return feats.detach().float().cpu().numpy()


# ================================================================= DINOv3
@torch.no_grad()
def dino3_patch_feats(model, frames_uint8: np.ndarray, n_prefix: int) -> np.ndarray:
    """DINOv3 ViT-L/16 via timm at 256x256 -> 16x16 patch grid.
    Model uses 1 CLS + 4 register tokens (5 prefix tokens total)."""
    x = _to_bchw01(frames_uint8)
    x = _resize_norm(x, 256, IMAGENET_MEAN, IMAGENET_STD)
    # timm forward_features returns (B, 1+num_reg+N, D)
    tokens = model.forward_features(x)                    # (T, 5+N, D)
    tokens = tokens[:, n_prefix:, :]
    T, N, D = tokens.shape
    grid = int(round(N ** 0.5))
    assert grid * grid == N, f"unexpected patch count {N} for DINOv3"
    feats = tokens.view(T, grid, grid, D).mean(dim=0)
    return feats.detach().float().cpu().numpy()


# ================================================================= SigLIP2
@torch.no_grad()
def siglip_patch_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    """SigLIP 2 ViT-L @ 256 -> 16x16 patch grid. No CLS token."""
    x = _to_bchw01(frames_uint8)
    x = _resize_norm(x, 256, SIGLIP_MEAN, SIGLIP_STD)
    out = model(pixel_values=x, output_hidden_states=False)
    tokens = out.last_hidden_state                        # (T, 256, 1024) - no CLS
    T, N, D = tokens.shape
    grid = int(round(N ** 0.5))
    assert grid * grid == N, f"unexpected patch count {N} for SigLIP2"
    feats = tokens.view(T, grid, grid, D).mean(dim=0)
    return feats.detach().float().cpu().numpy()


# ================================================================= maps
def compute_maps(lq_feat: np.ndarray, hq_feat: np.ndarray):
    lq = lq_feat.astype(np.float64)
    hq = hq_feat.astype(np.float64)
    diff = lq - hq
    diff_norm = np.linalg.norm(diff, axis=-1)
    hq_norm   = np.linalg.norm(hq,   axis=-1)
    lq_norm   = np.linalg.norm(lq,   axis=-1)
    shift = diff_norm / (hq_norm + 1e-6)
    return (
        shift.astype(np.float32),
        lq_norm.astype(np.float32),
        hq_norm.astype(np.float32),
    )


# ================================================================= main
def main():
    from transformers import CLIPVisionModel, Dinov2Model, SiglipVisionModel
    import timm

    t0 = time.time()

    # Determine grid sizes ahead of time
    grids = {"clip": 16, "dino": 16, "dino3": 16, "siglip": 16, "vjepa": 32}

    out_arrays = {"seq_names": np.array(SEQ_NAMES)}
    for enc, g in grids.items():
        out_arrays[f"{enc}_shift"]   = np.zeros((15, g, g), dtype=np.float32)
        out_arrays[f"{enc}_lq_norm"] = np.zeros((15, g, g), dtype=np.float32)
        out_arrays[f"{enc}_hq_norm"] = np.zeros((15, g, g), dtype=np.float32)
    out_arrays["lq_thumbs"] = np.zeros((15, CROP, CROP, 3), dtype=np.uint8)
    out_arrays["hq_thumbs"] = np.zeros((15, CROP, CROP, 3), dtype=np.uint8)

    # Preload all frames once so we can iterate one encoder at a time and free VRAM
    print("[*] loading frames for all 15 pairs ...", flush=True)
    lq_frames = {}
    hq_frames = {}
    for i, seq in enumerate(SEQ_NAMES):
        lq_frames[seq] = load_pair_frames(seq, "input")
        hq_frames[seq] = load_pair_frames(seq, "GT")
        out_arrays["lq_thumbs"][i] = lq_frames[seq][CLIP_LEN // 2]
        out_arrays["hq_thumbs"][i] = hq_frames[seq][CLIP_LEN // 2]
    print(f"    loaded 15 pairs in {time.time()-t0:.1f}s", flush=True)

    def run_encoder(enc_key, feat_fn, model):
        print(f"[*] encoder={enc_key}", flush=True)
        for i, seq in enumerate(SEQ_NAMES):
            lq_feat = feat_fn(model, lq_frames[seq])
            hq_feat = feat_fn(model, hq_frames[seq])
            s, ln, hn = compute_maps(lq_feat, hq_feat)
            out_arrays[f"{enc_key}_shift"][i]   = s
            out_arrays[f"{enc_key}_lq_norm"][i] = ln
            out_arrays[f"{enc_key}_hq_norm"][i] = hn
        del model
        torch.cuda.empty_cache()

    # ---- V-JEPA ----
    print("[*] loading V-JEPA ...", flush=True)
    vjepa = build_vjepa()
    run_encoder("vjepa", vjepa_patch_feats, vjepa)

    # ---- CLIP ----
    print("[*] loading CLIP ...", flush=True)
    clip = CLIPVisionModel.from_pretrained(CLIP_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    run_encoder("clip", clip_patch_feats, clip)

    # ---- DINOv2 ----
    print("[*] loading DINOv2 ...", flush=True)
    dino = Dinov2Model.from_pretrained(DINO_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    run_encoder("dino", dino_patch_feats, dino)

    # ---- DINOv3 (timm) ----
    print("[*] loading DINOv3 ViT-L/16 (timm) ...", flush=True)
    dino3 = timm.create_model(DINO3_ID, pretrained=True, num_classes=0).to(DEVICE).eval()
    # figure out prefix (CLS + register tokens)
    n_prefix = int(getattr(dino3, "num_prefix_tokens", 5))
    n_reg = int(dino3.reg_token.shape[1]) if hasattr(dino3, "reg_token") else 0
    print(f"    dino3 num_prefix_tokens = {n_prefix}  (1 CLS + {n_reg} reg)", flush=True)
    # bind n_prefix
    def dino3_fn(m, frames):
        return dino3_patch_feats(m, frames, n_prefix=n_prefix)
    run_encoder("dino3", dino3_fn, dino3)

    # ---- SigLIP 2 ----
    print("[*] loading SigLIP 2 ViT-L/16 ...", flush=True)
    siglip = SiglipVisionModel.from_pretrained(SIGLIP2_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    run_encoder("siglip", siglip_patch_feats, siglip)

    print("[*] saving npz ...", flush=True)
    npz_path = "/tmp/encoder_features_5x.npz"
    np.savez(npz_path, **out_arrays)

    # summary json
    label_map = {"clip":"CLIP", "dino":"DINOv2", "dino3":"DINOv3",
                 "siglip":"SigLIP2", "vjepa":"V-JEPA"}
    summary = {"encoders": {}}
    print("\n=== mean per-patch shift (per encoder, per pair) ===", flush=True)
    print(f"{'seq':<10s} " + "  ".join(f"{label_map[e]:>8s}" for e in ["vjepa","clip","dino","dino3","siglip"]))
    for e in ["vjepa","clip","dino","dino3","siglip"]:
        pp = [float(out_arrays[f"{e}_shift"][i].mean()) for i in range(15)]
        mean_shift = float(np.mean(pp))
        summary["encoders"][label_map[e]] = {
            "mean_shift": mean_shift,
            "per_pair": {seq: pp[i] for i, seq in enumerate(SEQ_NAMES)},
        }
    for i, seq in enumerate(SEQ_NAMES):
        row = f"{seq:<10s} " + "  ".join(
            f"{summary['encoders'][label_map[e]]['per_pair'][seq]:>8.4f}"
            for e in ["vjepa","clip","dino","dino3","siglip"])
        print(row)
    print("-" * 60)
    row = f"{'MEAN':<10s} " + "  ".join(
        f"{summary['encoders'][label_map[e]]['mean_shift']:>8.4f}"
        for e in ["vjepa","clip","dino","dino3","siglip"])
    print(row)

    json_path = "/tmp/encoder_features_5x_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] wrote {json_path}", flush=True)

    # tar
    tar_path = "/tmp/encoder_features_5x.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(npz_path, arcname=os.path.basename(npz_path))
        tar.add(json_path, arcname=os.path.basename(json_path))
    print(f"[*] wrote {tar_path}", flush=True)
    print(f"[*] total wall time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
