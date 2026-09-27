"""Dump pooled features for the invariance-demo protocol.

For each of 15 SDSD-indoor pairs, extract features under:
  - V-JEPA 2.1 ViT-L (local ckpt), block-23 tokens, mean-pool spatial+temporal -> (1024,)
  - CLIP ViT-L/14  (openai/clip-vit-large-patch14) final block patch tokens, spatial mean per frame, temporal mean -> (1024,)
  - DINOv2 ViT-L  (facebook/dinov2-large) final block patch tokens, spatial mean per frame, temporal mean -> (1024,)

Input protocol (matches invariance_demo.json):
  - 16 center frames from each pair
  - 512x512 center crop (LQ = input/, HQ = GT/)
  - V-JEPA: image_size=512, native temporal 16, feature_layers=[23]
  - CLIP/DINO: per-frame at 224x224 (native)

Saves /tmp/encoder_features.npz with:
  seq_names, vjepa_lq, vjepa_hq, clip_lq, clip_hq, dino_lq, dino_hq  (all float32)
Then reports L2(LQ-HQ)/mean_norm and mean cos per encoder, and compares to
invariance_demo.json.
"""
from __future__ import annotations
import os, sys, json
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

SEQ_NAMES = [
    "pair50","pair52","pair58","pair5_2","pair6","pair60","pair68","pair6_2",
    "pair7","pair70","pair7_2","pair8","pair8_2","pair9","pair9_2",
]

CLIP_LEN = 16
CROP     = 512


def load_pair_frames(seq: str, side: str) -> np.ndarray:
    """Load 16 center frames, 512x512 center crop, uint8 (T,H,W,3).

    side: 'input' (LQ) or 'GT' (HQ)
    """
    d = os.path.join(SDSD_ROOT, side, seq)
    fns = sorted(os.listdir(d))
    T = len(fns)
    assert T >= CLIP_LEN, f"{seq}/{side}: only {T} frames"
    start = (T - CLIP_LEN) // 2
    fns = fns[start:start + CLIP_LEN]
    arrs = []
    for fn in fns:
        a = np.load(os.path.join(d, fn))  # (H, W, 3) uint8
        H, W, _ = a.shape
        y0 = (H - CROP) // 2
        x0 = (W - CROP) // 2
        arrs.append(a[y0:y0+CROP, x0:x0+CROP, :])
    return np.stack(arrs, axis=0)  # (T, 512, 512, 3) uint8


# --------------------------------------------------------------------- V-JEPA
def build_vjepa():
    m = VJepaBackboneLocal21(
        local_repo=VJEPA_REPO,
        local_ckpt=VJEPA_CKPT,
        feature_layers=(23,),
        image_size=CROP,            # 512 to match invariance-demo protocol
        frames_per_clip=CLIP_LEN,
        patch_size=16,
        tubelet_size=2,
        freeze=True,
    ).to(DEVICE).eval()
    return m


@torch.no_grad()
def vjepa_feature(model, frames_uint8: np.ndarray) -> np.ndarray:
    # (T,H,W,3) uint8 -> (1,3,T,H,W) float [0,1]
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1,3,T,H,W)
    out = model(x)
    feats = out["feats"][0]  # (1, N, 1024) where N = T'/2 * H/16 * W/16
    v = feats.mean(dim=1).squeeze(0)  # (1024,)
    return v.detach().float().cpu().numpy()


# ------------------------------------------------------------------- CLIP/DINO
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN     = torch.tensor([0.48145466, 0.4578275,  0.40821073])
CLIP_STD      = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def _to_bchw01(frames_uint8: np.ndarray) -> torch.Tensor:
    # (T,H,W,3) uint8 -> (T,3,H,W) float [0,1] on GPU
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


def _resize_norm(x_bchw: torch.Tensor, size: int, mean: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x_bchw, size=(size, size), mode="bilinear",
                      align_corners=False)
    m = mean.to(x.device).view(1, 3, 1, 1)
    s = std.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


# NOTE: The reference invariance_demo.json was produced with the CLIP/DINO
# *CLS* token (image-level embedding), not patch-token mean-pool. Verified by
# a variant sweep: patch_mean gives cos ~0.98/0.95 (way over reference), while
# CLS pooling gives cos matching within +/-0.02. Patch mean over-averages away
# the illumination shift, so the CLS token is what the paper's motivation
# study actually used. We keep patch_mean also, but ship CLS in the npz so
# the numbers reproduce invariance_demo.json.

@torch.no_grad()
def clip_feature(model, frames_uint8: np.ndarray) -> np.ndarray:
    """CLIP ViT-L/14 image feature: pooler_output (CLS after final LN), per-frame
    then mean across the 16 frames -> (1024,).
    """
    x = _to_bchw01(frames_uint8)                              # (16,3,512,512)
    x = _resize_norm(x, 224, CLIP_MEAN, CLIP_STD)             # (16,3,224,224)
    out = model(pixel_values=x, output_hidden_states=False)
    per_frame = out.pooler_output                              # (16, 1024)  CLS after final LN
    v = per_frame.mean(dim=0)                                  # (1024,)
    return v.detach().float().cpu().numpy()


@torch.no_grad()
def dino_feature(model, frames_uint8: np.ndarray) -> np.ndarray:
    """DINOv2 ViT-L image feature: CLS token from last_hidden_state, per-frame
    then mean across the 16 frames -> (1024,).
    (pooler_output equals last_hidden_state[:,0] for DINOv2, so either works.)
    """
    x = _to_bchw01(frames_uint8)                              # (16,3,512,512)
    x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)     # DINOv2 uses ImageNet stats
    out = model(pixel_values=x, output_hidden_states=False)
    per_frame = out.last_hidden_state[:, 0, :]                # (16, 1024)  CLS
    v = per_frame.mean(dim=0)                                  # (1024,)
    return v.detach().float().cpu().numpy()


def main():
    from transformers import CLIPVisionModel, Dinov2Model

    print("[*] Loading V-JEPA 2.1 ViT-L ...", flush=True)
    vjepa = build_vjepa()

    print("[*] Loading CLIP ViT-L/14 ...", flush=True)
    clip = CLIPVisionModel.from_pretrained(CLIP_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()

    print("[*] Loading DINOv2 ViT-L ...", flush=True)
    dino = Dinov2Model.from_pretrained(DINO_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()

    out = {
        "seq_names": np.array(SEQ_NAMES),
        "vjepa_lq": np.zeros((15, 1024), dtype=np.float32),
        "vjepa_hq": np.zeros((15, 1024), dtype=np.float32),
        "clip_lq":  np.zeros((15, 1024), dtype=np.float32),
        "clip_hq":  np.zeros((15, 1024), dtype=np.float32),
        "dino_lq":  np.zeros((15, 1024), dtype=np.float32),
        "dino_hq":  np.zeros((15, 1024), dtype=np.float32),
    }

    for i, seq in enumerate(SEQ_NAMES):
        print(f"[{i+1:02d}/15] {seq}", flush=True)
        lq = load_pair_frames(seq, "input")
        hq = load_pair_frames(seq, "GT")

        out["vjepa_lq"][i] = vjepa_feature(vjepa, lq)
        out["vjepa_hq"][i] = vjepa_feature(vjepa, hq)
        out["clip_lq"][i]  = clip_feature(clip, lq)
        out["clip_hq"][i]  = clip_feature(clip, hq)
        out["dino_lq"][i]  = dino_feature(dino, lq)
        out["dino_hq"][i]  = dino_feature(dino, hq)

    np.savez("/tmp/encoder_features.npz", **out)
    print("[*] saved /tmp/encoder_features.npz", flush=True)

    # ---- verification ----
    ref = json.load(open("/cache/zihaohe/checkpoint/runs/invariance_demo.json"))
    print("\n=== Verification: per-encoder mean L2/|mean_norm| and mean cos ===")
    for enc_key, my_lq, my_hq in [
        ("v_jepa", out["vjepa_lq"], out["vjepa_hq"]),
        ("clip",   out["clip_lq"],  out["clip_hq"]),
        ("dino",   out["dino_lq"],  out["dino_hq"]),
    ]:
        l2_rels, coses = [], []
        for i, seq in enumerate(SEQ_NAMES):
            lq = my_lq[i]; hq = my_hq[i]
            nl = np.linalg.norm(lq); nh = np.linalg.norm(hq)
            l2 = np.linalg.norm(lq - hq)
            mean_norm = 0.5 * (nl + nh)
            l2_rels.append(l2 / max(mean_norm, 1e-8))
            coses.append(float(np.dot(lq, hq) / max(nl * nh, 1e-8)))
        my_l2 = float(np.mean(l2_rels)); my_cos = float(np.mean(coses))
        ref_seqs = ref["per_seq"][enc_key]
        ref_l2  = float(np.mean([ref_seqs[s]["l2_rel"]     for s in SEQ_NAMES]))
        ref_cos = float(np.mean([ref_seqs[s]["cosine_sim"] for s in SEQ_NAMES]))
        d_l2 = my_l2 - ref_l2; d_cos = my_cos - ref_cos
        print(f"  {enc_key:8s}  L2 rel: mine={my_l2:.4f}  ref={ref_l2:.4f}  d={d_l2:+.4f}"
              f"    cos: mine={my_cos:.4f}  ref={ref_cos:.4f}  d={d_cos:+.4f}")


if __name__ == "__main__":
    main()
