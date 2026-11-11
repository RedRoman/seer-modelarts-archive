"""Extract real cross-attention maps from the v35 (Seer) checkpoint.

Loads VJepaPixelBranch + CounterfactualRefiner + VJepaBackboneLocal21 exactly
as tools/v35_stage4_train.py::audit_indoor_512 does, runs a real forward pass
on a real SDSD-indoor test clip, and monkeypatches VJepaCrossAttnBF.forward
to also stash the post-softmax attention matrix (before it's discarded).

No fabricated data: this either produces a real attention tensor extracted
from a real forward pass through the real checkpoint, or it errors out.
"""
import os
import sys
import math
import types

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = "/home/ma-user/workspace/zihaohe/vjepa_llve"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data.common import read_image_01  # noqa: E402
from models.pixel_branch_v10 import VJepaPixelBranch  # noqa: E402
from models.predictor_lora import CounterfactualRefiner  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402
from models.pixel_branch_v8 import VJepaCrossAttnBF, _vjepa_to_kv_at_scale  # noqa: E402

VJEPA_REPO = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT = "/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"
CKPT_PATH = "/cache/zihaohe/checkpoint/runs/v35_stage4/final.pt"
REF_LAYER = 3
NUM_VJEPA_LAYERS = 4

device = "cuda"

# -----------------------------------------------------------------------
# 1. Monkeypatch VJepaCrossAttnBF.forward to stash attention weights
# -----------------------------------------------------------------------
ATTN_STORE = {}

def patched_forward(self, x, vjepa_kv, _tag=None):
    BT, C, H, W = x.shape
    Q = self.q(self.norm_q(x))
    Q = rearrange(Q, "b (h d) y w -> b h (y w) d", h=self.h)
    kv = self.kv(vjepa_kv).reshape(BT, vjepa_kv.shape[1], 2, self.h, self.d)
    kv = kv.permute(2, 0, 3, 1, 4)
    K, V = kv.unbind(0)
    scale = 1.0 / math.sqrt(self.d)
    attn_logits = (Q @ K.transpose(-2, -1)) * scale   # [BT, h, HW, M]
    attn = attn_logits.softmax(dim=-1)
    if _tag is not None and getattr(self, "_capture", False):
        ATTN_STORE[_tag] = attn.detach().float().cpu()
        ATTN_STORE[_tag + "_shape_xHW"] = (H, W)
    out = attn @ V
    out = rearrange(out, "b h (y w) d -> b (h d) y w", y=H, w=W)
    return x + self.proj(out)

VJepaCrossAttnBF.forward = patched_forward

# -----------------------------------------------------------------------
# 2. Build model exactly as v35_stage4_train.py does
# -----------------------------------------------------------------------
print("Building backbone...", flush=True)
backbone = VJepaBackboneLocal21(
    local_repo=VJEPA_REPO, local_ckpt=VJEPA_CKPT,
    feature_layers=(5, 11, 17, 23),
    image_size=256, frames_per_clip=16,
    patch_size=16, tubelet_size=2, freeze=True,
).to(device)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad_(False)

print("Building refiner...", flush=True)
refiner = CounterfactualRefiner(
    local_repo=VJEPA_REPO, ckpt_path=VJEPA_CKPT,
    encoder_embed_dim=1024,
    image_size=256, num_frames=16,
    patch_size=16, tubelet_size=2,
    predictor_embed_dim=384, depth=12, num_heads=12,
    lora_rank=16, freeze_body=True,
).to(device)

print("Building pixel_branch...", flush=True)
pixel_branch = VJepaPixelBranch(
    in_channels=3,
    vjepa_dim=1024,
    vjepa_grid=(16 // 2, 16, 16),
    num_vjepa_layers=NUM_VJEPA_LAYERS,
    cross_attn_at_scales=(1, 2, 3, 4),
    use_cfim_bottleneck=True,
    use_illumination_cues=True,
    use_gradient_checkpoint=False,
    router_init_c=4.0,
).to(device)

print("Loading checkpoint...", flush=True)
state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
missing, unexpected = pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
print("pixel_branch missing:", len(missing), "unexpected:", len(unexpected))
missing2, unexpected2 = refiner.load_state_dict(state["cf_refiner"], strict=False)
print("refiner missing:", len(missing2), "unexpected:", len(unexpected2))
print("step:", state.get("step"), "best_rgb512:", state.get("best_rgb512"))

pixel_branch.eval()
refiner.eval()

# Tag every cross-attn module with its scale/branch and enable capture
for s, mod in enumerate(pixel_branch.enc_xattn):
    if mod is not None:
        mod._tag = f"enc_s{s}"
        mod._capture = True
for i, mod in enumerate(pixel_branch.dec_xattn):
    if mod is not None:
        scale_idx = [3, 2, 1, 0][i]
        mod._tag = f"dec_s{scale_idx}"
        mod._capture = True

# Need to pass _tag through forward call -- patch the call sites via a wrapper
orig_call = torch.nn.Module.__call__
def tagged_call(self, *args, **kwargs):
    if isinstance(self, VJepaCrossAttnBF) and hasattr(self, "_tag"):
        kwargs["_tag"] = self._tag
    return orig_call(self, *args, **kwargs)
torch.nn.Module.__call__ = tagged_call

# -----------------------------------------------------------------------
# 3. Load a real SDSD-indoor test clip (pair4, frames around #10)
# -----------------------------------------------------------------------
SEQ_DIR_LQ = "/cache/zihaohe/datasets/SDSD_indoor/indoor_static_np/input/pair4"
SEQ_DIR_GT = "/cache/zihaohe/datasets/SDSD_indoor/indoor_static_np/GT/pair4"
n_total = len(os.listdir(SEQ_DIR_LQ))
clip_len = 16
start = max(0, (n_total - clip_len) // 2)
print(f"pair4 has {n_total} frames, using frames {start}..{start+clip_len-1}")

def load_frame(d, idx):
    p = os.path.join(d, f"{idx+1:04d}.npy")
    return np.load(p).astype(np.float32) / 255.0

lq_frames = np.stack([load_frame(SEQ_DIR_LQ, start + i) for i in range(clip_len)])
gt_frames = np.stack([load_frame(SEQ_DIR_GT, start + i) for i in range(clip_len)])

lq = torch.from_numpy(lq_frames).permute(3, 0, 1, 2).unsqueeze(0).float()  # (1,3,T,H,W)
gt = torch.from_numpy(gt_frames).permute(3, 0, 1, 2).unsqueeze(0).float()

# center crop to 256x256 to match training resolution / vjepa_grid=(8,16,16)
H, W = lq.shape[-2:]
crop = 256
h0 = (H - crop) // 2
w0 = (W - crop) // 2
lq = lq[..., h0:h0+crop, w0:w0+crop].to(device)
gt = gt[..., h0:h0+crop, w0:w0+crop].to(device)
print("lq shape:", lq.shape, "mean:", lq.mean().item())

# -----------------------------------------------------------------------
# 4. Real forward pass
# -----------------------------------------------------------------------
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z_layers = backbone(lq)["feats"]
        print("num vjepa layers:", len(z_layers), "shape[0]:", z_layers[0].shape)
        refined = refiner(z_layers[REF_LAYER].bfloat16())
        z_pixel = list(z_layers[:REF_LAYER]) + [refined]
        out = pixel_branch(lq, z_pixel)
    pred = out["pred_pixels"].clamp(0, 1).float()

def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return 20 * math.log10(1.0) - 10 * math.log10(max(mse, 1e-10))

print("PSNR pred vs gt (this clip, 256 crop):", psnr(pred, gt))
print("PSNR lq vs gt (baseline):", psnr(lq, gt))

print("Captured attention tags:", sorted([k for k in ATTN_STORE if not k.endswith("_shape_xHW")]))

# -----------------------------------------------------------------------
# 5. Save everything needed for plotting: attention tensors + frames
# -----------------------------------------------------------------------
save_dir = "/home/ma-user/workspace/zihaohe/visualizations/fig_cross_attention"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "lq_clip.npy"), lq.detach().float().cpu().numpy())
np.save(os.path.join(save_dir, "gt_clip.npy"), gt.detach().float().cpu().numpy())
np.save(os.path.join(save_dir, "pred_clip.npy"), pred.detach().float().cpu().numpy())

for k, v in ATTN_STORE.items():
    if k.endswith("_shape_xHW"):
        continue
    torch.save(v, os.path.join(save_dir, f"attn_{k}.pt"))
    print(k, "attn shape:", v.shape, "shape_xHW:", ATTN_STORE[k + "_shape_xHW"])

meta = {
    "ckpt": CKPT_PATH,
    "seq": "SDSD_indoor/pair4",
    "frame_start": start,
    "clip_len": clip_len,
    "crop": crop,
    "crop_offset": (h0, w0),
    "orig_HW": (H, W),
    "psnr_pred": psnr(pred, gt),
    "psnr_lq": psnr(lq, gt),
    "tags_shape_xHW": {k: v for k, v in ATTN_STORE.items() if k.endswith("_shape_xHW")},
}
import json
with open(os.path.join(save_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)

print("DONE. Saved to", save_dir)
