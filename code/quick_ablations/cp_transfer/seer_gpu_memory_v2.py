"""Task 2: real peak GPU memory measurement for Seer on a 16-frame 512x512
clip, using the actual trained checkpoint
/cache/zihaohe/checkpoint/runs/v35_stage4/final.pt (step 6000, the final
Stage-4 Seer checkpoint reported in the paper).

Model construction mirrors tools/v35_stage4_train.py exactly (backbone +
CounterfactualRefiner + VJepaPixelBranch, same kwargs), loaded at
image_size=512 (matching the paper's Table-1 16-frame 512x512 protocol)
instead of the 256 used only for the *training* crop size.

We measure:
  - torch.cuda.max_memory_allocated() around a single forward pass
    (reset_peak_memory_stats right before, after a warmup pass, so the
    number reflects one steady-state inference, not first-call allocator
    overhead).
  - nvidia-smi memory.used snapshot before/after as a cross-check.

This runs standalone (own process) so measurements aren't contaminated by
anything else that may have been resident from earlier experiments.
"""
from __future__ import annotations
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

try:
    import torch._dynamo  # noqa: F401
except Exception:
    pass
try:
    import einops  # noqa: F401
    _ = einops.rearrange(torch.zeros(1, 2), "a b -> b a")
except Exception:
    pass

_REPO = "/home/ma-user/workspace/zihaohe/vjepa_llve"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.pixel_branch_v10 import VJepaPixelBranch  # noqa: E402
from models.predictor_lora import CounterfactualRefiner  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402

VJEPA_REPO = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT = "/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"
SEER_CKPT = "/cache/zihaohe/checkpoint/runs/v35_stage4/final.pt"
REF_LAYER = 3
NUM_VJEPA_LAYERS = 4
DEVICE = torch.device("cuda")
OUT_PATH = "/home/ma-user/workspace/zihaohe/quick_ablations/cp_transfer/seer_gpu_memory.json"


def nvidia_smi_used_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip().splitlines()[0]
    return float(out.strip())


def build_seer(clip_len=16, image_size=512):
    backbone = VJepaBackboneLocal21(
        local_repo=VJEPA_REPO, local_ckpt=VJEPA_CKPT,
        feature_layers=(5, 11, 17, 23),
        image_size=image_size, frames_per_clip=clip_len,
        patch_size=16, tubelet_size=2, freeze=True,
    ).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    refiner = CounterfactualRefiner(
        local_repo=VJEPA_REPO, ckpt_path=VJEPA_CKPT,
        encoder_embed_dim=1024,
        image_size=image_size, num_frames=clip_len,
        patch_size=16, tubelet_size=2,
        predictor_embed_dim=384, depth=12, num_heads=12,
        lora_rank=16, freeze_body=True,
    ).to(DEVICE)

    pixel_branch = VJepaPixelBranch(
        in_channels=3,
        vjepa_dim=1024,
        vjepa_grid=(clip_len // 2, image_size // 16, image_size // 16),
        num_vjepa_layers=NUM_VJEPA_LAYERS,
        cross_attn_at_scales=(1, 2, 3, 4),
        use_cfim_bottleneck=True,
        use_illumination_cues=True,
        use_gradient_checkpoint=True,
        router_init_c=4.0,
    ).to(DEVICE)

    state = torch.load(SEER_CKPT, map_location="cpu", weights_only=False)
    miss_pb, unex_pb = pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    miss_rf, unex_rf = refiner.load_state_dict(state["cf_refiner"], strict=False)
    print(f"[load] pixel_branch missing={len(miss_pb)} unexpected={len(unex_pb)}")
    print(f"[load] cf_refiner   missing={len(miss_rf)} unexpected={len(unex_rf)}")
    print(f"[load] ckpt step={state.get('step')} best_y512={state.get('best_y512')} "
          f"best_rgb512={state.get('best_rgb512')}")

    backbone.eval(); refiner.eval(); pixel_branch.eval()
    return backbone, refiner, pixel_branch, state.get("step", -1)


@torch.no_grad()
def run_one_forward(backbone, refiner, pixel_branch, lq):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z_layers = backbone(lq)["feats"]
        refined = refiner(z_layers[REF_LAYER].bfloat16())
        z_pixel = list(z_layers[:REF_LAYER]) + [refined]
        out = pixel_branch(lq, z_pixel)
    return out["pred_pixels"].clamp(0, 1).float()


def main():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    used_before_build = nvidia_smi_used_mb()
    print(f"[nvidia-smi] used before model build: {used_before_build:.1f} MB")

    t0 = time.time()
    backbone, refiner, pixel_branch, ckpt_step = build_seer(clip_len=16, image_size=512)
    torch.cuda.synchronize()
    print(f"[build] {time.time()-t0:.1f}s")

    used_after_build = nvidia_smi_used_mb()
    print(f"[nvidia-smi] used after model build (weights resident): {used_after_build:.1f} MB")

    lq = torch.rand(1, 3, 16, 512, 512, device=DEVICE)

    # Warmup (allocator settles, cudnn autotune, etc.)
    for _ in range(2):
        _ = run_one_forward(backbone, refiner, pixel_branch, lq)
    torch.cuda.synchronize()

    # Fresh peak-memory measurement around a single clean forward.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    used_before_fwd = nvidia_smi_used_mb()

    t0 = time.time()
    pred = run_one_forward(backbone, refiner, pixel_branch, lq)
    torch.cuda.synchronize()
    fwd_time_s = time.time() - t0

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    used_after_fwd = nvidia_smi_used_mb()

    print(f"[forward] pred shape={tuple(pred.shape)} time={fwd_time_s:.3f}s")
    print(f"[torch.cuda] peak_allocated={peak_bytes/1024/1024:.1f} MB  "
          f"peak_reserved={peak_reserved_bytes/1024/1024:.1f} MB")
    print(f"[nvidia-smi] used before fwd={used_before_fwd:.1f} MB  after fwd={used_after_fwd:.1f} MB  "
          f"delta={used_after_fwd-used_before_fwd:.1f} MB")

    result = {
        "checkpoint": SEER_CKPT,
        "ckpt_step": ckpt_step,
        "input_shape": [1, 3, 16, 512, 512],
        "image_size_backbone": 512,
        "torch_cuda_peak_allocated_MB": peak_bytes / 1024 / 1024,
        "torch_cuda_peak_reserved_MB": peak_reserved_bytes / 1024 / 1024,
        "nvidia_smi_used_before_build_MB": used_before_build,
        "nvidia_smi_used_after_build_MB": used_after_build,
        "nvidia_smi_used_before_fwd_MB": used_before_fwd,
        "nvidia_smi_used_after_fwd_MB": used_after_fwd,
        "nvidia_smi_delta_fwd_MB": used_after_fwd - used_before_fwd,
        "fwd_time_s": fwd_time_s,
        "autocast": "bfloat16",
        "note": ("Single full forward, no chunking (T=16 all at once), matching "
                 "the appendix runtime protocol (16-frame 512x512 clip). "
                 "torch.cuda peak_allocated is the authoritative number; "
                 "nvidia-smi numbers are a cross-check and include CUDA context "
                 "+ cudnn workspace overhead beyond torch's allocator."),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVE] {OUT_PATH}")


if __name__ == "__main__":
    main()
