"""tools/vjepa_highres_sanity.py — BJ3 Step 1.

Acceptance probe: can VJepaBackboneLocal21 be instantiated at non-256 image_size
and produce token_grid != (8, 16, 16) without OOM / shape error?

Test img_size: 512 (a safe stepping stone toward full 1088x1920).
Input: random noise (1, 3, 16, 512, 512) -- no need for real data for this probe.

Pass criterion:
  - Backbone returns token_grid_h/w == 32 (= 512/16) [or non-(16,16)].
  - VRAM peak < 60GB.
  - No exception.

If pass: print token_grid, VRAM peak, runtime. Higher-res experiments unblocked.
If fail: print failure mode + traceback. tile=512 protocol stays as the workaround.
"""
from __future__ import annotations
import os
import sys
import time
import traceback

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone


def main():
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    for hires in (512, 768, 1024):
        print(f"\n===== Trying image_size = {hires} =====")
        try:
            cfg = {
                "source": "local_21",
                "local_repo": "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main",
                "local_ckpt": "/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt",
                "feature_layers": (5, 11, 17, 23),
                "image_size": hires,
                "frames_per_clip": 16,
                "patch_size": 16,
                "tubelet_size": 2,
                "freeze": True,
            }
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            backbone = build_vjepa_backbone(cfg).to(device).eval()
            t_build = time.time() - t0

            # forward random video
            B, C, T = 1, 3, 16
            x = torch.rand(B, C, T, hires, hires, device=device)
            t0 = time.time()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = backbone(x)
            t_fwd = time.time() - t0
            vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

            feats = out["feats"]
            token_grid = out["token_grid"]
            print(f"  ✓ image_size={hires}: token_grid={token_grid}")
            print(f"    n_feature_layers={len(feats)}  per-layer shape={tuple(feats[0].shape)}")
            print(f"    build={t_build:.1f}s  forward={t_fwd:.1f}s  vram_peak={vram_peak_gb:.2f} GB")

            # Cleanup before next
            del backbone, x, out, feats
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ image_size={hires}: FAILED")
            print(f"    {type(e).__name__}: {e}")
            tb = traceback.format_exc().splitlines()
            print("    " + "\n    ".join(tb[-8:]))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
