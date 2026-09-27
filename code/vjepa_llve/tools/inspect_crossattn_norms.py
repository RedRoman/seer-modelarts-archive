"""tools/inspect_crossattn_norms.py — BB-supplement.

Dump per-(layer, scale) L2 norm of pixel_branch_v8 VJepaCrossAttnBF
proj.weight and proj.bias for v9.0_best. The proj is zero-initialized
(pixel_branch_v8.py line 130-131) so if it stays near zero after training,
the cross-attn output is being multiplicatively suppressed regardless of
V-JEPA feature content. This is exactly the case BB4 flags as needing to
be disambiguated before v9.4 (zero/noise probes).

Also dump q.weight, kv.weight norms as reference (those are not zero-init).
"""
from __future__ import annotations
import os
import sys
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from tools.audit_llvestcd import build_v9_model

CKPT = "/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt"
CFG = "configs/train_v9.yaml"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_v9_model(CFG, CKPT, device)

    rows = []
    for name, mod in model.pixel_branch.named_modules():
        if mod.__class__.__name__ == "VJepaCrossAttnBF":
            q_w = mod.q.weight.detach()
            kv_w = mod.kv.weight.detach()
            proj_w = mod.proj.weight.detach()
            proj_b = mod.proj.bias.detach()
            rows.append({
                "name": name,
                "q_w_norm": q_w.norm().item(),
                "q_w_max_abs": q_w.abs().max().item(),
                "kv_w_norm": kv_w.norm().item(),
                "kv_w_max_abs": kv_w.abs().max().item(),
                "proj_w_norm": proj_w.norm().item(),
                "proj_w_max_abs": proj_w.abs().max().item(),
                "proj_b_norm": proj_b.norm().item(),
                "proj_b_max_abs": proj_b.abs().max().item(),
                "shape_q_w": tuple(q_w.shape),
                "shape_proj_w": tuple(proj_w.shape),
            })

    print(f"\n=== Found {len(rows)} VJepaCrossAttnBF modules in v9.0_best ===\n")

    print(f"{'name':<60} {'q.w_n':>9} {'kv.w_n':>9} {'proj.w_n':>10} {'proj.w_max':>12} {'proj.b_n':>10}")
    for r in rows:
        print(f"{r['name']:<60} "
              f"{r['q_w_norm']:9.3f} "
              f"{r['kv_w_norm']:9.3f} "
              f"{r['proj_w_norm']:10.4f} "
              f"{r['proj_w_max_abs']:12.4f} "
              f"{r['proj_b_norm']:10.4f}")

    proj_norms = [r["proj_w_norm"] for r in rows]
    proj_max_abs = [r["proj_w_max_abs"] for r in rows]
    print(f"\nproj.weight norm: min={min(proj_norms):.4f} max={max(proj_norms):.4f} mean={sum(proj_norms)/len(proj_norms):.4f}")
    print(f"proj.weight max-abs: min={min(proj_max_abs):.4f} max={max(proj_max_abs):.4f} mean={sum(proj_max_abs)/len(proj_max_abs):.4f}")

    print("\n=== Interpretation rules ===")
    print("If proj.weight norm is near 0 (e.g. < 0.01) for most modules:")
    print("  -> cross-attn output is multiplicatively suppressed regardless of V-JEPA content")
    print("  -> v9.4-zero will trivially show ~0 dB, and the noJEPA gap is NOT inference-time feature-dep")
    print("  -> training-dynamics interpretation (v9.0 vs v9.0-noJEPA gap = different optimization paths) is supported")
    print("If proj.weight norm is non-trivial (>> 0.1):")
    print("  -> cross-attn output IS being mixed in")
    print("  -> v9.4-zero/noise/scene-swap will give the actual content-dependence answer")


if __name__ == "__main__":
    main()
