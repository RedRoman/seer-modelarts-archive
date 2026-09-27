"""Verify environment setup before training.

Checks:
  1. Imports (torch, diffusers, lpips, einops, etc.)
  2. V-JEPA 2.1 backbone load (verifies ckpt + modelzoo path)
  3. SD VAE load (downloads if missing, verifies HF cache)
  4. Dataset roots exist and are readable
  5. CUDA + bf16 support
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def section(name: str):
    print(f"\n=== {name} ===")


def check_imports() -> bool:
    section("imports")
    ok = True
    for name in ("torch", "torch.nn", "diffusers", "lpips", "einops", "yaml",
                 "numpy", "PIL", "transformers"):
        try:
            __import__(name)
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            ok = False
    return ok


def check_cuda() -> bool:
    section("cuda + bf16")
    import torch
    if not torch.cuda.is_available():
        print("  [FAIL] CUDA not available")
        return False
    print(f"  [OK] CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}")
    print(f"       device 0: {torch.cuda.get_device_name(0)}")
    print(f"       bf16 supported: {torch.cuda.is_bf16_supported()}")
    return True


def check_backbone(skip: bool) -> bool:
    if skip:
        print("\n[skip] V-JEPA backbone")
        return True
    section("V-JEPA 2.1 backbone")
    try:
        from utils.config import load_config
        cfg = load_config(os.path.join(_REPO_ROOT, "configs/base_21.yaml"))
        from models.vjepa_backbone import build_vjepa_backbone
        backbone = build_vjepa_backbone(cfg["model"]["backbone"])
        print(f"  [OK] V-JEPA 2.1 ViT-L hidden_dim={backbone.hidden_dim}")
        # Tiny forward
        import torch
        backbone = backbone.to("cuda:0").eval()
        x = torch.rand(1, 3, 16, 256, 256, device="cuda:0").clamp(0, 1)
        with torch.no_grad():
            out = backbone(x)
        print(f"       feats: {len(out['feats'])} layers, "
              f"each {tuple(out['feats'][0].shape)}, grid={out['token_grid']}")
        return True
    except Exception as e:
        print(f"  [FAIL] backbone: {e}")
        return False


def check_vae(skip: bool) -> bool:
    if skip:
        print("\n[skip] VAE")
        return True
    section("SD VAE")
    try:
        from models.vae_wrapper import SDVAEWrapper
        import torch
        vae = SDVAEWrapper().to("cuda:0").eval()
        print(f"  [OK] VAE latent_channels={vae.latent_channels}, "
              f"downsample={vae.downsample}")
        x = torch.rand(1, 3, 4, 256, 256, device="cuda:0").clamp(0, 1)
        with torch.no_grad():
            z = vae.encode(x)
            x_rec = vae.decode(z)
        print(f"       z shape {tuple(z.shape)}, recon shape {tuple(x_rec.shape)}")
        return True
    except Exception as e:
        print(f"  [FAIL] VAE: {e}")
        return False


def check_datasets(skip: bool) -> bool:
    if skip:
        print("\n[skip] datasets")
        return True
    section("datasets")
    roots = [
        "/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor",
        "/home/ma-user/workspace/zihaohe/datasets/SDSD_outdoor",
    ]
    ok = True
    for root in roots:
        if os.path.isdir(root):
            n = sum(1 for _ in os.scandir(root))
            print(f"  [OK] {root}  ({n} entries)")
        else:
            print(f"  [WARN] {root} not found")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_backbone", action="store_true")
    ap.add_argument("--skip_vae", action="store_true")
    ap.add_argument("--skip_datasets", action="store_true")
    args = ap.parse_args()

    results = []
    results.append(("imports", check_imports()))
    results.append(("cuda", check_cuda()))
    results.append(("backbone", check_backbone(args.skip_backbone)))
    results.append(("vae", check_vae(args.skip_vae)))
    results.append(("datasets", check_datasets(args.skip_datasets)))

    print("\n=== summary ===")
    for name, ok in results:
        print(f"  {'[OK]' if ok else '[FAIL]'}  {name}")
    if not all(ok for _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
