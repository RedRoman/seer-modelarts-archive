"""V-JEPA invariance probe — GO/NO-GO before training.

Per-layer cosine similarity of V-JEPA features between LQ and HQ frames:
    cos(φ_layer(LQ), φ_layer(HQ))

Decision: GO if best-layer cos >= threshold (default 0.85). Justifies the
"V-JEPA prior is illumination-invariant" claim of the paper.

Probe results expected on SDSD-indoor (per the original 2026-04 study):
    L0 (layer 5):  0.962
    L1 (layer 11): 0.946
    L2 (layer 17): 0.890
    L3 (layer 23): 0.561  (deeper layers more illumination-sensitive)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data import build_dataset
from models.vjepa_backbone import build_vjepa_backbone
from utils import dump_config, ensure_dir, load_config, set_seed, setup_logger


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = F.normalize(a.reshape(-1, a.shape[-1]), dim=-1)
    b_n = F.normalize(b.reshape(-1, b.shape[-1]), dim=-1)
    return (a_n * b_n).sum(dim=-1).mean()


def norm_ratio(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    an = a.reshape(-1, a.shape[-1]).norm(dim=-1)
    bn = b.reshape(-1, b.shape[-1]).norm(dim=-1)
    return (an / (bn + eps)).mean()


def linear_probe_r2(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1.0) -> float:
    assert X.shape == Y.shape
    N, D = X.shape
    perm = torch.randperm(N)
    half = N // 2
    tr, te = perm[:half], perm[half:]
    Xt, Yt = X[tr].float(), Y[tr].float()
    Xe, Ye = X[te].float(), Y[te].float()
    XtX = Xt.T @ Xt + ridge * torch.eye(D, device=X.device, dtype=Xt.dtype)
    W = torch.linalg.solve(XtX, Xt.T @ Yt)
    Yhat = Xe @ W
    ss_res = ((Ye - Yhat) ** 2).sum()
    ss_tot = ((Ye - Ye.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1.0 - ss_res / ss_tot.clamp_min(1e-12)).item())


def pca_to_rgb(tokens: torch.Tensor, grid: Tuple[int, int, int]) -> np.ndarray:
    T, H, W = grid
    X = tokens.detach().float().cpu().numpy()
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / max(Xc.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov)
    proj = (Xc @ v[:, -3:])
    proj = proj - proj.min(axis=0, keepdims=True)
    proj = (proj / proj.max(axis=0, keepdims=True).clip(min=1e-8) * 255).astype(np.uint8)
    img = proj.reshape(T, H, W, 3)
    return img.transpose(1, 0, 2, 3).reshape(H, T * W, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    out_dir = ensure_dir(cfg["output_dir"])
    samples_dir = ensure_dir(os.path.join(out_dir, "samples"))
    logger = setup_logger(log_dir=out_dir, name="vjepa_probe")
    dump_config(cfg, os.path.join(out_dir, "probe_config.yaml"))

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))

    pcfg = cfg["probe"]
    n_samples = int(pcfg.get("n_samples", 24))
    save_pca = bool(pcfg.get("save_pca_viz", True))
    th_cos = float(pcfg.get("threshold_best_layer_cos", 0.85))
    th_r2  = float(pcfg.get("threshold_r2", 0.4))

    # Load val data
    val_set = build_dataset(
        cfg["data"]["val"],
        clip_len=pcfg.get("clip_len", 16),
        crop_size=pcfg.get("crop_size", 256),
        is_train=False,
    )
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    logger.info(f"Val set: {len(val_set)} sequences")

    backbone = build_vjepa_backbone(cfg["model"]["backbone"]).to(device).eval()

    layers = list(cfg["model"]["backbone"].get("feature_layers", [5, 11, 17, 23]))
    L = len(layers)

    cos_per_layer = [[] for _ in range(L)]
    norm_per_layer = [[] for _ in range(L)]
    r2_deepest = []

    seen = 0
    pca_saved = 0
    for batch in loader:
        if seen >= n_samples:
            break
        lq = batch["lq"].to(device)
        hq = batch["hq"].to(device)
        with torch.no_grad():
            fl = backbone(lq)["feats"]
            fh = backbone(hq)["feats"]
        for k in range(L):
            cos_per_layer[k].append(cosine_sim(fl[k], fh[k]).item())
            norm_per_layer[k].append(norm_ratio(fl[k], fh[k]).item())
        # Linear probe on deepest layer
        Xl = fl[-1].reshape(-1, fl[-1].shape[-1])
        Yh = fh[-1].reshape(-1, fh[-1].shape[-1])
        r2_deepest.append(linear_probe_r2(Xl, Yh))

        if save_pca and pca_saved < 4:
            grid = backbone(lq)["token_grid"]
            img_lq = pca_to_rgb(fl[-1].squeeze(0), grid)
            img_hq = pca_to_rgb(fh[-1].squeeze(0), grid)
            comb = np.concatenate([img_lq, img_hq], axis=0)
            Image.fromarray(comb).save(
                os.path.join(samples_dir, f"pca_{pca_saved:03d}.png")
            )
            pca_saved += 1

        seen += 1

    cos_mean = [float(np.mean(c)) for c in cos_per_layer]
    norm_mean = [float(np.mean(c)) for c in norm_per_layer]
    r2_mean = float(np.mean(r2_deepest))
    cos_best_idx = int(np.argmax(cos_mean))
    cos_best = cos_mean[cos_best_idx]

    summary = {
        "n_samples": seen,
        "feature_layers": layers,
        "cos_per_layer_mean": cos_mean,
        "norm_ratio_per_layer_mean": norm_mean,
        "deepest_layer_linear_probe_r2": r2_mean,
        "best_layer_cos": cos_best,
        "best_layer_idx": cos_best_idx,
        "best_layer_global": layers[cos_best_idx],
        "thresholds": {"best_layer_cos": th_cos, "deepest_r2": th_r2},
        "decision": "GO" if cos_best >= th_cos else "NO-GO",
    }

    out_json = os.path.join(out_dir, "summary.json")
    out_txt  = os.path.join(out_dir, "summary.txt")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"=== V-JEPA invariance probe (n={seen}) ===",
        f"feature_layers (V-JEPA block indices): {layers}",
        "",
        "Per-layer cosine sim (LQ vs HQ):",
    ]
    for k, l in enumerate(layers):
        lines.append(f"  L{k} (layer {l}): cos={cos_mean[k]:.4f}  |φ(LQ)|/|φ(HQ)|={norm_mean[k]:.4f}")
    lines.extend([
        "",
        f"Deepest-layer linear probe R² (HQ ~= W·LQ): {r2_mean:.4f}",
        f"Best-layer cosine: layer {layers[cos_best_idx]} (idx {cos_best_idx}) = {cos_best:.4f}",
        "",
        f"Threshold: best-layer cos >= {th_cos}",
        f"Decision: {summary['decision']}",
    ])
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

    for line in lines:
        logger.info(line)


if __name__ == "__main__":
    main()
