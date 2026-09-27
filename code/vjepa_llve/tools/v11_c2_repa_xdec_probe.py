"""F6: Cross-decoder portability probe for E2b.

Question: does REPA training teach the decoder to encode "V-JEPA-aligned features",
or does it just learn arbitrary task-specific features that happen to satisfy the
alignment loss?

Test: train two decoders with REPA (different seeds), check if the projector trained
on decoder A also produces good alignment when fed decoder B's bottleneck (and vice
versa). Strong portability → REPA shapes a portable representation; poor portability
→ projector co-adapts to one decoder's idiosyncratic basis.

This is a lightweight version: load 3 existing E2b checkpoints (seeds 42/0/100), and
for each pair (i, j), evaluate cosine alignment of decoder_i's h_b through projector_j.
Report a 3x3 matrix.
"""
from __future__ import annotations
import argparse, os, sys, time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.v11_oracle_delta_probe import load_clip, encode, build_split_seqs


class REPAProjector(nn.Module):
    def __init__(self, in_ch=256, out_dim=1024):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)

    def forward(self, h_b, vjepa_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=vjepa_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


@torch.no_grad()
def alignment_cos(decoder, projector, backbone, lq, ref_layer):
    h_b, _ = decoder.encode(lq, kv_layers=None, delta_layers=None)
    z_L = encode(backbone, lq)
    z_stu = projector(h_b, tuple(decoder.vjepa_grid)).float()
    z_tea = z_L[ref_layer].float()
    m = z_tea.mean(dim=1, keepdim=True); s = z_tea.std(dim=1, keepdim=True) + 1e-6
    z_tea = (z_tea - m) / s
    z_stu_n = F.normalize(z_stu, dim=-1, eps=1e-3)
    z_tea_n = F.normalize(z_tea, dim=-1, eps=1e-3)
    return float((z_stu_n * z_tea_n).sum(dim=-1).clamp(-1.5, 1.5).mean().item())


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    ckpts = args.ckpts.split(",")
    print(f"[xdec] loading {len(ckpts)} ckpts: {ckpts}")
    decoders, projectors = [], []
    for ck in ckpts:
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        d = DeltaModulatedDecoder(delta_in_encoder=False).to(device).eval()
        d.load_state_dict(sd["decoder"])
        p = REPAProjector(in_ch=d.dims[-1], out_dim=d.vjepa_dim).to(device).eval()
        p.load_state_dict(sd["projector"])
        decoders.append(d); projectors.append(p)
        for q in d.parameters(): q.requires_grad_(False)
        for q in p.parameters(): q.requires_grad_(False)

    _, test_seqs = build_split_seqs()
    n = len(decoders)
    M = np.zeros((n, n))
    for i, (di, _) in enumerate(zip(decoders, projectors)):
        for j, (_, pj) in enumerate(zip(decoders, projectors)):
            cos_vals = []
            with torch.no_grad():
                for name, lq_p, gt_p in test_seqs:
                    lq, _ = load_clip(lq_p, gt_p, crop_size=args.crop)
                    lq = lq.to(device)
                    cos_vals.append(alignment_cos(di, pj, backbone, lq, args.ref_layer))
            M[i, j] = float(np.mean(cos_vals))
            print(f"  decoder_{i}  ×  projector_{j}  cos = {M[i,j]:.4f}")

    md = os.path.join(args.out_dir, "XDEC_PROBE.md")
    with open(md, "w") as f:
        f.write("# Cross-decoder REPA portability probe\n\n")
        f.write("Cell [i,j] = mean cosine on SDSD held-out, decoder_i's h_b -> projector_j -> V-JEPA layer 3.\n\n")
        f.write("Diagonal = self (in-distribution); off-diagonal = cross-decoder portability.\n\n")
        f.write("ckpts:\n")
        for i, ck in enumerate(ckpts): f.write(f"  - [{i}] {ck}\n")
        f.write("\n| i\\j | " + " | ".join(str(j) for j in range(n)) + " |\n")
        f.write("|---|" + "|".join(["---"] * n) + "|\n")
        for i in range(n):
            f.write(f"| {i} | " + " | ".join(f"{M[i,j]:.4f}" for j in range(n)) + " |\n")
        diag_mean = float(np.mean([M[i, i] for i in range(n)]))
        offdiag = [M[i, j] for i in range(n) for j in range(n) if i != j]
        off_mean = float(np.mean(offdiag)) if offdiag else 0.0
        gap = diag_mean - off_mean
        f.write(f"\n## diag mean = {diag_mean:.4f}\n")
        f.write(f"## off-diag mean = {off_mean:.4f}\n")
        f.write(f"## portability gap = {gap:.4f}  (small gap = portable; large gap = decoder-specific)\n")
    print(f"\n[xdec] diag={diag_mean:.4f}  off-diag={off_mean:.4f}  gap={gap:.4f}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpts", required=True, help="comma-sep list of ckpt.pt paths")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--ref_layer", type=int, default=3)
    main(ap.parse_args())
