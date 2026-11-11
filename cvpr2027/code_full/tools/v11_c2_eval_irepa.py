"""Eval for iREPA adapter variant (E1)."""
from __future__ import annotations
import argparse, csv, os, sys

import numpy as np
import torch
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from models.v11.content_attn_adapter_v3 import ContentAttnAdapterIREPA
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode, build_split_seqs)


@torch.no_grad()
def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device).eval()
    adapter = ContentAttnAdapterIREPA(
        bottleneck_channels=decoder.dims[-1], vjepa_grid=tuple(decoder.vjepa_grid),
        vjepa_dim=decoder.vjepa_dim, active_scales=(3, 4), ref_layer=3,
        attn_heads=8, ffn_mult=args.ffn_mult,
    ).to(device).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    decoder.load_state_dict(sd["decoder"]); adapter.load_state_dict(sd["adapter"])

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    conditions = ["no_prior", "passive_zL", "adapted_zL", "shuffled_adapted",
                  "zero_zL_hb_only", "gt_vjepa_diag"]
    results = {}
    for cond in conditions:
        psnrs = []
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_L = encode(backbone, lq)
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
            if cond == "no_prior":      kv_l, kv_o = None, None
            elif cond == "passive_zL":  kv_l, kv_o = z_L, None
            elif cond == "adapted_zL":  kv_l, kv_o = None, adapter(z_L, h_b)
            elif cond == "shuffled_adapted":
                other = int(rng.integers(0, len(train_seqs)))
                _, lq_o, gt_o = train_seqs[other]
                lq_oo, _ = load_clip(lq_o, gt_o, crop_size=args.crop)
                z_L_o = encode(backbone, lq_oo.to(device))
                kv_l, kv_o = None, adapter(z_L_o, h_b)
            elif cond == "zero_zL_hb_only":
                kv_l, kv_o = None, adapter([torch.zeros_like(z) for z in z_L], h_b)
            else:  # gt_vjepa_diag
                z_H = encode(backbone, gt)
                kv_l, kv_o = None, adapter(z_H, h_b)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=kv_l, delta_layers=None, kv_override=kv_o)
            psnrs.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
        results[cond] = float(np.mean(psnrs))
        print(f"  {cond}: {results[cond]:.3f}")

    no_prior = results["no_prior"]; passive = results["passive_zL"]
    adapted = results["adapted_zL"]; shuffled = results["shuffled_adapted"]
    zerov = results["zero_zL_hb_only"]; gt_diag = results["gt_vjepa_diag"]
    md = os.path.join(args.out_dir, "C2_EVAL_SUMMARY.md")
    with open(md, "w") as f:
        f.write(f"# C-2 iREPA variant (E1) Eval\n\n")
        f.write(f"ckpt: {args.ckpt}\n\n")
        for c in conditions: f.write(f"| {c} | {results[c]:.3f} |\n")
        f.write(f"\nGates: adp-pass={adapted-passive:+.2f}, adp-none={adapted-no_prior:+.2f}, "
                f"sep={adapted-shuffled:+.2f}, killer={shuffled-passive:+.2f}, "
                f"anti-cheat={adapted-zerov:+.2f}\n")
    print(f"\n[GATES] adp-pass={adapted-passive:+.2f}, adp-none={adapted-no_prior:+.2f}, "
          f"sep={adapted-shuffled:+.2f}, killer={shuffled-passive:+.2f}, "
          f"anti-cheat={adapted-zerov:+.2f}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--ffn_mult", type=float, default=1.0)
    main(ap.parse_args())
