"""C-2-Direct Stage 1 with iREPA conv projector adapter (E1)."""
from __future__ import annotations
import argparse, csv, os, sys, time

import numpy as np
import torch
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from models.v11.content_attn_adapter_v3 import ContentAttnAdapterIREPA
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


CONDITIONS = ["adapted_zL", "passive_zL", "no_prior"]
PROBS = [0.50, 0.25, 0.25]


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    adapter = ContentAttnAdapterIREPA(
        bottleneck_channels=decoder.dims[-1],
        vjepa_grid=tuple(decoder.vjepa_grid),
        vjepa_dim=decoder.vjepa_dim,
        active_scales=(3, 4),
        ref_layer=3,
        attn_heads=8,
        ffn_mult=1.0,
    ).to(device)
    print(f"[adapter-iREPA] {adapter.num_trainable_params()/1e6:.2f}M (3x3 conv projector + spatial norm)")

    optim = torch.optim.AdamW(list(decoder.parameters()) + list(adapter.parameters()),
                              lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)
    train_seqs, _ = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    log = []
    cond_running = {c: [] for c in CONDITIONS}
    t0 = time.time()
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)
        with torch.no_grad():
            z_L = encode(backbone, lq)
        h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
        cond = str(rng.choice(CONDITIONS, p=PROBS))
        if cond == "no_prior":
            kv_layers, kv_override = None, None
        elif cond == "passive_zL":
            kv_layers, kv_override = z_L, None
        else:
            kv_layers, kv_override = None, adapter(z_L, h_b)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=kv_layers,
                                  delta_layers=None, kv_override=kv_override)
        L = charbonnier(pred.float(), gt)
        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + list(adapter.parameters()), 1.0)
        optim.step(); sched.step()
        cond_running[cond].append(L.item())
        if (it + 1) % args.log_every == 0:
            running = {c: float(np.mean(cond_running[c][-50:])) if cond_running[c] else float('nan') for c in CONDITIONS}
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} cond={cond} "
                  f"running[adp={running['adapted_zL']:.4f} pass={running['passive_zL']:.4f} "
                  f"none={running['no_prior']:.4f}] dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item(), "cond": cond,
                        **{f"running_{c}": running[c] for c in CONDITIONS}})
            t0 = time.time()
    torch.save({"decoder": decoder.state_dict(), "adapter": adapter.state_dict(),
                "args": vars(args), "log": log, "iter": args.iters,
                "stage": 1, "variant": "irepa"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
