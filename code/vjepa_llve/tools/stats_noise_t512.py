"""B (BJ1 close-out): stats_noise re-run under tile=512 protocol.

Original v9.4 stats_noise was on tile=256 (broken protocol). Re-run under tile=512
to get clean answer to "is V-JEPA content load-bearing or just stats?".

Modes (matches v9.4 protocol but at tile=512):
  full_vjepa
  cross_attn_off
  zero_vjepa
  stats_noise_vjepa  (matched mean+std Gaussian noise)
"""
import os, sys, csv, time
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset
from tools.audit_llvestcd import build_v9_model, rgb_psnr_uint8, load_seq_full, make_taper_window
from tools.prior_ablation_v9_4 import (
    install_cross_attn_off, uninstall_hooks, manipulate_features,
    tiled_inference_with_intervention,
)


def list_seqs(root, split="test"):
    ds = SDSDDataset(root=root, subset="direct", split=split, clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


def to_u8(x): return (x[0].permute(1,2,3,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
def mean_psnr(p, g): return float(np.mean([rgb_psnr_uint8(p[t], g[t]) for t in range(p.shape[0])]))


def main():
    out_dir = "/cache/zihaohe/checkpoint/stats_noise_t512"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")
    seqs = list_seqs("/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor", "test")
    print(f"[B] {len(seqs)} indoor test seqs (our 14)")

    model = build_v9_model("/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml",
                           "/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt", device)

    modes = ["full_vjepa", "cross_attn_off", "zero_vjepa", "stats_noise_vjepa"]
    results = {}
    for mode in modes:
        print(f"\n=== {mode} (tile=512) ===")
        handles = install_cross_attn_off(model) if mode == "cross_attn_off" else []
        try:
            for seq_name, lq_paths, gt_paths in seqs:
                t0 = time.time()
                lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=16)
                lq, gt = lq.to(device), gt.to(device)
                pred = tiled_inference_with_intervention(model, lq, mode=mode,
                                                          tile=512, overlap=128)
                psnr = mean_psnr(to_u8(pred), to_u8(gt))
                results.setdefault(seq_name, {})[mode] = psnr
                print(f"  [{mode}/{seq_name}] PSNR={psnr:.3f} ({time.time()-t0:.1f}s)")
        finally:
            uninstall_hooks(handles)

    # Aggregate
    print(f"\n=== AGGREGATE (tile=512) ===")
    agg = {m: float(np.mean([results[s][m] for s in results if m in results[s]])) for m in modes}
    base = agg["full_vjepa"]
    for m in modes:
        d = agg[m] - base
        print(f"  {m:25s}: PSNR={agg[m]:.3f}  Δ={d:+.3f}")

    # Compare to BJ1's broken-protocol (v9.4 tile=256)
    print(f"\n=== COMPARISON to v9.4 tile=256 (broken protocol) ===")
    print(f"  ckpt v9.0b_mix (this run, NOT v9.0_best from v9.4)")
    print(f"  Note: v9.4 was on v9.0_best; this is on v9.0b_mix (now project best)")

    # CSV
    csv_path = os.path.join(out_dir, "per_seq.csv")
    with open(csv_path, "w") as f:
        keys = ["seq"] + modes + [f"delta_{m}" for m in modes if m != "full_vjepa"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for seq in sorted(results):
            d = results[seq]
            row = {"seq": seq}
            for m in modes: row[m] = d.get(m, 0)
            for m in modes:
                if m != "full_vjepa":
                    row[f"delta_{m}"] = d.get(m, 0) - d.get("full_vjepa", 0)
            w.writerow(row)
    print(f"\nwrote {csv_path}")

    # Verdict
    d_off = agg["cross_attn_off"] - base
    d_zero = agg["zero_vjepa"] - base
    d_noise = agg["stats_noise_vjepa"] - base
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# B: stats_noise re-run under tile=512 (closes BJ1)\n\n")
        f.write(f"Ckpt: v9.0b_mix_best  Protocol: tile=512/overlap=128  14 indoor test seqs\n\n")
        f.write("| mode | mean PSNR | Δ vs full | interpretation |\n|---|---:|---:|---|\n")
        f.write(f"| full_vjepa | {agg['full_vjepa']:.3f} | — | baseline |\n")
        f.write(f"| cross_attn_off | {agg['cross_attn_off']:.3f} | {d_off:+.3f} | bypass cross-attn entirely |\n")
        f.write(f"| zero_vjepa | {agg['zero_vjepa']:.3f} | {d_zero:+.3f} | feed zeros (only kv.bias) |\n")
        f.write(f"| stats_noise_vjepa | {agg['stats_noise_vjepa']:.3f} | {d_noise:+.3f} | matched mean+std Gaussian noise |\n")
        f.write("\n## Verdict (closes BJ1)\n\n")
        if abs(d_noise) > 0.5:
            v = ("Stats-noise differs meaningfully from full → V-JEPA CONTENT is load-bearing "
                 f"(not just stats). Δ_noise = {d_noise:+.3f} dB.")
        elif abs(d_noise) < 0.2:
            v = ("Stats-noise ≈ full → V-JEPA effectively used only via stats, not content. "
                 f"Δ_noise = {d_noise:+.3f} dB. v9.0b_mix could likely be replaced with random "
                 "noise at correct stats and lose <0.2 dB.")
        else:
            v = (f"Mixed. Δ_noise = {d_noise:+.3f} dB. Marginal stats-vs-content evidence.")
        f.write(f"**{v}**\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
