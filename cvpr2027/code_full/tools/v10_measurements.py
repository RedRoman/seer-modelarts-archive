"""A. v10 design spec measurements (BU4 mandatory).

Measure (not guess):
  1. Exact v9.0b_mix param count (total + trainable + per-component)
  2. Memory profile bf16 batch=1 T=16 crop={256, 384, 512}
  3. Throughput estimate: forward+backward time per crop size

Output: /cache/zihaohe/checkpoint/v10_measurements/measurements.md
"""
import os, sys, time, json
import torch
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from tools.audit_llvestcd import build_v9_model

OUT_DIR = "/cache/zihaohe/checkpoint/v10_measurements"
CFG = "/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml"
CKPT = "/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt"


def count_params(m):
    """Return (total, trainable) param counts."""
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def per_component_counts(model):
    """Hierarchy: top-level submodules in v9 model."""
    rows = []
    for name, child in model.named_children():
        t, tr = count_params(child)
        rows.append({"name": name, "total": t, "trainable": tr})
    # Also pixel_branch internals
    if hasattr(model, "pixel_branch"):
        for name, child in model.pixel_branch.named_children():
            t, tr = count_params(child)
            rows.append({"name": f"pixel_branch.{name}", "total": t, "trainable": tr})
    return rows


@torch.no_grad()
def memory_profile_inference(model, device, crop, T=16, batch=1, autocast_dtype=torch.bfloat16):
    """Measure memory usage for inference at given crop size."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand(batch, 3, T, crop, crop, device=device)
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        out = model.sample(x)
    torch.cuda.synchronize()
    t = time.time() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    output_shape = out["pred_pixels"].shape
    del x, out
    torch.cuda.empty_cache()
    return {"crop": crop, "T": T, "batch": batch,
            "vram_peak_gb": vram_peak, "fwd_s": t,
            "output_shape": list(output_shape)}


def memory_profile_training(model, device, crop, T=16, batch=1, autocast_dtype=torch.bfloat16):
    """Measure memory usage for forward + backward at given crop size.
    Training would also include optimizer state (Adam: 2x weights for momentum/variance).
    Approximated as: act_mem + 4× param mem (weights + grads + 2 optim states)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand(batch, 3, T, crop, crop, device=device, requires_grad=False)
    gt = torch.rand(batch, 3, T, crop, crop, device=device)
    # Enable gradients on trainable params
    model.train()
    for p in model.parameters():
        if p.requires_grad: p.grad = None
    t0 = time.time()
    try:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model(x)
            loss = ((out["pred_pixels"] - gt) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t = time.time() - t0
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        ok = True
        err = None
    except Exception as e:
        t = time.time() - t0
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        ok = False
        err = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        model.eval()
        del x, gt
        torch.cuda.empty_cache()
    return {"crop": crop, "T": T, "batch": batch,
            "vram_peak_gb": vram_peak, "fwd_bwd_s": t,
            "ok": ok, "err": err}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda")
    print(f"[A] device: {device}")

    model = build_v9_model(CFG, CKPT, device)
    model.eval()

    # 1. Param counts
    total, trainable = count_params(model)
    print(f"\n=== v9.0b_mix param count ===")
    print(f"  total:     {total:,} = {total/1e6:.2f} M")
    print(f"  trainable: {trainable:,} = {trainable/1e6:.2f} M")

    components = per_component_counts(model)
    print(f"\n  per-component (top-level + pixel_branch internals):")
    for c in components:
        print(f"    {c['name']:30s}  total={c['total']:>12,} ({c['total']/1e6:>7.2f} M)  "
              f"trainable={c['trainable']:>12,}")

    # 2. Inference memory profile
    print(f"\n=== inference memory profile (bf16, batch=1, T=16) ===")
    inf_results = []
    for crop in [256, 384, 512]:
        try:
            r = memory_profile_inference(model, device, crop)
            inf_results.append(r)
            print(f"  crop={crop}: vram_peak={r['vram_peak_gb']:.2f} GB  "
                  f"fwd={r['fwd_s']*1000:.0f} ms  output={r['output_shape']}")
        except Exception as e:
            print(f"  crop={crop}: FAILED {type(e).__name__}: {str(e)[:200]}")
            inf_results.append({"crop": crop, "err": str(e)[:200]})

    # 3. Training memory profile (forward + backward only, no optimizer)
    print(f"\n=== training fwd+bwd memory profile (bf16, batch=1, T=16) ===")
    train_results = []
    for crop in [256, 384, 512]:
        r = memory_profile_training(model, device, crop)
        train_results.append(r)
        if r['ok']:
            print(f"  crop={crop}: vram_peak={r['vram_peak_gb']:.2f} GB  "
                  f"fwd+bwd={r['fwd_bwd_s']*1000:.0f} ms")
        else:
            print(f"  crop={crop}: FAILED  vram_at_fail={r['vram_peak_gb']:.2f} GB  err={r['err']}")

    # 4. Throughput estimate: train iter/sec → time for 5000 iters
    print(f"\n=== training throughput estimate (bf16, batch=1, single A100-80GB) ===")
    print(f"  v10 budget: 3000-5000 iter (per BP5)")
    for r in train_results:
        if r['ok']:
            iter_per_sec = 1.0 / r['fwd_bwd_s']
            for n in [3000, 5000]:
                est_h = n / iter_per_sec / 3600
                print(f"  crop={r['crop']}: {iter_per_sec:.2f} iter/s → {n} iters = {est_h:.1f} h")
        else:
            print(f"  crop={r['crop']}: NOT FEASIBLE (OOM)")

    # Save
    md_path = os.path.join(OUT_DIR, "measurements.md")
    with open(md_path, "w") as f:
        f.write("# v10 design spec measurements (BU4 mandatory)\n\n")
        f.write(f"Ckpt: v9.0b_mix_best  Hardware: single A100-80GB  Precision: bf16\n\n")
        f.write(f"## 1. Param count (v9.0b_mix architecture)\n\n")
        f.write(f"- total:     {total:,} = {total/1e6:.2f} M\n")
        f.write(f"- trainable: {trainable:,} = {trainable/1e6:.2f} M\n\n")
        f.write(f"### per-component\n\n| name | total params | M | trainable |\n|---|---:|---:|---:|\n")
        for c in components:
            f.write(f"| {c['name']} | {c['total']:,} | {c['total']/1e6:.2f} | {c['trainable']:,} |\n")
        f.write(f"\n## 2. Inference memory + latency (bf16, batch=1, T=16)\n\n")
        f.write("| crop | vram peak (GB) | fwd (ms) | output shape |\n|---:|---:|---:|---|\n")
        for r in inf_results:
            if "err" in r:
                f.write(f"| {r['crop']} | OOM | — | err: {r['err']} |\n")
            else:
                f.write(f"| {r['crop']} | {r['vram_peak_gb']:.2f} | {r['fwd_s']*1000:.0f} | {r['output_shape']} |\n")
        f.write(f"\n## 3. Training fwd+bwd memory + latency (bf16, batch=1, T=16)\n\n")
        f.write("| crop | vram peak (GB) | fwd+bwd (ms) | feasible? |\n|---:|---:|---:|---|\n")
        for r in train_results:
            if r['ok']:
                f.write(f"| {r['crop']} | {r['vram_peak_gb']:.2f} | {r['fwd_bwd_s']*1000:.0f} | yes |\n")
            else:
                f.write(f"| {r['crop']} | {r['vram_peak_gb']:.2f} | — | no — {r['err']} |\n")
        f.write(f"\n## 4. v10 training budget extrapolation\n\n")
        f.write("Per BP5 spec: 3000-5000 iter at lr=1e-5/5e-6, init from v9.0b_mix.\n\n")
        f.write("| crop | iter/sec | 3000 iter time | 5000 iter time |\n|---:|---:|---:|---:|\n")
        for r in train_results:
            if r['ok']:
                ips = 1.0 / r['fwd_bwd_s']
                f.write(f"| {r['crop']} | {ips:.2f} | {3000/ips/3600:.1f} h | {5000/ips/3600:.1f} h |\n")
            else:
                f.write(f"| {r['crop']} | — | OOM | OOM |\n")
        f.write(f"\nNote: actual training will be slower than fwd+bwd alone due to data loading,\n")
        f.write(f"optimizer step, gradient sync, periodic checkpointing/eval. Multiply estimates by 1.5-2×.\n")
        f.write(f"With batch=1 the throughput is bottleneck-limited; gradient accumulation can help effective batch.\n")
    print(f"\nwrote {md_path}")


if __name__ == "__main__":
    main()
