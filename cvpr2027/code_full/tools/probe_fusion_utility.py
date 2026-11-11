"""Training-only feasibility screen for predicting useful fusion corrections.

Compare the same fixed restorer with its fusion message withheld, with its
learned gate, and with that gate forced open. Report a GT-assisted patch oracle
and exact quadratic correction moments. These are training upper bounds, never
inference metrics or evidence of a learned selector's performance.
"""
from pathlib import Path
import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F

from eval_method_pilot import load_pilot
from train_method_pilot import PlannedSDSD, digest, state_digest, write_json
from eval_e4 import SPLITS


def patch_mean(tensor, size):
    # B,C,T,H,W -> B,T,1,H/size,W/size, averaging RGB and spatial pixels.
    batch, channels, time, height, width = tensor.shape
    if height % size or width % size:
        raise ValueError("Patch size must divide the output dimensions")
    flat = tensor.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
    pooled = F.avg_pool2d(flat.mean(1, keepdim=True), size, size)
    return pooled.reshape(batch, time, 1, height // size, width // size)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=20261112)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "npu"], default="npu")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.samples < 1 or args.patch_size < 1:
        raise ValueError("Positive samples and patch size required")
    if args.device == "npu":
        import torch_npu  # noqa: F401
    torch.set_num_threads(4)
    dev = torch.device(args.device + ":0")
    model, backbone, saved = load_pilot(args.checkpoint, dev)
    if saved["args"]["variant"] == "baseline":
        raise ValueError("A trained fusion checkpoint is required")
    if digest(SPLITS) != saved["provenance"]["split_sha256"]:
        raise ValueError("Current split differs from checkpoint provenance")
    root = Path(os.environ["SEER_ROOT"])
    domain = saved["args"].get("domain", "sdsd")
    if domain == "bvi":
        from data.planned_bvi import PlannedBVI
        dataset_class = PlannedBVI
    elif domain == "sdsd":
        dataset_class = PlannedSDSD
    else:
        raise ValueError(domain)
    data = dataset_class(SPLITS, root, args.samples, args.data_seed, 256, 16)
    original_state = state_digest(model.state_dict())
    rows = []
    for index in range(len(data)):
        sample = data[index]
        low = sample["lq"][None].to(dev)
        gt = sample["gt"][None].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            tokens = [value.float() for value in backbone(low)["feats"]]
            off = model(low, tokens, fusion_bypass=True)["pred_pixels"].float()
            gated = model(low, tokens)["pred_pixels"].float()
            # A hook changes only this forward's logits; checkpoint state is
            # unchanged. sigmoid(30) is exactly one in the deployed precision.
            handle = model.temporal_fusion.confidence.register_forward_hook(
                lambda module, inputs, output: torch.full_like(output, 30.))
            try:
                opened = model(low, tokens)["pred_pixels"].float()
            finally:
                handle.remove()
        error = gt - off
        correction = opened - off
        e0 = patch_mean(error.square(), args.patch_size)
        e1 = patch_mean((gt - opened).square(), args.patch_size)
        eg = patch_mean((gt - gated).square(), args.patch_size)
        a = patch_mean(error * correction, args.patch_size)
        gram = patch_mean(correction.square(), args.patch_size)
        # alpha in [0,1] minimizes exact patch MSE along the fixed correction.
        alpha = (a / gram.clamp_min(1e-12)).clamp(0, 1)
        oracle = e0 - 2 * alpha * a + alpha.square() * gram
        binary_oracle = torch.minimum(e0, e1)
        torch.testing.assert_close(e1, e0 - 2 * a + gram, atol=2e-7, rtol=2e-4)
        if not all(torch.isfinite(value).all() for value in (e0, e1, eg, a, gram, oracle)):
            raise FloatingPointError(f"Nonfinite utility probe at {sample['name']}")
        if (oracle - e0).max() > 2e-7 or (oracle - binary_oracle).max() > 2e-7:
            raise AssertionError("Bounded correction oracle must include both endpoints")
        rows.append({"name": sample["name"], "sample_index": index,
                     "off_mse": float(e0.mean()), "gated_mse": float(eg.mean()),
                     "open_mse": float(e1.mean()),
                     "binary_oracle_mse": float(binary_oracle.mean()),
                     "bounded_oracle_mse": float(oracle.mean()),
                     "helpful_patch_fraction": float((e1 < e0 - 1e-8).float().mean()),
                     "alpha_mean": float(alpha.mean()),
                     "correction_energy": float(gram.mean())})
        print(json.dumps(rows[-1]), flush=True)
    if state_digest(model.state_dict()) != original_state:
        raise AssertionError("Probe altered model state")
    report = {"purpose": "train-only GT-assisted upper bound; not learned/inference performance",
              "action": "intervention on the new fusion message, not all temporal information",
              "checkpoint_sha256": digest(args.checkpoint), "probe_source_sha256": digest(__file__),
              "split_sha256": digest(SPLITS), "domain": domain,
              "data_seed": args.data_seed, "plan": data.plan, "patch_size": args.patch_size,
              "model_state_unchanged": True, "n": len(rows), "per_sample": rows,
              "mean": {key: float(np.mean([row[key] for row in rows]))
                       for key in rows[0] if key not in ("name", "sample_index")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps(report["mean"]), flush=True)


if __name__ == "__main__":
    main()
