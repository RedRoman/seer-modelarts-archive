"""Cache train-only, source-bound RGB corrections and their risk oracle.

One frozen restoration forward produces all target/donor frames. Frozen RAFT
estimates target-to-donor flow on these predictions, never on GT. Ground truth
enters only quadratic risk targets/oracles. No validation/test image is opened.
The oracle is a feasibility bound within this convex candidate family.
"""
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from data.planned_bvi import PlannedBVI, parse_sequence
from train_method_pilot import digest, state_digest, write_json
from eval_e4 import SPLITS, build_model


@torch.no_grad()
def donor_correction(base, raft, delta, flow_steps, flow_batch):
    """Sample a donor at target-grid + target->donor flow (pixel units)."""
    batch, channels, time, height, width = base.shape
    if batch != 1 or height % 8 or width % 8:
        raise ValueError("Cache expects one clip and RAFT-compatible dimensions")
    frames = base[0].permute(1, 0, 2, 3).contiguous()
    times = torch.arange(time, device=base.device)
    source_times = times + delta
    valid_times = (source_times >= 0) & (source_times < time)
    selected = times[valid_times]
    flow = torch.zeros(time, 2, height, width, device=base.device)
    for start in range(0, len(selected), flow_batch):
        target_index = selected[start:start + flow_batch]
        source_index = target_index + delta
        target = frames.index_select(0, target_index).float() * 2 - 1
        source = frames.index_select(0, source_index).float() * 2 - 1
        # RAFT's output is target-to-source displacement, exactly the flow
        # required to sample the source RGB on the target output grid.
        value = raft(target, source, num_flow_updates=flow_steps)[-1]
        flow.index_copy_(0, target_index, value)
    yy, xx = torch.meshgrid(torch.arange(height, device=base.device),
                            torch.arange(width, device=base.device), indexing="ij")
    x = xx[None] + flow[:, 0]
    y = yy[None] + flow[:, 1]
    valid = ((x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
             & valid_times[:, None, None])[:, None]
    grid = torch.stack(((x + .5) * (2 / width) - 1,
                        (y + .5) * (2 / height) - 1), dim=-1)
    donor = frames.index_select(0, source_times.clamp(0, time - 1))
    warped = F.grid_sample(donor.float(), grid.float(), mode="bilinear",
                           padding_mode="border", align_corners=False)
    correction = (warped - frames) * valid
    return (correction.permute(1, 0, 2, 3)[None].contiguous(),
            flow.permute(1, 0, 2, 3)[None].contiguous(),
            valid.permute(1, 0, 2, 3)[None].contiguous())


def patch_mean(value, patch):
    batch, channels, time, height, width = value.shape
    if height % patch or width % patch:
        raise ValueError("Patch size must divide image dimensions")
    flat = value.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
    return F.avg_pool2d(flat.mean(1, keepdim=True), patch, patch).reshape(
        batch, time, height // patch, width // patch)


def expand_patch(value, patch):
    return value.repeat_interleave(patch, -2).repeat_interleave(patch, -1)[:, None]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20261113)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--offset", type=int, default=2)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--flow-batch", type=int, default=2)
    parser.add_argument("--max-sequences", type=int, default=48)
    parser.add_argument("--device", choices=["npu", "cuda"], default="npu")
    args = parser.parse_args()
    if not (1 <= args.max_sequences <= 48 and 1 <= args.offset < 16
            and args.patch_size > 0 and 256 % args.patch_size == 0
            and args.flow_steps > 0 and args.flow_batch > 0):
        raise ValueError("Invalid cache/flow/patch budget")
    args.out.mkdir(parents=True, exist_ok=False)
    if args.device == "npu":
        import torch_npu  # noqa: F401
    from torchvision.models.optical_flow import raft_small
    from models.correction_risk import patch_moments, solve_two_candidate_risk
    torch.set_num_threads(4)
    torch.manual_seed(0)
    dev = torch.device(args.device + ":0")
    dataset = PlannedBVI(SPLITS, os.environ["SEER_ROOT"], 48, args.data_seed, 256, 16)
    rng = np.random.default_rng(args.data_seed)
    # Exactly one deterministic crop/temporal start per frozen training sequence.
    dataset.plan = [{"sequence_index": i, "augmentation_seed": int(rng.integers(0, 2**63 - 1))}
                    for i in range(args.max_sequences)]
    model, backbone, refiner, exposure, arm, clip_len, saved = build_model(args.init, dev)
    if refiner is not None or exposure is not None or arm != "pretrained" or clip_len != 16:
        raise ValueError("Cache requires an unmodified pretrained-encoder restoration model")
    model.eval()
    base_state = state_digest(model.state_dict())
    raft = raft_small(weights=None).eval()
    raft.load_state_dict(torch.load(args.raft_weights, map_location="cpu", weights_only=True), strict=True)
    raft.requires_grad_(False)
    raft = raft.to(dev)
    metadata = {"purpose": "TRAIN-only candidate-family oracle and selector cache; not learned performance",
                "domain": "bvi", "split_sha256": digest(SPLITS),
                "base_checkpoint_sha256": digest(args.init), "base_saved_args": saved,
                "base_reconstruction": model.eval_reconstruction,
                "base_state_sha256": base_state, "raft_weights_sha256": digest(args.raft_weights),
                "flow_inputs": "clamped restored RGB only, target-to-source direction",
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "plan": dataset.plan, "source_sha256": {
                    str(path.relative_to(CODE)): digest(path) for path in (
                        Path(__file__), CODE / "models/correction_risk.py", CODE / "data/planned_bvi.py",
                        CODE / "tools/eval_e4.py", CODE / "models/pixel_branch_v2.py")},
                "historical_training_split": "Checkpoint args retained; no historical split hash inferred from counts"}
    write_json(args.out / "provenance.json", metadata)
    rows = []
    for i in range(len(dataset)):
        sample = dataset[i]
        low, gt = sample["lq"][None].to(dev), sample["gt"][None].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            tokens = [value.float() for value in backbone(low)["feats"]]
            base = model(low, tokens)["pred_pixels"].float().clamp(0, 1)
        corrections, flows, masks = zip(*(donor_correction(base, raft, delta, args.flow_steps,
                                                           args.flow_batch)
                                         for delta in (-args.offset, args.offset)))
        error = gt - base
        a, gram = patch_moments(error, torch.stack(corrections, dim=1), patch=args.patch_size)
        weights, risk = solve_two_candidate_risk(a, gram)
        combined = base + sum(expand_patch(weights[..., j], args.patch_size) * corrections[j]
                              for j in range(2))
        e0 = patch_mean(error.square(), args.patch_size)
        oracle = patch_mean((gt - combined).square(), args.patch_size)
        torch.testing.assert_close(oracle, e0 + risk, atol=5e-7, rtol=3e-4)
        if (oracle - e0).max() > 5e-7:
            raise AssertionError("Oracle candidate set must preserve the zero-correction option")
        singles = [patch_mean((gt - base - d).square(), args.patch_size) for d in corrections]
        fixed_average = patch_mean((gt - base - .5 * (corrections[0] + corrections[1])).square(),
                                   args.patch_size)
        scene, condition = parse_sequence(sample["name"].split("/", 1)[1])
        tensors = {"lq": low.cpu(), "gt": gt.cpu(), "base": base.cpu(),
                   "corrections": torch.stack(corrections, dim=1).cpu(),
                   "flow": torch.stack(flows, dim=1).cpu(),
                   "valid_mask": torch.stack(masks, dim=1).cpu(),
                   "a": a.cpu(), "gram": gram.cpu(), "oracle_weights": weights.cpu()}
        path = args.out / f"clip_{i:03d}.pt"
        torch.save({"name": sample["name"], "scene": scene, "condition": condition,
                    "plan": dataset.plan[i], "tensors": tensors}, path)
        row = {"name": sample["name"], "scene": scene, "condition": condition,
               "base_mse": float(e0.mean()), "past_mse": float(singles[0].mean()),
               "future_mse": float(singles[1].mean()), "fixed_average_mse": float(fixed_average.mean()),
               "oracle_mse": float(oracle.mean()),
               "positive_gain_patch_fraction": float((oracle < e0 - 1e-8).float().mean()),
               "weights_mean": weights.mean((0, 1, 2, 3)).cpu().tolist(),
               "cache_path": path.name, "cache_sha256": digest(path)}
        if not all(np.isfinite(row[k]) for k in ("base_mse", "past_mse", "future_mse",
                                                  "fixed_average_mse", "oracle_mse")):
            raise FloatingPointError(sample["name"])
        rows.append(row)
        write_json(args.out / "progress.json", {"n": len(rows), "per_sequence": rows})
        print(json.dumps(row), flush=True)
    if state_digest(model.state_dict()) != base_state:
        raise AssertionError("Caching changed the fixed restoration model")
    report = {"purpose": metadata["purpose"], "n": len(rows), "per_sequence": rows,
              "base_state_unchanged": True,
              "mean": {key: float(np.mean([row[key] for row in rows])) for key in (
                  "base_mse", "past_mse", "future_mse", "fixed_average_mse", "oracle_mse")}}
    write_json(args.out / "oracle_summary.json", report)
    print(json.dumps(report["mean"]), flush=True)


if __name__ == "__main__":
    main()
