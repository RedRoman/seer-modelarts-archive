"""Fixed-budget learnability probe on TRAIN-only cached BVI corrections.

The 18/6 scene split is INTERNAL to the base model's 24 training scenes. The
base may have seen every scene: this is not final validation or test evidence.
Three identical tiny networks use identical observable inputs and batch plans.
No checkpoint selection, intermediate holdout evaluation or base-model calls.
Holdout MSE covers every RGB pixel in the cached clips, not original full videos.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
SPLIT_SEED = 20261114
MODES = ("risk_full", "direct_mse", "direct_mse_normalized")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def local_module(name, path):
    # Avoid importing models/__init__.py and its unrelated foundation models.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_scene_split(scene_names):
    """Fixed 24 TRAIN scenes -> 18 fit / 6 internal holdout, before outcomes."""
    scenes = sorted(set(scene_names))
    if len(scenes) != 24:
        raise ValueError(f"Expected exactly 24 base-training scenes, found {len(scenes)}")
    order = np.random.default_rng(SPLIT_SEED).permutation(len(scenes))
    return {"seed": SPLIT_SEED, "fit": [scenes[int(i)] for i in order[:18]],
            "internal_holdout": [scenes[int(i)] for i in order[18:]],
            "scope": "Selector internal holdout within base TRAIN scenes; NOT final VAL/test"}


def make_batch_plan(fit_frame_indices, steps, batch_frames, seed):
    indices = np.asarray(fit_frame_indices, dtype=np.int64)
    if (indices.ndim != 1 or not len(indices) or len(set(indices.tolist())) != len(indices)
            or min(steps, batch_frames) < 1 or np.any(indices < 0)):
        raise ValueError("Need distinct nonnegative fit-frame indices and positive budgets")
    # Uniform frame sampling WITH replacement, generated once before models.
    return np.random.default_rng(seed).choice(indices, size=(steps, batch_frames), replace=True)


def fit_normalization(features, a, gram, fit_indices, *, e0):
    """NumPy fit-only population feature moments and uncentered target RMS.

    features [F,C,h,w], a [F,h,w,2], gram [F,h,w,2,2], e0 [F,h,w]. Holdout rows are never
    indexed here. Target centering is intentionally absent to preserve w0.
    """
    indices = np.asarray(fit_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(set(indices.tolist())) != len(indices):
        raise ValueError("Need distinct fit-frame indices")
    if np.any(indices < 0) or np.any(indices >= len(features)):
        raise ValueError("Fit frame index is out of range")
    sums = np.zeros(features.shape[1], np.float64)
    square_sums, count, energy_sum, energy_count, e0_sum = sums.copy(), 0, 0., 0, 0.
    for index in indices:
        value = np.asarray(features[index], dtype=np.float64)
        sums += value.sum(axis=(1, 2))
        square_sums += np.square(value).sum(axis=(1, 2))
        count += value.shape[1] * value.shape[2]
        e0_sum += np.asarray(e0[index], dtype=np.float64).sum()
        g = np.asarray(gram[index], dtype=np.float64)
        energy_sum += g[..., 0, 0].sum() + g[..., 1, 1].sum()
        energy_count += 2 * g.shape[0] * g.shape[1]
    mean = sums / count
    std = np.sqrt(np.maximum(square_sums / count - mean * mean, 0)).clip(1e-6)
    energy_floor = max(1e-6, 1e-3 * np.sqrt(max(energy_sum / energy_count, 0)))
    target_squares, target_count = np.zeros(2, np.float64), 0
    for index in indices:
        g = np.asarray(gram[index], dtype=np.float64)
        denominator = np.sqrt(np.maximum(np.stack((g[..., 0, 0], g[..., 1, 1]), -1), 0)).clip(energy_floor)
        target = (np.asarray(a[index], dtype=np.float64) - g.sum(axis=-1) / 3) / denominator
        target_squares += np.square(target).sum(axis=(0, 1))
        target_count += target.shape[0] * target.shape[1]
    target_scale = np.sqrt(target_squares / target_count).clip(1e-6)
    direct_mse_normalizer = float(e0_sum / count)
    if not np.isfinite(direct_mse_normalizer) or direct_mse_normalizer <= 0:
        raise ValueError("Fit-only mean e0 must be finite and positive")
    result = {"feature_mean": mean.tolist(), "feature_std": std.tolist(),
              "energy_floor": float(energy_floor), "target_scale": target_scale.tolist(),
              "direct_mse_normalizer": direct_mse_normalizer,
              "source": "Fit scenes only; population feature std; uncentered target RMS; mean base MSE",
              "fit_frames": len(indices), "fit_patches": count}
    if not all(np.isfinite(result[k]).all() for k in ("feature_mean", "feature_std", "target_scale")):
        raise ValueError("Fit normalization contains nonfinite values")
    return result


def cache_manifest(directory):
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    summary = json.loads((directory / "oracle_summary.json").read_text())
    if (provenance.get("domain") != "bvi" or not str(provenance.get("purpose", "")).startswith("TRAIN-only")
            or summary.get("n") != 48 or len(provenance.get("plan", [])) != 48
            or summary.get("base_state_unchanged") is not True):
        raise ValueError("A completed TRAIN-only BVI48 oracle cache is required")
    for key in ("base_checkpoint_sha256", "base_state_sha256", "raft_weights_sha256", "split_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key, ""))) is None:
            raise ValueError(f"Cache provenance lacks a valid {key}")
    files = sorted(directory.glob("clip_*.pt"))
    expected = [f"clip_{i:03d}.pt" for i in range(48)]
    rows = summary.get("per_sequence", [])
    if [path.name for path in files] != expected or len(rows) != 48:
        raise ValueError("Expected exactly the 48 completed cache clips")
    if [row.get("cache_path") for row in rows] != expected:
        raise ValueError("Cache summary order/path mismatch")
    if sorted(item.get("sequence_index") for item in provenance["plan"]) != list(range(48)):
        raise ValueError("Cache plan must contain each training sequence exactly once")
    for path, row in zip(files, rows):
        if digest(path) != row.get("cache_sha256"):
            raise ValueError(f"Cache SHA256 mismatch: {path.name}")
    return files, provenance, rows


def prepare_cache(directory, selector, risk):
    """Read raw clips one at a time; retain only patch inputs and loss targets."""
    import torch
    files, provenance, summary_rows = cache_manifest(directory)
    patch = provenance["args"]["patch_size"]
    all_x, all_a, all_g, all_e0, frames, clips = [], [], [], [], [], []
    shape = None
    for index, (path, row) in enumerate(zip(files, summary_rows)):
        sample = torch.load(path, map_location="cpu", weights_only=True)
        match = re.fullmatch(r"BVI/([^/\\]+)__(?:(10|20)fps|fps(10|20))", sample["name"])
        if match is None or (sample["scene"], sample["condition"]) != (match[1], match[2] or match[3]):
            raise ValueError("Cache scene/condition identifier mismatch")
        if (sample["name"] != row["name"] or sample["scene"] != row["scene"]
                or sample["condition"] != row["condition"] or sample["plan"] != provenance["plan"][index]):
            raise ValueError("Clip disagrees with hashed manifest or provenance plan")
        tensors = sample["tensors"]
        low, base, directions, valid = (tensors[key] for key in ("lq", "base", "corrections", "valid_mask"))
        if low.shape[0] != 1 or (shape is not None and low.shape != shape):
            raise ValueError("All cache clips must have one batch item and the same shape")
        shape = low.shape
        if any(value.requires_grad for value in (low, base, directions, valid)):
            raise ValueError("Cache must contain detached frozen-model outputs")
        # This call receives no GT/a/oracle_weights; keep input construction explicit.
        x, gram = selector.observable_patch_features(low, base, directions, valid, patch)
        gt = tensors["gt"]
        if gt.shape != base.shape or gt.dtype != torch.float32 or not bool(torch.isfinite(gt).all()):
            raise ValueError("Invalid cached GT loss target")
        if bool(((gt < 0) | (gt > 1)).any()) or bool(((base < 0) | (base > 1)).any()):
            raise ValueError("Cache GT and base must use unrounded RGB [0,1]")
        a, gt_gram = risk.patch_moments(gt - base, directions, patch)
        torch.testing.assert_close(gram, gt_gram, atol=1e-8, rtol=1e-5)
        torch.testing.assert_close(a, tensors["a"], atol=1e-8, rtol=1e-5)
        torch.testing.assert_close(gram, tensors["gram"], atol=1e-8, rtol=1e-5)
        e0 = selector._patch_pool((gt - base).square(), patch).mean(2)
        all_x.append(x[0]); all_a.append(a[0]); all_g.append(gram[0]); all_e0.append(e0[0])
        clips.append({"name": sample["name"], "scene": sample["scene"],
                      "condition": sample["condition"], "path": path.name, "sha256": row["cache_sha256"]})
        for frame in range(low.shape[2]):
            frames.append({"clip_index": index, "frame": frame, "scene": sample["scene"],
                           "condition": sample["condition"], "name": sample["name"]})
    if len({row["name"] for row in clips}) != 48:
        raise ValueError("Duplicate cached sequence name")
    pairs = Counter((row["scene"], row["condition"]) for row in clips)
    scenes = {row["scene"] for row in clips}
    if len(scenes) != 24 or any(pairs[(scene, condition)] != 1 for scene in scenes for condition in ("10", "20")):
        raise ValueError("Require one condition10 and one condition20 clip for each of 24 scenes")
    split = make_scene_split(scenes)
    fit = [i for i, row in enumerate(frames) if row["scene"] in set(split["fit"])]
    holdout = [i for i, row in enumerate(frames) if row["scene"] in set(split["internal_holdout"])]
    return {"features": torch.cat(all_x), "a": torch.cat(all_a), "gram": torch.cat(all_g),
            "e0": torch.cat(all_e0), "frames": frames, "clips": clips, "split": split,
            "fit_indices": fit, "holdout_indices": holdout, "patch_size": patch,
            "cache_provenance": provenance}


def state_digest(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(key.encode()); result.update(str((array.shape, str(array.dtype))).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def final_holdout_evaluation(data, states, normalization, device, batch_frames, selector, risk):
    """Exactly one final evaluation pass; all cached holdout frames and patches."""
    import torch
    mean = torch.tensor(normalization["feature_mean"], device=device)[None, :, None, None]
    std = torch.tensor(normalization["feature_std"], device=device)[None, :, None, None]
    scale = torch.tensor(normalization["target_scale"], device=device)
    models = {}
    for mode, state in states.items():
        model = selector.TinyCorrectionSelector().to(device)
        model.load_state_dict(state, strict=True)
        models[mode] = model.eval()
    names = ["base", "past", "future", "fixed_average", "initial_thirds", *MODES, "gt_oracle"]
    accumulated = {name: {} for name in names}
    with torch.no_grad():
        indices = data["holdout_indices"]
        for start in range(0, len(indices), batch_frames):
            ix = indices[start:start + batch_frames]
            x, a, gram, e0 = (data[key][ix].to(device) for key in ("features", "a", "gram", "e0"))
            zero = torch.zeros_like(a)
            weights = {"base": zero, "past": zero + zero.new_tensor([1., 0.]),
                       "future": zero + zero.new_tensor([0., 1.]), "fixed_average": zero + .5,
                       "initial_thirds": zero + 1 / 3}
            for mode, model in models.items():
                head = model((x - mean) / std)
                weights[mode] = selector.weights_from_head(head, gram, mode, scale, normalization["energy_floor"])
            weights["gt_oracle"], _ = risk.solve_two_candidate_risk(a, gram)
            for name, weight in weights.items():
                mse = selector.quadratic_mse(e0, a, gram, weight).double().cpu().numpy()
                if not np.isfinite(mse).all() or mse.min() < -1e-6:
                    raise FloatingPointError(f"Invalid final MSE: {name}")
                for local, global_index in enumerate(ix):
                    clip = data["frames"][global_index]["clip_index"]
                    entry = accumulated[name].setdefault(clip, {"patch_mse_sum": 0., "patches": 0})
                    entry["patch_mse_sum"] += float(mse[local].sum())
                    entry["patches"] += int(mse[local].size)
    report = {}
    rgb_elements_per_patch = 3 * data["patch_size"] ** 2
    for name, rows in accumulated.items():
        per_sequence, per_scene = [], {}
        for clip, totals in sorted(rows.items()):
            meta = data["clips"][clip]
            value = totals["patch_mse_sum"] / totals["patches"]
            per_sequence.append({**meta, "mse": value, "patches": totals["patches"]})
            per_scene.setdefault(meta["scene"], []).append(value)
        scene_means = {scene: float(np.mean(values)) for scene, values in sorted(per_scene.items())}
        patches = sum(row["patches"] for row in rows.values())
        summed = sum(row["patch_mse_sum"] for row in rows.values())
        report[name] = {"rgb_mse": summed / patches, "scene_mean_mse": float(np.mean(list(scene_means.values()))),
                        "rgb_squared_error_sum": summed * rgb_elements_per_patch,
                        "rgb_elements": patches * rgb_elements_per_patch,
                        "per_scene": scene_means, "per_sequence": per_sequence}
    return {"scope": data["split"]["scope"], "evaluation": "Once after all fixed-budget training; no best selection",
            "coverage": "All pixels in every cached holdout clip, not full original BVI videos",
            "scenes": len(data["split"]["internal_holdout"]), "clips": len(report["base"]["per_sequence"]),
            "frames": len(data["holdout_indices"]), "methods": report,
            "oracle_scope": "GT-a simplex optimum inside these two fixed source-bound corrections only"}


def run_experiment(cache, out, *, steps=1000, lr=1e-3, seed=0, batch_frames=8, device_name="cpu"):
    import torch
    if min(steps, batch_frames) < 1 or not np.isfinite(lr) or lr <= 0 or seed < 0:
        raise ValueError("Invalid fixed training budget")
    if device_name not in ("cpu", "npu", "cuda"):
        raise ValueError("device must be cpu, npu or cuda")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if device_name == "npu":
        import torch_npu  # noqa: F401
    device = torch.device("cpu" if device_name == "cpu" else device_name + ":0")
    torch.set_num_threads(4)
    selector = local_module("_correction_selector_train", CODE / "models/correction_selector.py")
    risk = local_module("_correction_risk_train", CODE / "models/correction_risk.py")
    data = prepare_cache(cache, selector, risk)
    normalization = fit_normalization(*(data[key].numpy() for key in ("features", "a", "gram")),
                                      data["fit_indices"], e0=data["e0"].numpy())
    plan = make_batch_plan(data["fit_indices"], steps, batch_frames, seed)
    plan_record = {"sampling": "Uniform fit frames with replacement; identical for all modes",
                   "seed": seed, "batches": plan.tolist(), "frame_index": data["frames"]}
    write_json(out / "sample_plan.json", plan_record)
    torch.manual_seed(seed)
    initial = selector.TinyCorrectionSelector()
    initial_state = copy.deepcopy(initial.state_dict())
    initial_hash = state_digest(initial_state)
    torch.save({"model": initial_state}, out / "initial_state.pt")
    args = {"steps": steps, "lr": lr, "seed": seed, "batch_frames": batch_frames,
            "device": str(device), "optimizer": "Adam default betas/eps, no weight decay", "modes": list(MODES)}
    metadata = {"purpose": "TRAIN-only selector learnability; not benchmark/SOTA evidence",
                "args": args, "split": data["split"], "normalization": normalization,
                "features": list(selector.FEATURE_NAMES), "context": "Exactly 3x3 patch-grid context",
                "architecture": {"name": "TinyCorrectionSelector", "channels": [19, 32, 32, 2],
                                 "kernels": [3, 1, 1], "direct_none_logit": 0.},
                "cache_provenance_sha256": digest(Path(cache) / "provenance.json"),
                "cache_summary_sha256": digest(Path(cache) / "oracle_summary.json"),
                "cache_provenance": data["cache_provenance"],
                "base_may_have_seen_internal_holdout": True, "initial_state_sha256": initial_hash,
                "initial_weights": [1 / 3, 1 / 3], "initial_none_weight": 1 / 3,
                "sample_plan_sha256": digest(out / "sample_plan.json"), "cache_clips": data["clips"],
                "source_sha256": {str(path.relative_to(CODE)): digest(path) for path in (
                    Path(__file__), CODE / "models/correction_selector.py", CODE / "models/correction_risk.py")},
                "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                            "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                "gt_firewall": "Only low/base/corrections/valid_mask -> features; a/e0 are loss labels; oracle only final evaluation",
                "losses": {"risk_full": "Normalized moment regression MSE",
                           "direct_mse": "Exact raw RGB patch MSE",
                           "direct_mse_normalized": "Exact raw RGB patch MSE / fit-only mean e0 (one positive scalar)"},
                "holdout_policy": "No intermediate evaluation or checkpoint selection; one pass after all three arms"}
    write_json(out / "provenance.json", metadata)
    mean = torch.tensor(normalization["feature_mean"], device=device)[None, :, None, None]
    std = torch.tensor(normalization["feature_std"], device=device)[None, :, None, None]
    target_scale = torch.tensor(normalization["target_scale"], device=device)
    states, durations = {}, {}
    for mode in MODES:
        torch.manual_seed(seed)
        model = selector.TinyCorrectionSelector().to(device)
        model.load_state_dict(initial_state, strict=True)
        if state_digest(model.state_dict()) != initial_hash:
            raise AssertionError("Methods must share exact initial state")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        started = time.monotonic()
        with (out / f"curve_{mode}.jsonl").open("w") as stream:
            for step, ix in enumerate(plan, start=1):
                ix = ix.tolist()
                x, a, gram, e0 = (data[key][ix].to(device) for key in ("features", "a", "gram", "e0"))
                optimizer.zero_grad(set_to_none=True)
                head = model((x - mean) / std)
                if mode == "risk_full":
                    target = selector.normalized_moment_target(a, gram, target_scale, normalization["energy_floor"])
                    loss = (head - target).square().mean()
                    with torch.no_grad():
                        weights = selector.weights_from_head(head.detach(), gram, mode, target_scale, normalization["energy_floor"])
                else:
                    weights = selector.weights_from_head(head, gram, mode, target_scale, normalization["energy_floor"])
                    loss = selector.quadratic_mse(e0, a, gram, weights).mean()
                    if mode == "direct_mse_normalized":
                        loss = loss / normalization["direct_mse_normalizer"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite {mode} loss at step {step}")
                with torch.no_grad():
                    batch_mse = selector.quadratic_mse(e0, a, gram, weights.detach()).mean()
                    if step == 1 and not bool(torch.allclose(weights.detach(), torch.full_like(weights, 1 / 3), atol=1e-7, rtol=0)):
                        raise AssertionError("Initial functions differ from w0")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError(f"Nonfinite {mode} gradient")
                optimizer.step()
                row = {"step": step, "loss": float(loss.detach()), "batch_rgb_mse": float(batch_mse),
                       "batch_base_mse": float(e0.mean()), "prediction_timing": "Before optimizer update"}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if step == 1 or step % 50 == 0 or step == steps:
                    stream.flush()
                    print(json.dumps({"mode": mode, **row}), flush=True)
        states[mode] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        durations[mode] = time.monotonic() - started
        model_hash = state_digest(states[mode])
        checkpoint = out / f"selector_{mode}.pt"
        torch.save({"schema_version": 1, "model": states[mode], "mode": mode, "steps": steps, "args": args,
                    "normalization": normalization, "split": data["split"],
                    "initial_state_sha256": initial_hash, "sample_plan_sha256": metadata["sample_plan_sha256"],
                    "model_state_sha256": model_hash, "provenance": copy.deepcopy(metadata)}, checkpoint)
        # Keep the serialized-file digest outside the file to avoid self-reference.
        # All base/RAFT/split/source provenance is also INSIDE the checkpoint.
        write_json(out / f"selector_{mode}.sha256.json", {
            "checkpoint": checkpoint.name, "checkpoint_sha256": digest(checkpoint),
            "model_state_sha256": model_hash, "initial_state_sha256": initial_hash,
            "sample_plan_sha256": metadata["sample_plan_sha256"],
            "cache_provenance_sha256": metadata["cache_provenance_sha256"],
            "base_checkpoint_sha256": data["cache_provenance"].get("base_checkpoint_sha256"),
            "base_state_sha256": data["cache_provenance"].get("base_state_sha256"),
            "raft_weights_sha256": data["cache_provenance"].get("raft_weights_sha256"),
            "split_sha256": data["cache_provenance"].get("split_sha256"),
            "source_sha256": metadata["source_sha256"]})
    final = final_holdout_evaluation(data, states, normalization, device, batch_frames, selector, risk)
    final["training_seconds"] = durations
    final["checkpoint_selection"] = "Last step only; holdout never used by optimizer/normalization/selection"
    write_json(out / "internal_holdout.json", final)
    print(json.dumps({"internal_holdout_rgb_mse": {key: row["rgb_mse"] for key, row in final["methods"].items()}}), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "npu", "cuda"), default="cpu")
    args = parser.parse_args()
    run_experiment(args.cache, args.out, steps=args.steps, lr=args.lr, seed=args.seed,
                   batch_frames=args.batch_frames, device_name=args.device)


if __name__ == "__main__":
    main()
