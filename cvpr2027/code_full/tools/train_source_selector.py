"""Source-only fixed-budget correction heads; no target frames or selection.

Reuse the frozen observable features/network/MSE machinery. All three candidate
families use the same source clips, scene split, seed-specific model state and
frame sampling plan. Normalization is fit-scene-only and family-specific.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from train_correction_selector import (digest, write_json, local_module, state_digest,
                                       fit_normalization, make_batch_plan)
from cache_source_corrections import (source_scene, source_names, source_hashes,
                                      check_source_hashes)

MODES = ("direct_mse", "direct_mse_normalized")
FAMILIES = ("restored", "restored_detail", "raw_detail")
SPLIT_SEED = 20261114


def make_scene_split(names):
    scenes = sorted({source_scene(name) for name in names})
    if len(scenes) < 4:
        raise ValueError("Need at least four source scenes")
    order = np.random.default_rng(SPLIT_SEED).permutation(len(scenes))
    nfit = 3 * len(scenes) // 4
    return {"seed": SPLIT_SEED, "fit": [scenes[int(i)] for i in order[:nfit]],
            "internal_holdout": [scenes[int(i)] for i in order[nfit:]],
            "scope": "75/25 source TRAIN scenes; internal to base training; no target data or model selection"}


def cache_manifest(directory, splits):
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    summary = json.loads((directory / "cache_summary.json").read_text())
    names = source_names(json.loads(Path(splits).read_text()))
    if (provenance.get("domain") != "sdsd" or provenance.get("source_names") != names
            or provenance.get("split_sha256") != digest(splits)
            or summary.get("n") != 136 or summary.get("oracle_computed") is not False
            or summary.get("base_state_unchanged") is not True or summary.get("raft_state_unchanged") is not True):
        raise ValueError("Require completed source-only SDSD136 observation cache")
    for key in ("base_checkpoint_sha256", "base_state_sha256", "raft_weights_sha256", "raft_state_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key, ""))) is None:
            raise ValueError(f"Invalid cache hash: {key}")
    check_source_hashes(provenance["source_sha256"])
    expected = [f"clip_{i:03d}.pt" for i in range(136)]
    rows, plan = summary["per_sequence"], provenance["plan"]
    files = sorted(directory.glob("clip_*.pt"))
    if ([p.name for p in files] != expected or [r["cache_path"] for r in rows] != expected
            or [r["name"] for r in rows] != names or len(plan) != 136
            or [p["sequence_index"] for p in plan] != list(range(136))):
        raise ValueError("Source cache names/order/plan differ from frozen TRAIN136")
    for path, row in zip(files, rows):
        if digest(path) != row["cache_sha256"] or row["scene"] != source_scene(row["name"]):
            raise ValueError(f"Cache file or scene mismatch: {path}")
    return files, provenance, rows


def prepare_cache(directory, family, splits):
    import torch
    selector = local_module("_source_selector", CODE / "models/correction_selector.py")
    risk = local_module("_source_moments", CODE / "models/correction_risk.py")
    candidates = local_module("_source_candidates", CODE / "models/observation_candidates.py")
    files, provenance, rows = cache_manifest(directory, splits)
    all_x, all_a, all_g, all_e0, frames, clips = [], [], [], [], [], []
    recipe, parity = None, []
    patch = provenance["args"]["patch_size"]
    for i, (path, row) in enumerate(zip(files, rows)):
        sample = torch.load(path, map_location="cpu", weights_only=True)
        if (sample["name"] != row["name"] or sample["scene"] != row["scene"]
                or sample["plan"] != provenance["plan"][i]):
            raise ValueError("Source clip identity/plan mismatch")
        t = sample["tensors"]
        low, base, flow, mask = (t[k] for k in ("lq", "base", "flow", "valid_mask"))
        if low.shape != (1, 3, 16, 256, 256) or base.shape != low.shape:
            raise ValueError("Expected the declared source crop256/clip16")
        with torch.no_grad():
            directions, valid, meta = candidates.build_observation_candidates(
                low, base, flow, mask, mode=family, offset=provenance["args"]["offset"])
            if recipe is not None and meta != recipe:
                raise ValueError("Candidate recipe changed across source clips")
            recipe = meta
            if family == "restored":
                difference = (directions - t["restored_corrections"]).abs()
                parity.append({"name": row["name"], "max_abs": float(difference.max()),
                               "mean_abs": float(difference.mean())})
                if difference.max() > 2e-5:
                    raise ValueError("CPU reconstruction exceeds declared NPU cached-direction tolerance 2e-5")
            # Explicit firewall: this feature call receives no GT or GT moments.
            x, gram = selector.observable_patch_features(low, base, directions, valid, patch)
            gt = t["gt"]
            if (gt.shape != base.shape or gt.dtype != torch.float32 or not torch.isfinite(gt).all()
                    or gt.min() < 0 or gt.max() > 1):
                raise ValueError("Invalid source-only GT labels")
            a, checked_gram = risk.patch_moments(gt - base, directions, patch)
            torch.testing.assert_close(gram, checked_gram, atol=1e-8, rtol=1e-5)
            e0 = selector._patch_pool((gt - base).square(), patch).mean(2)
        all_x.append(x[0]); all_a.append(a[0]); all_g.append(gram[0]); all_e0.append(e0[0])
        clips.append({"name": row["name"], "scene": row["scene"], "sha256": row["cache_sha256"]})
        frames.extend({"clip_index": i, "frame": j, "scene": row["scene"], "name": row["name"]} for j in range(16))
    split = make_scene_split([r["name"] for r in rows])
    return {"features": torch.cat(all_x), "a": torch.cat(all_a), "gram": torch.cat(all_g),
            "e0": torch.cat(all_e0), "frames": frames, "clips": clips, "split": split,
            "fit_indices": [i for i, r in enumerate(frames) if r["scene"] in split["fit"]],
            "holdout_indices": [i for i, r in enumerate(frames) if r["scene"] in split["internal_holdout"]],
            "cache_provenance": provenance, "candidate_recipe": recipe,
            "restored_cpu_vs_cache_parity": parity}


def final_internal(data, states, norm, selector, device):
    """Source internal holdout only, once after both final heads. No oracle."""
    import torch
    mean = torch.tensor(norm["feature_mean"], device=device)[None, :, None, None]
    std = torch.tensor(norm["feature_std"], device=device)[None, :, None, None]
    models = {}
    for mode, state in states.items():
        model = selector.TinyCorrectionSelector().to(device).eval()
        model.load_state_dict(state, strict=True)
        models[mode] = model
    scores = {k: {} for k in ("base", "initial_thirds", *MODES)}
    with torch.no_grad():
        indices = data["holdout_indices"]
        for start in range(0, len(indices), 8):
            ix = indices[start:start + 8]
            x, a, g, e0 = (data[k][ix].to(device) for k in ("features", "a", "gram", "e0"))
            weights = {"base": torch.zeros_like(a), "initial_thirds": torch.full_like(a, 1 / 3)}
            weights.update({mode: selector.weights_from_head(model((x - mean) / std), g, mode, None, None)
                            for mode, model in models.items()})
            for method, w in weights.items():
                mse = selector.quadratic_mse(e0, a, g, w).mean((-2, -1)).double().cpu().tolist()
                for j, value in zip(ix, mse):
                    scores[method].setdefault(data["frames"][j]["name"], []).append(value)
    return {"scope": data["split"]["scope"], "selection": "None; final fixed-budget checkpoint only",
            "oracle_computed": False,
            "methods": {m: {"mean_rgb_mse": float(np.mean([np.mean(v) for v in rows.values()])),
                             "per_sequence": {n: float(np.mean(v)) for n, v in rows.items()}}
                        for m, rows in scores.items()}}


def run(cache, out, family, seed, device_name, splits):
    import torch
    if family not in FAMILIES or seed not in (0, 1, 2):
        raise ValueError("Use a predeclared candidate family and selector seed0/1/2")
    out.mkdir(parents=True, exist_ok=False)
    if device_name == "npu":
        import torch_npu  # noqa: F401
    device = torch.device("cpu" if device_name == "cpu" else device_name + ":0")
    torch.set_num_threads(4)
    data = prepare_cache(cache, family, splits)
    selector = local_module("_source_selector_train", CODE / "models/correction_selector.py")
    norm = fit_normalization(*(data[k].numpy() for k in ("features", "a", "gram")),
                             data["fit_indices"], e0=data["e0"].numpy())
    plan = make_batch_plan(data["fit_indices"], 1000, 8, seed)
    write_json(out / "sample_plan.json", {"seed": seed, "batches": plan.tolist(), "frame_index": data["frames"]})
    torch.manual_seed(seed)
    initial = selector.TinyCorrectionSelector()
    initial_state = copy.deepcopy(initial.state_dict())
    initial_sha = state_digest(initial_state)
    torch.save({"model": initial_state}, out / "initial_state.pt")
    metadata = {"schema_version": 1, "purpose": "Source-only SDSD correction selector; target untouched by training",
                "args": {"steps": 1000, "batch_frames": 8, "lr": 1e-3, "seed": seed,
                         "device": str(device), "modes": list(MODES), "candidate_mode": family,
                         "optimizer": "Adam default betas/eps, no weight decay"},
                "candidate_recipe": data["candidate_recipe"], "normalization": norm, "split": data["split"],
                "candidate_backend": "CPU float32 for all families, source and target",
                "restored_cpu_vs_cache_parity": data["restored_cpu_vs_cache_parity"],
                "features": list(selector.FEATURE_NAMES), "initial_state_sha256": initial_sha,
                "sample_plan_sha256": digest(out / "sample_plan.json"), "cache_clips": data["clips"],
                "cache_provenance": data["cache_provenance"],
                "cache_provenance_sha256": digest(cache / "provenance.json"),
                "cache_summary_sha256": digest(cache / "cache_summary.json"),
                "source_sha256": source_hashes("tools/train_source_selector.py", "tools/train_correction_selector.py",
                    "models/correction_selector.py", "models/correction_risk.py", "models/observation_candidates.py"),
                "base_may_have_seen_internal_holdout": True,
                "initial_weights": [1 / 3, 1 / 3], "initial_none_weight": 1 / 3,
                "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__), "cpu_threads": 4},
                "gt_firewall": "low/base/flow/mask -> candidates/features; source GT only for a/e0 loss labels",
                "selection": "No target data, intermediate holdout or best checkpoint selection"}
    write_json(out / "provenance.json", metadata)
    mean = torch.tensor(norm["feature_mean"], device=device)[None, :, None, None]
    std = torch.tensor(norm["feature_std"], device=device)[None, :, None, None]
    states = {}
    for mode in MODES:
        torch.manual_seed(seed)
        model = selector.TinyCorrectionSelector().to(device)
        model.load_state_dict(initial_state, strict=True)
        if state_digest(model.state_dict()) != initial_sha:
            raise AssertionError("Shared initialization differs")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with (out / f"curve_{mode}.jsonl").open("w") as stream:
            for step, ix in enumerate(plan, 1):
                x, a, g, e0 = (data[k][ix.tolist()].to(device) for k in ("features", "a", "gram", "e0"))
                optimizer.zero_grad(set_to_none=True)
                w = selector.weights_from_head(model((x - mean) / std), g, mode, None, None)
                mse = selector.quadratic_mse(e0, a, g, w).mean()
                loss = mse / norm["direct_mse_normalizer"] if mode == "direct_mse_normalized" else mse
                if not torch.isfinite(loss) or (step == 1 and not torch.allclose(w, torch.full_like(w, 1 / 3), atol=1e-7, rtol=0)):
                    raise AssertionError("Nonfinite loss or mismatched initial function")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite selector gradient")
                optimizer.step()
                row = {"step": step, "loss": float(loss.detach()), "batch_rgb_mse": float(mse.detach()),
                       "batch_base_mse": float(e0.mean()), "prediction_timing": "Before optimizer update"}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if step == 1 or step % 50 == 0:
                    stream.flush(); print(json.dumps({"mode": mode, **row}), flush=True)
        states[mode] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = {"schema_version": 1, "mode": mode, "steps": 1000, "model": states[mode],
                  "model_state_sha256": state_digest(states[mode]), "provenance": copy.deepcopy(metadata)}
        path = out / f"selector_{mode}.pt"
        torch.save(record, path)
        write_json(path.with_suffix(".sha256.json"), {"checkpoint_sha256": digest(path),
                   "model_state_sha256": record["model_state_sha256"], "initial_state_sha256": initial_sha,
                   "sample_plan_sha256": metadata["sample_plan_sha256"]})
    write_json(out / "internal_holdout.json", final_internal(data, states, norm, selector, device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-mode", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", choices=("cpu", "npu", "cuda"), default="cpu")
    parser.add_argument("--splits", type=Path, default=Path(os.environ.get("SEER_ROOT", "/cache/zihaohe/seer_cvpr")) / "datasets/splits_cvpr.json")
    args = parser.parse_args()
    run(args.cache, args.out, args.candidate_mode, args.seed, args.device, args.splits)


if __name__ == "__main__":
    main()
