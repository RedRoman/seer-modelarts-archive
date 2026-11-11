"""Score source-trained fixed-final heads on BVI VAL16, never target-tune.

Pass multiple family/seed directories to share exactly one base/RAFT candidate
computation per target clip. GT is used only after all predictions exist.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from train_correction_selector import digest, write_json, state_digest as head_digest
from train_source_selector import MODES, FAMILIES, make_scene_split
from cache_source_corrections import (source_names, source_scene, source_hashes,
                                      check_source_hashes, check_base_args, backbone_tokens)


def validate_training_metadata(provenance, splits):
    """Pure metadata guard; source-only split and fixed budgets are mandatory."""
    args = provenance["args"]
    if (args["candidate_mode"] not in FAMILIES or args["modes"] != list(MODES)
            or args["seed"] not in (0, 1, 2) or args["steps"] != 1000
            or args["batch_frames"] != 8 or args["lr"] != 1e-3):
        raise ValueError("Selector is outside the fixed source-only protocol")
    names = source_names(json.loads(Path(splits).read_text()))
    origin = provenance["cache_provenance"]
    if (origin["domain"] != "sdsd" or origin["source_names"] != names
            or origin["split_sha256"] != digest(splits)
            or provenance["split"] != make_scene_split(names)
            or [row["name"] for row in provenance["cache_clips"]] != names
            or any(row["scene"] != source_scene(row["name"]) for row in provenance["cache_clips"])):
        raise ValueError("Source-only cache, split or fit/holdout partition mismatch")
    norm = provenance["normalization"]
    if provenance["candidate_backend"] != "CPU float32 for all families, source and target":
        raise ValueError("Unexpected candidate backend")
    fit_frames = sum(row["scene"] in provenance["split"]["fit"] for row in provenance["cache_clips"]) * 16
    if (norm["fit_frames"] != fit_frames or norm["fit_patches"] != fit_frames * 32 * 32
            or "Fit scenes only" not in norm["source"]
            or not np.isfinite(norm["direct_mse_normalizer"]) or norm["direct_mse_normalizer"] <= 0):
        raise ValueError("Normalization does not record the source fit-only scope")


def load_selectors(directories, device, splits):
    import torch
    from models.correction_selector import TinyCorrectionSelector, FEATURE_NAMES
    heads, groups, hashes, fingerprints = {}, {}, {}, {}
    reference = None
    for directory in directories:
        pair = []
        for mode in MODES:
            path = directory / f"selector_{mode}.pt"
            sidecar = json.loads(path.with_suffix(".sha256.json").read_text())
            sha = digest(path)
            if sha != sidecar["checkpoint_sha256"]:
                raise ValueError(f"Selector file SHA mismatch: {path}")
            record = torch.load(path, map_location="cpu", weights_only=True)
            p = record["provenance"]
            if record["schema_version"] != 1 or record["mode"] != mode or record["steps"] != 1000:
                raise ValueError("Expected fixed final1000 source selector")
            validate_training_metadata(p, splits)
            check_source_hashes(p["source_sha256"])
            check_source_hashes(p["cache_provenance"]["source_sha256"])
            if p["features"] != list(FEATURE_NAMES):
                raise ValueError("Source feature definition differs")
            fingerprint = head_digest(record["model"])
            if fingerprint != record["model_state_sha256"] or fingerprint != sidecar["model_state_sha256"]:
                raise ValueError("Selector state SHA mismatch")
            for key in ("initial_state_sha256", "sample_plan_sha256"):
                if sidecar[key] != p[key]:
                    raise ValueError(f"Selector sidecar differs: {key}")
            group = f"{p['args']['candidate_mode']}_s{p['args']['seed']}"
            key = f"{group}/{mode}"
            if key in heads:
                raise ValueError(f"Duplicate family/seed/mode: {key}")
            model = TinyCorrectionSelector().to(device).eval()
            model.load_state_dict(record["model"], strict=True)
            model.requires_grad_(False)
            heads[key], hashes[key], fingerprints[key] = model, sha, fingerprint
            pair.append(p)
        if pair[0] != pair[1]:
            raise ValueError("The two losses do not share matched source training provenance")
        p = pair[0]
        if reference is not None and any(p[k] != reference[k] for k in
                ("cache_provenance", "cache_provenance_sha256", "cache_summary_sha256", "split", "source_sha256")):
            raise ValueError("Source cache/split/code changed between families/seeds")
        reference = p
        groups[group] = p
    if not heads:
        raise ValueError("At least one complete source-trained selector pair is required")
    # Across candidate families, each seed must preserve exact state/sample plan.
    for seed in (0, 1, 2):
        selected = [p for p in groups.values() if p["args"]["seed"] == seed]
        for key in ("initial_state_sha256", "sample_plan_sha256"):
            if len({p[key] for p in selected}) > 1:
                raise ValueError(f"Family comparison is not matched: seed{seed}/{key}")
    return heads, groups, hashes, fingerprints


def target_inventory(splits, root):
    from data.planned_bvi import _dataset_root, parse_sequence, validate_split
    from data.common import list_images
    entry = json.loads(Path(splits).read_text())["BVI_RLV"]
    validate_split(entry)
    folder = _dataset_root(entry["root"], root)
    rows = []
    for name in entry["val"]:
        scene, condition = parse_sequence(name)
        low = list_images(str(folder / scene / f"low_light_{condition}"))
        gt = list_images(str(folder / scene / f"normal_light_{condition}"))
        if (len(low) < 16 or len(low) != len(gt)
                or [Path(p).name for p in low] != [Path(p).name for p in gt]):
            raise ValueError(f"Target VAL frame pairing mismatch: {name}")
        rows.append({"name": f"BVI/{name}", "paired_frames": len(low), "centre_start": (len(low) - 16) // 2,
                     "matching_complete_basename_lists": True})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectors", type=Path, nargs="+", required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
    from torchvision.models.optical_flow import raft_small
    from models.observation_candidates import build_observation_candidates
    from models.correction_selector import observable_patch_features, weights_from_head
    from cache_correction_oracle import donor_correction, expand_patch
    from eval_correction_selector import score_prediction
    from eval_e4 import SPLITS, VJEPA_CKPT, build_model, held_out_sequences
    from train_method_pilot import state_digest
    torch.set_num_threads(4)
    dev = torch.device(args.device + ":0")
    heads, groups, head_shas, fingerprints = load_selectors(args.selectors, dev, SPLITS)
    origin = next(iter(groups.values()))["cache_provenance"]
    if digest(args.init) != origin["base_checkpoint_sha256"] or digest(args.raft_weights) != origin["raft_weights_sha256"]:
        raise ValueError("Source base/RAFT file differs at target evaluation")
    if origin["base_saved_args"]["vjepa"] == "pretrained":
        if origin.get("encoder_eval") is not True or origin.get("encoder_weights_sha256") != digest(VJEPA_CKPT):
            raise ValueError("Missing or changed frozen encoder weights/eval provenance")
    inventory = target_inventory(SPLITS, os.environ["SEER_ROOT"])
    base_model, backbone, refiner, exposure, arm, length, base_args = build_model(args.init, dev)
    check_base_args(base_args, arm, length)
    if refiner is not None or exposure is not None or base_model.eval_reconstruction != origin["base_reconstruction"]:
        raise ValueError("Source base reconstruction changed")
    base_model.eval().requires_grad_(False)
    if state_digest(base_model.state_dict()) != origin["base_state_sha256"]:
        raise ValueError("Source base state changed")
    raft = raft_small(weights=None).eval()
    raft.load_state_dict(torch.load(args.raft_weights, map_location="cpu", weights_only=True), strict=True)
    raft.requires_grad_(False).to(dev)
    if state_digest(raft.state_dict()) != origin["raft_state_sha256"]:
        raise ValueError("Source RAFT state changed")
    config = origin["args"]
    patch, offset = config["patch_size"], config["offset"]
    families = sorted({p["args"]["candidate_mode"] for p in groups.values()})
    rows, parity = [], []
    with torch.no_grad():
        for name, low_u8, gt_u8, kind in held_out_sequences("bvi", "val", 16, 256):
            if kind != "video" or low_u8.shape != (16, 256, 256, 3) or gt_u8.shape != low_u8.shape:
                raise ValueError("Target does not match frozen center16/crop256")
            low = torch.from_numpy(low_u8.copy()).float().div_(255).permute(3, 0, 1, 2)[None].to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                tokens = backbone_tokens(backbone, low)
                base = base_model(low, tokens)["pred_pixels"].float().clamp(0, 1)
            old_dirs, flows, masks = zip(*(donor_correction(base, raft, d, config["flow_steps"], config["flow_batch"])
                                          for d in (-offset, offset)))
            flows, masks = torch.stack(flows, 1), torch.stack(masks, 1)
            # Match source cache preparation: every candidate family and feature
            # is constructed on CPU float32, then shared across all seed heads.
            low_cpu, base_cpu, flows_cpu, masks_cpu = (v.cpu() for v in (low, base, flows, masks))
            predictions = {"base": base}
            for family in families:
                directions_cpu, valid_cpu, recipe = build_observation_candidates(
                    low_cpu, base_cpu, flows_cpu, masks_cpu, mode=family, offset=offset)
                if family == "restored":
                    difference = (directions_cpu - torch.stack(old_dirs, 1).cpu()).abs()
                    parity.append({"name": name, "max_abs": float(difference.max()), "mean_abs": float(difference.mean())})
                    if difference.max() > 2e-5:
                        raise ValueError("Target CPU/NPU donor reconstruction exceeds declared 2e-5 tolerance")
                features_cpu, gram_cpu = observable_patch_features(low_cpu, base_cpu, directions_cpu, valid_cpu, patch)
                directions, features, gram = (v.to(dev) for v in (directions_cpu, features_cpu, gram_cpu))
                predictions[f"{family}/fixed_average"] = base + .5 * directions.sum(1)
                predictions[f"{family}/initial_thirds"] = base + directions.sum(1) / 3
                for group, trained in groups.items():
                    if trained["args"]["candidate_mode"] != family:
                        continue
                    if trained["candidate_recipe"] != recipe:
                        raise ValueError("Target candidate recipe differs from source training")
                    norm = trained["normalization"]
                    mean = torch.tensor(norm["feature_mean"], device=dev)[None, :, None, None]
                    std = torch.tensor(norm["feature_std"], device=dev)[None, :, None, None]
                    x = (features[0] - mean) / std
                    for mode in MODES:
                        key = f"{group}/{mode}"
                        w = weights_from_head(heads[key](x), gram[0], mode, None, None)
                        if not torch.isfinite(w).all() or w.min() < 0 or w.sum(-1).max() > 1 + 1e-6:
                            raise ValueError("Target selector weights invalid")
                        predictions[key] = base + sum(expand_patch(w[..., j][None], patch) * directions[:, j] for j in range(2))
            # GT first enters computation here, after all family/seed predictions.
            scores = {method: score_prediction(pred, gt_u8, dev) for method, pred in predictions.items()}
            if not all(np.isfinite(v) for values in scores.values() for v in values.values()):
                raise FloatingPointError(name)
            rows.append({"name": name, "methods": scores})
            print(json.dumps(rows[-1]), flush=True)
    if sorted(r["name"] for r in rows) != sorted(r["name"] for r in inventory):
        raise ValueError("Incomplete or duplicate target VAL16 output")
    if state_digest(base_model.state_dict()) != origin["base_state_sha256"] or state_digest(raft.state_dict()) != origin["raft_state_sha256"]:
        raise AssertionError("Target evaluation altered frozen base/RAFT state")
    if any(head_digest(model.state_dict()) != fingerprints[key] for key, model in heads.items()):
        raise AssertionError("Target evaluation altered selector state")
    result = {"scope": "Source SDSD-only E17 base and selectors; fixed final heads; target BVI VAL16 scoring only",
              "domain": "bvi", "split": "val", "n": 16, "per_sequence": rows,
              "means": {m: {k: float(np.mean([r["methods"][m][k] for r in rows])) for k in rows[0]["methods"][m]}
                        for m in rows[0]["methods"]}, "selector_checkpoint_sha256": head_shas,
              "selector_training": groups, "base_cache_provenance": origin, "validation_pair_inventory": inventory,
              "candidate_backend": "CPU float32 for all families, source and target",
              "restored_cpu_vs_npu_parity": parity,
              "source_sha256": source_hashes("tools/eval_source_selector.py", "tools/eval_correction_selector.py",
                  "tools/train_source_selector.py", "models/observation_candidates.py", "models/correction_selector.py"),
              "state_unchanged": {"base": True, "raft": True, "all_selectors": True},
              "protocol": "One shared base/RAFT per clip; center16/crop256; rounded uint8 RGB/Y/LPIPS/tLP",
              "selection": "No target fitting, normalization, oracle or checkpoint selection; all predeclared final1000 heads"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
