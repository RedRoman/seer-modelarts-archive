"""Frozen BVI VAL16 evaluation of final correction-selector checkpoints.

One base forward and prediction-only optical flow serve every comparison.
Ground truth is introduced only after all candidate weights/predictions exist.
No oracle selection or GT-derived statistics are used to produce predictions.
"""
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from models.correction_selector import (MODES, FEATURE_NAMES, TinyCorrectionSelector,
                                        observable_patch_features, weights_from_head)
from data.planned_bvi import _dataset_root, parse_sequence
from data.common import list_images
from train_correction_selector import state_digest as selector_state_digest
from train_method_pilot import digest, state_digest, write_json
from cache_correction_oracle import donor_correction, expand_patch
from eval_e4 import (SPLITS, build_model, held_out_sequences, lpips_of,
                     psnr_y_uint8, rgb_psnr_uint8)


def load_selectors(directory, dev):
    models, saved, hashes = {}, {}, {}
    for mode in MODES:
        path = directory / f"selector_{mode}.pt"
        sidecar = json.loads(path.with_suffix(".sha256.json").read_text())
        file_sha = digest(path)
        if file_sha != sidecar["checkpoint_sha256"]:
            raise ValueError(f"Selector checkpoint hash mismatch: {mode}")
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["schema_version"] != 1 or record["mode"] != mode or record["steps"] != 1000:
            raise ValueError("Expected the declared last-step 1000-update selector")
        provenance = record["provenance"]
        if provenance["features"] != list(FEATURE_NAMES):
            raise ValueError("Observable feature definition differs from training")
        for relative, expected in provenance["source_sha256"].items():
            if digest(CODE / relative) != expected:
                raise ValueError(f"Selector runtime differs from frozen source: {relative}")
        if (record["normalization"] != provenance["normalization"]
                or record["split"] != provenance["split"]
                or provenance["args"]["modes"] != list(MODES)):
            raise ValueError("Selector configuration and provenance disagree")
        fingerprint = selector_state_digest(record["model"])
        if fingerprint != record["model_state_sha256"] or fingerprint != sidecar["model_state_sha256"]:
            raise ValueError("Selector model-state fingerprint differs")
        model = TinyCorrectionSelector()
        model.load_state_dict(record["model"], strict=True)
        models[mode] = model.to(dev).eval()
        saved[mode], hashes[mode] = record, file_sha
    first = saved[MODES[0]]
    for record in saved.values():
        if (record["provenance"] != first["provenance"]
                or record["initial_state_sha256"] != first["initial_state_sha256"]
                or record["sample_plan_sha256"] != first["sample_plan_sha256"]):
            raise ValueError("Selectors do not share the declared matched training provenance")
    return models, first, hashes


def validate_scene_partition(saved):
    entry = json.loads(Path(SPLITS).read_text())["BVI_RLV"]
    train = {parse_sequence(name)[0] for name in entry["train"]}
    val = {parse_sequence(name)[0] for name in entry["val"]}
    split = saved["split"]
    fit, internal = set(split["fit"]), set(split["internal_holdout"])
    if (len(fit) != 18 or len(internal) != 6 or fit & internal
            or fit | internal != train or train & val or len(val) != 8):
        raise ValueError("Selector fit/holdout does not match the frozen BVI training scenes")
    root = _dataset_root(entry["root"], os.environ["SEER_ROOT"])
    pairs = []
    for name in entry["val"]:
        scene, condition = parse_sequence(name)
        low = list_images(str(root / scene / f"low_light_{condition}"))
        high = list_images(str(root / scene / f"normal_light_{condition}"))
        if (len(low) < 16 or len(low) != len(high)
                or [Path(p).name for p in low] != [Path(p).name for p in high]):
            raise ValueError(f"Unpaired frozen BVI validation frames: {name}")
        pairs.append({"name": f"BVI/{name}", "paired_frames": len(low),
                      "centre_start": (len(low) - 16) // 2,
                      "matching_complete_basename_lists": True})
    return pairs


@torch.no_grad()
def score_prediction(prediction, gt_u8, dev):
    pred_u8 = (prediction.float().clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy()
               * 255).round().astype(np.uint8)
    p = torch.from_numpy(pred_u8.copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    g = torch.from_numpy(gt_u8.copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    tlp, gt_tlp = lpips_of(p[1:], p[:-1], dev), lpips_of(g[1:], g[:-1], dev)
    return {"psnr_y": float(np.mean([psnr_y_uint8(a, b) for a, b in zip(pred_u8, gt_u8)])),
            "psnr_rgb": float(np.mean([rgb_psnr_uint8(a, b) for a, b in zip(pred_u8, gt_u8)])),
            "lpips": lpips_of(p, g, dev), "tlp": tlp, "gt_tlp": gt_tlp,
            "tlp_gap": abs(tlp - gt_tlp)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selectors", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["npu", "cuda"], default="npu")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.device == "npu":
        import torch_npu  # noqa: F401
    from torchvision.models.optical_flow import raft_small
    torch.set_num_threads(4)
    dev = torch.device(args.device + ":0")
    models, saved, selector_hashes = load_selectors(args.selectors, dev)
    origin = saved["provenance"]["cache_provenance"]
    if (digest(args.init) != origin["base_checkpoint_sha256"]
            or digest(args.raft_weights) != origin["raft_weights_sha256"]
            or digest(SPLITS) != origin["split_sha256"]):
        raise ValueError("Base checkpoint, RAFT weights or frozen split changed")
    validation_pairs = validate_scene_partition(saved)
    for relative, expected in origin["source_sha256"].items():
        if digest(CODE / relative) != expected:
            raise ValueError(f"Base/cache runtime differs from recorded source: {relative}")
    base_model, backbone, refiner, exposure, arm, length, _ = build_model(args.init, dev)
    if refiner is not None or exposure is not None or arm != "pretrained" or length != 16:
        raise ValueError("Unexpected base reconstruction")
    if (base_model.eval_reconstruction != origin["base_reconstruction"]
            or state_digest(base_model.state_dict()) != origin["base_state_sha256"]):
        raise ValueError("Loaded base differs from the cached training restorer")
    base_model.eval()
    raft = raft_small(weights=None).eval()
    raft.load_state_dict(torch.load(args.raft_weights, map_location="cpu", weights_only=True), strict=True)
    raft.requires_grad_(False)
    raft = raft.to(dev)
    config = origin["args"]
    patch, offset = config["patch_size"], config["offset"]
    norm = saved["normalization"]
    mean = torch.tensor(norm["feature_mean"], device=dev)[None, :, None, None]
    std = torch.tensor(norm["feature_std"], device=dev)[None, :, None, None]
    scale = torch.tensor(norm["target_scale"], device=dev)
    rows = []
    for name, lq_u8, gt_u8, kind in held_out_sequences("bvi", "val", 16, 256):
        if kind != "video" or len(lq_u8) != 16:
            raise ValueError("Expected frozen BVI validation video clips")
        low = torch.from_numpy(lq_u8.copy()).float().div_(255).permute(3, 0, 1, 2)[None].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            tokens = [value.float() for value in backbone(low)["feats"]]
            base = base_model(low, tokens)["pred_pixels"].float().clamp(0, 1)
        corrections, _, masks = zip(*(donor_correction(base, raft, delta, config["flow_steps"],
                                                       config["flow_batch"])
                                     for delta in (-offset, offset)))
        features, gram = observable_patch_features(low, base, torch.stack(corrections, 1),
                                                   torch.stack(masks, 1), patch)
        x = (features[0] - mean) / std
        predictions = {"base": base,
                       "fixed_average": base + .5 * (corrections[0] + corrections[1]),
                       "initial_thirds": base + (corrections[0] + corrections[1]) / 3}
        for mode, model in models.items():
            head = model(x)
            weights = weights_from_head(head, gram[0], mode, scale, norm["energy_floor"])
            if (not torch.isfinite(weights).all() or weights.min() < -1e-6
                    or weights.sum(-1).max() > 1 + 1e-6):
                raise ValueError("Selector output escaped its feasible correction set")
            predictions[mode] = base + sum(expand_patch(weights[..., j][None], patch) * corrections[j]
                                           for j in range(2))
        # GT first enters here, after all learned weights and RGB predictions.
        scores = {method: score_prediction(prediction, gt_u8, dev)
                  for method, prediction in predictions.items()}
        if not all(np.isfinite(value) for score in scores.values() for value in score.values()):
            raise FloatingPointError(name)
        rows.append({"name": name, "methods": scores})
        print(json.dumps(rows[-1]), flush=True)
    if len(rows) != 16 or len({row["name"] for row in rows}) != 16:
        raise ValueError("Expected all 16 distinct frozen BVI validation sequences")
    if state_digest(base_model.state_dict()) != origin["base_state_sha256"]:
        raise AssertionError("Validation altered base state")
    for mode, model in models.items():
        original = torch.load(args.selectors / f"selector_{mode}.pt", map_location="cpu", weights_only=True)
        if selector_state_digest(model.state_dict()) != original["model_state_sha256"]:
            raise AssertionError("Validation altered selector state")
    result = {"split": "val", "domain": "bvi", "n": 16, "per_sequence": rows,
              "protocol": "frozen BVI VAL16; centre16 frames/centre256 crop; rounded uint8 RGB/Y/LPIPS/tLP",
              "means": {method: {metric: float(np.mean([row["methods"][method][metric] for row in rows]))
                                  for metric in rows[0]["methods"][method]}
                        for method in rows[0]["methods"]},
              "selector_checkpoint_sha256": selector_hashes,
              "validation_pair_inventory": validation_pairs,
              "source_sha256": {str(path.relative_to(CODE)): digest(path) for path in (
                  Path(__file__), CODE / "tools/cache_correction_oracle.py",
                  CODE / "models/correction_selector.py", CODE / "models/correction_risk.py")},
              "base_cache_provenance": origin, "selector_training": saved["provenance"],
              "selection": "Predeclared last-step1000; exploratory VAL; no test evaluation or SOTA claim"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result["means"]), flush=True)


if __name__ == "__main__":
    main()
