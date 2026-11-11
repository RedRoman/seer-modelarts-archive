"""Evaluate every declared source-trained observation/trajectory head on BVI VAL.

All methods share one base, +/-2 donor flow and +/-1 graph flow per clip.
Candidates, graph heads and reconstruction use CPU float32; only frozen base,
RAFT and common uint8 metrics run on NPU. Target GT is used only for scoring.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from train_observation_v2 import (VARIANTS, CANDIDATES, EXTRA_SOURCES, modules,
                                  STEPS, CROP, CLIP_LEN, LOSS_FRAMES, LR, make_plan)
from train_correction_selector import digest, write_json, state_digest
from cache_source_corrections import check_source_hashes, source_hashes


def validate_preparation(meta, manifest, splits):
    """Fail closed on source identity and the fixed selector-only scene split."""
    from train_source_selector import make_scene_split
    split = json.loads(Path(splits).read_text())
    names = [f"{d}/{n}" for d in ("SDSD_indoor", "SDSD_outdoor") for n in split[d]["train"]]
    origin = meta["source_cache_provenance"]
    if (meta["n"] != 136 or meta["oracle_computed"] is not False
            or meta["target_frames_read"] is not False or len(names) != 136
            or [r["name"] for r in meta["per_sequence"]] != names or origin["source_names"] != names
            or origin["split_sha256"] != digest(splits) or meta["split"] != make_scene_split(names)
            or meta["variants"] != {k: list(v) for k, v in VARIANTS.items()}):
        raise ValueError("Require the frozen completed SDSD-only prepared cache")
    check_source_hashes(meta["source_sha256"])
    check_source_hashes(origin["source_sha256"])
    if meta["source_sha256"] != source_hashes(*EXTRA_SOURCES) or origin["source_sha256"] != source_hashes():
        raise ValueError("Incomplete or unexpected source-code hash manifest")
    return origin, digest(manifest)


def load_heads(directories, meta, manifest_sha, graph_module):
    import torch
    expected = {f"{variant}_s{seed}" for variant in VARIANTS for seed in (0, 1, 2)}
    models, records, hashes, fingerprints = {}, {}, {}, {}
    initial, plans = {}, {}
    for directory in directories:
        directory = Path(directory)
        checkpoint = directory / "final.pt"
        record = torch.load(checkpoint, map_location="cpu", weights_only=True)
        p = record["provenance"]
        key = f"{p['variant']}_s{p['seed']}"
        if key not in expected or key in models:
            raise ValueError("Duplicate or undeclared final head")
        candidate, mode = VARIANTS[p["variant"]]
        a = p["args"]
        if (p["candidate"] != candidate or record["steps"] != STEPS or a["steps"] != STEPS or a["source_crop"] != CROP
                or a["loss_frames"] != LOSS_FRAMES or a["lr"] != LR or a["device"] != "cpu"
                or a["clips_per_update"] != 1 or a["clip_len"] != CLIP_LEN
                or p["prepared_manifest_sha256"] != manifest_sha
                or p["normalization"] != meta["normalization"][candidate]
                or p["candidate_recipe"] != meta["candidate_recipes"][candidate]
                or p["source_names"] != meta["source_cache_provenance"]["source_names"]
                or p["source_scene_split"] != meta["split"]
                or p["source_split_sha256"] != meta["source_cache_provenance"]["split_sha256"]
                or p["base_checkpoint_sha256"] != meta["source_cache_provenance"]["base_checkpoint_sha256"]):
            raise ValueError("Source scope, fixed budget, normalization or candidate lineage differs")
        check_source_hashes(p["source_sha256"])
        if p["source_sha256"] != meta["source_sha256"]:
            raise ValueError("Training source manifest differs from preparation")
        sha = digest(checkpoint)
        side = json.loads((directory / "final.sha256.json").read_text())
        if sha != side["checkpoint_sha256"] or record["model_state_sha256"] != state_digest(record["model"]):
            raise ValueError("Final checkpoint hash differs")
        plan_path = directory / "sample_plan.json"
        actual_plan = json.loads(plan_path.read_text())
        if (digest(plan_path) != p["sample_plan_sha256"]
                or actual_plan != make_plan(p["source_names"], p["source_scene_split"], p["seed"])):
            raise ValueError("Actual sample plan differs from the frozen source-only budget")
        initial_record = torch.load(directory / "initial_state.pt", map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p["seed"])
            expected_initial = graph_module.TrajectoryReconstruction(mode=mode)
        if (state_digest(initial_record["model"]) != p["initial_state_sha256"]
                or state_digest(expected_initial.state_dict()) != p["initial_state_sha256"]):
            raise ValueError("Actual initialization differs from the declared seed")
        for collection, field in ((initial, "initial_state_sha256"), (plans, "sample_plan_sha256")):
            if p["seed"] in collection and collection[p["seed"]] != p[field]:
                raise ValueError("Initialization or sampling differs across variants")
            collection[p["seed"]] = p[field]
        model = graph_module.TrajectoryReconstruction(mode=mode).eval().requires_grad_(False)
        model.load_state_dict(record["model"], strict=True)
        if model.get_config() != p["model_config"]:
            raise ValueError("Graph configuration changed")
        models[key], records[key], hashes[key] = model, p, sha
        fingerprints[key] = state_digest(model.state_dict())
    if set(models) != expected or len(set(initial.values())) != 3 or len(set(plans.values())) != 3:
        raise ValueError("Need all seven variants and three distinct selector seeds")
    return models, records, hashes, fingerprints


def load_v1_references(directories, origin, selector_module):
    import torch
    models, records, hashes = {}, {}, {}
    for directory in directories:
        path = Path(directory) / "selector_direct_mse.pt"
        record = torch.load(path, map_location="cpu", weights_only=True)
        p = record["provenance"]
        seed = p["args"]["seed"]
        if (seed not in (0, 1, 2) or seed in models or p["args"]["candidate_mode"] != "raw_detail"
                or p["cache_provenance"] != origin or record["mode"] != "direct_mse"
                or record["steps"] != 1000):
            raise ValueError("V1 reference differs from the fixed source raw/direct head")
        check_source_hashes(p["source_sha256"])
        if p["source_sha256"] != source_hashes("tools/train_source_selector.py", "tools/train_correction_selector.py",
                "models/correction_selector.py", "models/correction_risk.py", "models/observation_candidates.py"):
            raise ValueError("Incomplete V1 source hash manifest")
        side = json.loads(path.with_suffix(".sha256.json").read_text())
        if digest(path) != side["checkpoint_sha256"] or state_digest(record["model"]) != record["model_state_sha256"]:
            raise ValueError("V1 reference file/state changed")
        model = selector_module.TinyCorrectionSelector().eval().requires_grad_(False)
        model.load_state_dict(record["model"], strict=True)
        models[seed], records[seed], hashes[f"v1_raw_direct_s{seed}"] = model, p, digest(path)
    if set(models) != {0, 1, 2}:
        raise ValueError("Need all three frozen V1 raw/direct references")
    return models, records, hashes


def standardized(features, norm):
    import torch
    mean = torch.tensor(norm["feature_mean"])[None, None, :, None, None]
    std = torch.tensor(norm["feature_std"])[None, None, :, None, None]
    return (features - mean) / std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--heads", type=Path, nargs="+", required=True)
    parser.add_argument("--v1-references", type=Path, nargs="+", required=True)
    parser.add_argument("--seer-root", type=Path, default=Path("/temp/zihaohe/seer_cvpr"))
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    os.environ["SEER_ROOT"] = str(args.seer_root)
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
    from torchvision.models.optical_flow import raft_small
    from cache_correction_oracle import donor_correction, expand_patch
    from cache_adjacent_flows import adjacent_flows
    from cache_source_corrections import backbone_tokens, check_base_args
    from eval_source_selector import target_inventory
    from eval_correction_selector import score_prediction
    from eval_e4 import SPLITS, VJEPA_CKPT, build_model, held_out_sequences
    from train_method_pilot import state_digest as base_state_digest
    import common.eval_lib as metric_lib
    torch.set_num_threads(4)
    dev = torch.device(args.device + ":0")
    meta = json.loads(args.prepared_manifest.read_text())
    origin, manifest_sha = validate_preparation(meta, args.prepared_manifest, SPLITS)
    m = modules()
    heads, training, head_hashes, fingerprints = load_heads(args.heads, meta, manifest_sha, m["trajectory"])
    old, old_records, old_hashes = load_v1_references(args.v1_references, origin, m["selector"])
    old_fingerprints = {seed: state_digest(model.state_dict()) for seed, model in old.items()}
    if digest(args.init) != origin["base_checkpoint_sha256"] or digest(args.raft_weights) != origin["raft_weights_sha256"]:
        raise ValueError("Base/RAFT input differs from source preparation")
    if origin["base_saved_args"]["vjepa"] == "pretrained" and digest(VJEPA_CKPT) != origin["encoder_weights_sha256"]:
        raise ValueError("Frozen encoder file differs")
    inventory = target_inventory(SPLITS, os.environ["SEER_ROOT"])
    base_model, backbone, refiner, exposure, arm, length, base_args = build_model(args.init, dev)
    check_base_args(base_args, arm, length)
    if refiner is not None or exposure is not None or base_model.eval_reconstruction != origin["base_reconstruction"]:
        raise ValueError("Base reconstruction changed")
    base_model.eval().requires_grad_(False)
    if base_state_digest(base_model.state_dict()) != origin["base_state_sha256"]:
        raise ValueError("Base state differs")
    raft = raft_small(weights=None).eval().requires_grad_(False)
    raft.load_state_dict(torch.load(args.raft_weights, map_location="cpu", weights_only=True), strict=True)
    raft.to(dev)
    if base_state_digest(raft.state_dict()) != origin["raft_state_sha256"]:
        raise ValueError("RAFT state differs")
    rows, diagnostics = [], []
    start = time.monotonic()
    with torch.no_grad():
        for name, low_u8, gt_u8, kind in held_out_sequences("bvi", "val", 16, 256):
            if kind != "video" or low_u8.shape != (16, 256, 256, 3) or gt_u8.shape != low_u8.shape:
                raise ValueError("Expected frozen center16/crop256 target clips")
            low = torch.from_numpy(low_u8.copy()).float().div_(255).permute(3, 0, 1, 2)[None].to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                base = base_model(low, backbone_tokens(backbone, low))["pred_pixels"].float().clamp(0, 1)
            pairs = [donor_correction(base, raft, delta, 12, 2) for delta in (-2, 2)]
            flow = torch.stack([p[1] for p in pairs], 1).cpu()
            masks = torch.stack([p[2] for p in pairs], 1).cpu()
            aflow, amask = adjacent_flows(base, raft)
            aflow, amask = aflow.cpu(), amask.cpu()
            low_cpu, base_cpu = low.cpu(), base.cpu()
            predictions = {"base": base_cpu}
            candidate_data = {}
            for candidate in CANDIDATES:
                d, mask, recipe = m["control"].build_observation_controls(low_cpu, base_cpu, flow, masks, mode=candidate)
                if recipe != meta["candidate_recipes"][candidate]:
                    raise ValueError("Target candidate recipe differs from source")
                x, g = m["selector"].observable_patch_features(low_cpu, base_cpu, d, mask, 8)
                candidate_data[candidate] = d, x, g
                predictions[f"{candidate}/initial_thirds"] = base_cpu + d.sum(1) / 3
                predictions[f"{candidate}/fixed_average"] = base_cpu + .5 * d.sum(1)
            lp = m["trajectory"].lowpass
            low_l, base_l = lp(low_cpu), lp(base_cpu)
            q = (base_l + (base_l / (low_l + 1e-4)).clamp(.25, 16) * (low_cpu - low_l)).clamp(0, 1)
            detail = q - lp(q)
            aux = {}
            for key, model in heads.items():
                candidate = training[key]["candidate"]
                d, x, _ = candidate_data[candidate]
                result = model(base_cpu, d, standardized(x, training[key]["normalization"]), detail, aflow, amask)
                predictions[key] = result["pred_pixels"]
                aux[key] = result["trajectory_aux"]
            d, x, g = candidate_data["raw_detail"]
            for seed, model in old.items():
                features = standardized(x, old_records[seed]["normalization"])[0]
                w = m["selector"].weights_from_head(model(features), g[0], "direct_mse", None, None)
                predictions[f"v1_raw_direct_s{seed}"] = base_cpu + sum(
                    expand_patch(w[..., j][None], 8) * d[:, j] for j in range(2))
            # Target GT enters only after every prediction has been constructed.
            scores = {key: score_prediction(prediction, gt_u8, dev) for key, prediction in predictions.items()}
            if not all(np.isfinite(v) for score in scores.values() for v in score.values()):
                raise FloatingPointError("Nonfinite target score")
            rows.append({"name": name, "methods": scores})
            diagnostics.append({"name": name, "heads": aux, "elapsed_seconds": time.monotonic() - start})
            print(json.dumps(rows[-1]), flush=True)
    if sorted(r["name"] for r in rows) != sorted(r["name"] for r in inventory):
        raise ValueError("Incomplete target VAL16 coverage")
    if (base_state_digest(base_model.state_dict()) != origin["base_state_sha256"]
            or base_state_digest(raft.state_dict()) != origin["raft_state_sha256"]
            or any(state_digest(model.state_dict()) != fingerprints[k] for k, model in heads.items())
            or any(state_digest(model.state_dict()) != old_fingerprints[k] for k, model in old.items())):
        raise AssertionError("Inference modified frozen model state")
    result = {"scope": "Exploratory SDSD-only observation/trajectory training to BVI VAL16; no target selection",
              "domain": "bvi", "split": "val", "n": 16, "per_sequence": rows,
              "means": {key: {metric: float(np.mean([r["methods"][key][metric] for r in rows]))
                               for metric in rows[0]["methods"][key]} for key in rows[0]["methods"]},
              "training": training, "checkpoint_sha256": {**head_hashes, **old_hashes},
              "prepared_manifest": meta, "prepared_manifest_sha256": manifest_sha,
              "v1_reference_training": old_records, "diagnostics": diagnostics,
              "validation_pair_inventory": inventory, "split_sha256": digest(SPLITS),
              "state_unchanged": {"base": True, "raft": True, "new_heads": True, "v1_heads": True},
              "source_sha256": source_hashes(*EXTRA_SOURCES, "tools/eval_observation_v2.py",
                  "tools/eval_source_selector.py", "tools/eval_correction_selector.py"),
              "scoring_library": {"path": str(Path(metric_lib.__file__).resolve()), "sha256": digest(metric_lib.__file__)},
              "runtime": {"torch": str(torch.__version__), "pyiqa": importlib.metadata.version("pyiqa"),
                          "graph_backend": "CPU float32", "metric_device": str(dev), "threads": 4},
              "protocol": "center16/crop256; rounded uint8 RGB/Y PSNR, LPIPS, per-sequence absolute tLP gap",
              "cost": "Shared base once and RAFT +/-2 plus extra +/-1; graph/candidate CPU cost is additional; no equal-latency claim",
              "limitations": "One historical base seed per arm; BVI VAL informed earlier development; no target-blind-history, novelty or SOTA conclusion"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
