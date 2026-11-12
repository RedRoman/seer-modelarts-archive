"""Canonical-RGB fixed-seven HVI diagnostic, restricted to source FIT51/102.

Reuse the pinned previous diagnostic's prediction/statistic arithmetic. New HVI
values are already RGB; only legacy source GT is converted from declared BGR.
All excluded tensor archives remain unopened, including for file hashing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time

import numpy as np

PHASE = Path(__file__).resolve().parent
MATH_SHA = "c314d9b37f71009af0f4974cf59b3bcc41b0d48698184c9ec01de66a7caecd54"
COLOR_SHA = "7815887150b2e92ed7c2cf4c30f334a4a0c435d58b4ba017fd1b746a97dcab60"
SOURCE_SHA = "0fe83dc30cd7772eb4aa8326deb8911e5b5538ae63470821c1c5aa21129d19f0"
SUMMARY_SHA = "63df74b887dc66543137569d58b6fd57c6fbdb931c6f4c6077c53b77585031ae"
WEIGHT_SHA = "6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27"
STATE_SHA = "afa01e29d1314dae7fadc68d7c95e3ddaf3da3a5b469fa83febb2759dc0bc41b"
RECEIPT_SHA = "fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70"
GEOMETRY_SHA = "d7f90214ab2da0b30da9f3f5b9fba648bb4fcabfee548e34dd7788caa6610805"
BOUNDARY = {"source_cache": "BGR", "hvi_input": "RGB", "hvi_output": "RGB",
            "input_permutation": [2, 1, 0], "output_permutation": [0, 1, 2]}
METHODS = ("hvi", "temporal_mean_lambda_0.25", "temporal_mean_lambda_0.5",
           "temporal_mean_lambda_1", "sameframe_identity_lambda_0.25",
           "sameframe_identity_lambda_0.5", "sameframe_identity_lambda_1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def module_at(name, path, expected):
    require(digest(path) == expected, "Frozen dependency SHA differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dependencies(math_source=None):
    path = Path(math_source) if math_source else PHASE.parent / "hvi_temporal_fit_feasibility_v1/run_fit_feasibility.py"
    math = module_at("_rgb_fit_frozen_math", path, MATH_SHA)
    color = module_at("_rgb_fit_color_boundary", PHASE / "color_contract.py", COLOR_SHA)
    require(tuple(math.METHODS) == METHODS and tuple(math.LAMBDAS) == (.25, .5, 1.),
            "Fixed seven diagnostic conditions changed")
    return math, color, path


def selected_fit_rows(rows, split):
    fit, excluded = set(split["fit"]), set(split["internal_holdout"])
    require(not fit & excluded, "FIT and excluded scenes overlap")
    return [row for row in rows if row["scene"] in fit]


def check_color_metadata(hvi):
    require(hvi.get("complete") is True and hvi.get("input_color") == "RGB"
            and hvi.get("native_source_color") == "BGR" and hvi.get("output_color") == "RGB"
            and hvi.get("channel_boundary") == BOUNDARY,
            "Require completed explicitly canonical RGB HVI export")


def check_bgr_evidence(path, expected_sha):
    require(digest(path) == expected_sha == RECEIPT_SHA, "Raw BGR closure receipt SHA differs")
    evidence = json.loads(Path(path).read_text())
    require(evidence.get("verified") is True and evidence.get("native_source_color") == "BGR"
            and evidence.get("status") == "closed" and evidence.get("source_provenance_sha256") == SOURCE_SHA
            and evidence.get("raw_disk_byte_identity_verified") is True,
            "Require closed raw-archive BGR contract for this source cache")
    require(len(evidence.get("proof_files", [])) == 5 and all(
        digest(p["path"]) == p["sha256"] for p in evidence["proof_files"]), "BGR proof-file identity differs")
    return evidence


def bound_inputs(source_cache, hvi_cache, geometry_manifest, expected_hvi_manifest_sha):
    """Validate metadata globally; never open excluded-scene tensor archives."""
    source_cache, hvi_cache = Path(source_cache), Path(hvi_cache)
    require(digest(source_cache / "provenance.json") == SOURCE_SHA, "Wrong legacy source provenance")
    require(digest(source_cache / "cache_summary.json") == SUMMARY_SHA, "Wrong legacy source summary")
    require(digest(geometry_manifest) == GEOMETRY_SHA, "Wrong fixed geometry manifest")
    require(digest(hvi_cache / "manifest.json") == expected_hvi_manifest_sha,
            "HVI manifest differs from the prescore frozen identity")
    origin = json.loads((source_cache / "provenance.json").read_text())
    summary = json.loads((source_cache / "cache_summary.json").read_text())
    hvi = json.loads((hvi_cache / "manifest.json").read_text())
    geometry = json.loads(Path(geometry_manifest).read_text())
    check_color_metadata(hvi)
    check_bgr_evidence(hvi["bgr_evidence_file"], hvi["bgr_evidence_sha256"])
    require(geometry["source_cache_provenance"] == origin and geometry["n"] == 136,
            "Geometry manifest does not bind the original source")
    require(hvi["n"] == 136 and hvi["scenes"] == 68 and hvi["model_state_unchanged"] is True
            and hvi["weight_sha256"] == WEIGHT_SHA and hvi["model_state_sha256"] == STATE_SHA
            and hvi["source_provenance_sha256"] == SOURCE_SHA
            and hvi["source_summary_sha256"] == SUMMARY_SHA
            and hvi["gt_storage_entries_read"] == 0 and hvi["target_inputs_read"] == 0
            and hvi["quality_scores_computed"] is False, "Invalid GT-free HVI RGB cache lineage")
    sr, hr, gr = summary["per_sequence"], hvi["per_sequence"], geometry["per_sequence"]
    names = origin["source_names"]
    require(len(sr) == len(hr) == len(gr) == len(names) == len(set(names)) == 136,
            "Incomplete source coverage")
    require([r["name"] for r in sr] == [r["name"] for r in hr] == [r["name"] for r in gr] == names,
            "Metadata sequence order differs")
    entries = []
    for i, (s, h, g) in enumerate(zip(sr, hr, gr)):
        require(s["scene"] == h["scene"] == g["scene"]
                and s["cache_path"] == f"clip_{i:03d}.pt" and h["file"] == f"hvi_rgb_{i:03d}.pt"
                and h["source_cache_sha256"] == s["cache_sha256"] == g["source_sha256"]
                and h["shape"] == [1, 3, 16, 256, 256] and h["dtype"] == "float32"
                and h["finite_range_verified"] is True, "Per-clip RGB cache lineage differs")
        sp, hp = source_cache / s["cache_path"], hvi_cache / h["file"]
        require(sp.resolve() == Path(g["source_path"]).resolve(), "Legacy source location differs")
        entries.append({"name": s["name"], "scene": s["scene"], "source_path": str(sp.resolve()),
            "source_sha256": s["cache_sha256"], "hvi_path": str(hp.resolve()), "hvi_sha256": h["file_sha256"],
            "native_lq_tensor_sha256": h["native_lq_tensor_sha256"],
            "rgb_lq_tensor_sha256": h["rgb_lq_tensor_sha256"], "base_tensor_sha256": h["base_tensor_sha256"]})
    split = geometry["split"]
    scenes = sorted({r["scene"] for r in entries})
    require(split["seed"] == 20261114 and len(scenes) == 68 and len(split["fit"]) == 51
            and len(split["internal_holdout"]) == 17, "Wrong fixed source scene split")
    order = np.random.default_rng(20261114).permutation(68)
    require(split["fit"] == [scenes[int(i)] for i in order[:51]]
            and split["internal_holdout"] == [scenes[int(i)] for i in order[51:]], "Scene split changed")
    require(set(names) == {n for scene in scenes for n in (scene, scene + "_2")},
            "Require exactly the paired passes of each source scene")
    fit = selected_fit_rows(entries, split)
    require(len(fit) == 102 and set(Counter(r["scene"] for r in fit).values()) == {2}, "Wrong FIT coverage")
    lineage = {"source_provenance_sha256": SOURCE_SHA, "source_summary_sha256": SUMMARY_SHA,
        "hvi_manifest_path": str((hvi_cache / "manifest.json").resolve()),
        "hvi_manifest_sha256": expected_hvi_manifest_sha, "hvi_weight_sha256": WEIGHT_SHA,
        "hvi_state_sha256": STATE_SHA, "bgr_evidence_path": hvi["bgr_evidence_file"],
        "bgr_evidence_sha256": hvi["bgr_evidence_sha256"], "channel_boundary": BOUNDARY,
        "geometry_manifest_path": str(Path(geometry_manifest).resolve()),
        "geometry_manifest_sha256": digest(geometry_manifest), "source_raft_state_sha256": origin["raft_state_sha256"],
        "flow_provenance": "Unchanged legacy Seer-native restored-array RAFT +/-2 proposals; not recomputed from RGB HVI",
        "excluded_tensor_archives_opened": 0}
    return entries, split, lineage


def load_fit_sample(row, fit_scenes, color):
    """The scene guard precedes even digesting archives containing GT storage."""
    require(row["scene"] in set(fit_scenes), "Refusing excluded scene tensor/label access")
    require(digest(row["source_path"]) == row["source_sha256"]
            and digest(row["hvi_path"]) == row["hvi_sha256"], "FIT tensor archive SHA differs")
    import torch
    source = torch.load(row["source_path"], map_location="cpu", weights_only=True, mmap=True)
    hvi = torch.load(row["hvi_path"], map_location="cpu", weights_only=True, mmap=True)
    require(set(hvi) == {"name", "scene", "base", "lineage"}
            and all(v["name"] == row["name"] and v["scene"] == row["scene"] for v in (source, hvi)),
            "FIT payload identity differs")
    native, meta = source["tensors"], hvi["lineage"]
    require(meta["source_cache_sha256"] == row["source_sha256"]
            and meta["source_provenance_sha256"] == SOURCE_SHA
            and meta["weight_sha256"] == WEIGHT_SHA and meta["model_state_sha256"] == STATE_SHA
            and meta["channel_boundary"] == BOUNDARY and meta["input_reencoded_to_uint8"] is False,
            "FIT RGB payload provenance differs")
    rgb_low = color.convert_video(native["lq"], source_order="BGR", target_order="RGB")
    require(tensor_digest(native["lq"]) == row["native_lq_tensor_sha256"] == meta["native_lq_tensor_sha256"]
            and tensor_digest(rgb_low) == row["rgb_lq_tensor_sha256"] == meta["rgb_lq_tensor_sha256"]
            and tensor_digest(hvi["base"]) == row["base_tensor_sha256"] == meta["base_tensor_sha256"],
            "FIT canonical input/output tensor SHA differs")
    # HVI was already evaluated with RGB input; do NOT permute its output again.
    base = hvi["base"]
    gt = color.convert_video(native["gt"], source_order="BGR", target_order="RGB")
    require(base.shape == gt.shape == (1, 3, 16, 256, 256)
            and base.dtype == gt.dtype == torch.float32
            and all(bool(torch.isfinite(v).all() & (v >= 0).all() & (v <= 1).all()) for v in (base, gt)),
            "Invalid canonical RGB base/GT")
    return (base, native["flow"], native["valid_mask"]), gt


def evaluate_fit_rows(rows, split, loader, math):
    scores = []
    for row in selected_fit_rows(rows, split):
        inputs, labels = loader(row, split["fit"])
        predictions, coverage = math.fixed_predictions(*inputs)
        scores.append({"name": row["name"], "scene": row["scene"],
            "methods": math.score_predictions(predictions, labels), "coverage": coverage,
            "sameframe_exact_anchor": True, "gt_color": "RGB", "hvi_output_color": "RGB"})
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-cache", "hvi-cache", "geometry-manifest", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--expected-hvi-manifest-sha256", required=True)
    parser.add_argument("--math-source", type=Path)
    args = parser.parse_args()
    require(args.out.is_absolute() and args.out.resolve().is_relative_to(Path("/cache"))
            and not args.out.exists(), "Use a new absolute /cache output directory")
    math, color, math_path = load_dependencies(args.math_source)
    protocol_path = PHASE / "rgb_fit_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    require(protocol["fixed_methods"] == list(METHODS) and protocol["legacy_gt_color"] == "BGR"
            and protocol["scoring_color"] == "RGB" and protocol["lambda_selection"] is False,
            "Frozen diagnostic protocol differs")
    rows, split, lineage = bound_inputs(args.source_cache, args.hvi_cache, args.geometry_manifest,
                                       args.expected_hvi_manifest_sha256)
    import torch
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {str(p.resolve()): digest(p) for p in (Path(__file__), PHASE / "test_rgb_fit.py",
        protocol_path, PHASE / "color_contract.py", math_path)}
    manifest = {"schema_version": 1, "scope": "Canonical RGB SDSD FIT51/102 fixed diagnostic only",
        "protocol": protocol, "source_sha256": sources, "lineage": lineage, "split": split,
        "fit_names": [r["name"] for r in selected_fit_rows(rows, split)],
        "prediction_helpers_have_no_gt_parameter": not ({"gt", "labels", "fit_labels"}
            & set(inspect.signature(math.fixed_predictions).parameters)),
        "internal_holdout_tensor_archives_opened": 0, "internal_holdout_scores_computed": False,
        "target_inputs_read": False, "optimizer_or_fitting": False, "prediction_or_gt_arrays_written": False,
        "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__), "threads": 4, "device": "cpu"},
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(args.out / "execution_manifest.json", manifest)
    started = time.monotonic()
    with torch.no_grad():
        scores = evaluate_fit_rows(rows, split, lambda r, f: load_fit_sample(r, f, color), math)
    require(len(scores) == 102 and len({r["scene"] for r in scores}) == 51,
            "Incomplete FIT diagnostic")
    require(all(digest(path) == sha for path, sha in sources.items()), "Diagnostic source changed during execution")
    require(digest(args.hvi_cache / "manifest.json") == args.expected_hvi_manifest_sha256,
            "HVI manifest changed during execution")
    report = {"schema_version": 1, "scope": manifest["scope"],
        "manifest_sha256": digest(args.out / "execution_manifest.json"),
        "metric": "float32 squared canonical-RGB residual; float64 full16x256 mean; per-clip PSNR",
        "per_sequence": scores, **math.aggregate(scores), "elapsed_seconds": time.monotonic() - started,
        "internal_holdout_scores_computed": False, "target_scores_computed": False,
        "interpretation": "All seven fixed FIT rows; no lambda selection, learned improvement or transfer claim"}
    write_new(args.out / "fit_results.json", report)
    print(json.dumps({"completed_fit_clips": len(scores), "fit_scenes": 51,
                      "results_sha256": digest(args.out / "fit_results.json")}), flush=True)


if __name__ == "__main__":
    main()
