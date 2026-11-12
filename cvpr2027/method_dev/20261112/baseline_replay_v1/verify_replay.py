"""Separate NumPy-only verification of baseline coverage, means and input bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--reference-inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    raw = json.loads(args.reference_raw.read_text())
    names = [r["name"] for r in raw["per_sequence"]]
    refs = {r["name"]: r["methods"]["base"] for r in raw["per_sequence"]}
    fingerprints = json.loads(args.reference_inputs.read_text())
    inputs = {r["name"]: r["roles"] for r in fingerprints["content"]["clips"]}
    assert len(names) == len(set(names)) == 16
    summary, checked, maximum_error = {}, 0, 0.0
    for method in ("hvi_cidnet", "darkir", "retinexformer"):
        path = args.results_dir / (method + "_bvi_val16.json")
        result = json.loads(path.read_text())
        rows = result["per_sequence"]
        assert [r["name"] for r in rows] == names and result["n"] == 16
        assert result["state_unchanged"] is True
        assert result["provenance"]["strict_load"] is True
        assert result["split_sha256"] == fingerprints["split_sha256"]
        assert result["scoring_source_sha256"]["eval_lib"] == "a26ff0c81f0e6fa380ca26ac87dc2197c66248d3b6945e0da1c99e5b3d3f451d"
        assert [r["name"] for r in result["inputs"]] == names
        for sample in result["inputs"]:
            assert sample["shape"] == [16, 256, 256, 3] and sample["dtype"] == "uint8"
            for role in ("lq", "gt"):
                assert sample[role + "_sha256"] == inputs[sample["name"]][role]["scoring_tensor_sha256"]
                checked += 1
        for row in rows:
            assert row["gt_tlp"] == refs[row["name"]]["gt_tlp"]
            assert row["tlp_gap"] == abs(row["tlp"] - row["gt_tlp"])
            checked += 2
        for metric, recorded in result["means"].items():
            recomputed = float(np.asarray([r[metric] for r in rows], dtype=np.float64).sum() / 16)
            assert np.isfinite(recomputed)
            error = abs(recomputed - recorded)
            maximum_error = max(maximum_error, error)
            assert error < 1e-13
            checked += 1
        summary[method] = {"result_sha256": digest(path), "means": result["means"],
                           "model_device": result["runtime"]["model_device"],
                           "elapsed_seconds": result["elapsed_seconds"]}
    report = {"verified": True, "methods": 3, "sequences_per_method": 16,
        "whole_scenes": 8, "checked_scalars_and_hashes": checked,
        "maximum_aggregate_error": maximum_error,
        "all_96_actual_input_tensors_match_prior_frozen_audit": True,
        "all_48_gt_temporal_values_exact": True,
        "reference_raw_sha256": digest(args.reference_raw),
        "reference_input_manifest_sha256": digest(args.reference_inputs),
        "verifier_sha256": digest(__file__), "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
