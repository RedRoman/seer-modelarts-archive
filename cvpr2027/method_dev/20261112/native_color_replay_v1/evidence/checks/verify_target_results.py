"""Independent scalar/input-hash verification, without pixels or new scoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline-inputs", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    results = {arm: json.loads((root / f"target_{arm}_v2.json").read_text()) for arm in ("pretrained", "none")}
    checks, errors = 0, []
    names = None
    for arm, result in results.items():
        assert result["passed"] and result["stage"] == "target" and result["arm"] == arm
        assert result["state_unchanged"] and result["n"] == 16
        assert result["gt_mean_alignment"] is False and result["selection"] == "none"
        assert result["predicted_rgb_saved"] is False
        assert result["protocol_sha256"] == "3423db39aa7792557320f3836f5dbf31b39069db19294c9485bc040898e29be4"
        these = [row["name"] for row in result["per_sequence"]]
        assert len(these) == len(set(these)) == 16
        if names is not None: assert names == these
        names = these
        for key, reported in result["means"].items():
            values = [row["metrics"][key] for row in result["per_sequence"]]
            assert all(math.isfinite(value) for value in values)
            error = abs(math.fsum(values) / 16 - reported)
            assert error <= 1e-12
            errors.append(error); checks += 1
        for row in result["per_sequence"]:
            metrics = row["metrics"]
            assert abs(abs(metrics["tlp"] - metrics["gt_tlp"]) - metrics["tlp_gap"]) <= 1e-12
            assert row["low_shape"] == [1, 3, 16, 256, 256]
            assert row["score_gt_shape"] == [16, 3, 256, 256]
            checks += 3
    for first, second in zip(results["pretrained"]["per_sequence"], results["none"]["per_sequence"]):
        for key in ("low_u8_sha256", "gt_u8_sha256", "low_rgb_tensor_sha256", "score_gt_float_tensor_sha256"):
            assert first[key] == second[key], (first["name"], key)
            checks += 1
        assert abs(first["metrics"]["gt_tlp"] - second["metrics"]["gt_tlp"]) <= 1e-10
        checks += 1
    manifest = json.loads((root / "metric_dependencies_v1.json").read_text())
    for path, expected in manifest["files"].items():
        assert sha(path) == expected
    output = {"passed": True, "checks": checks, "max_scalar_mean_error": max(errors),
        "cross_arm_input_hashes_exact": True, "metric_sources_and_weights_unchanged": True,
        "result_sha256": {arm: sha(root / f"target_{arm}_v2.json") for arm in results},
        "source_sha256": sha(__file__), "no_pixels_read": True, "no_scores_recomputed": True,
        "means": {arm: result["means"] for arm, result in results.items()}}
    # Optional baseline receipt schema is explicitly supplied by its owner and
    # audited separately; no guessing of heterogeneous hash tensor layouts.
    if a.baseline_inputs:
        output["baseline_receipt_sha256"] = sha(a.baseline_inputs)
    with a.out.open("x") as stream:
        json.dump(output, stream, indent=2); stream.write("\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
