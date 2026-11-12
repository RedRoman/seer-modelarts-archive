"""Verify only the exported HVI outputs and observable lineage; no original pixels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from cache_hvi_source import (STATE_SHA, WEIGHT_SHA, SOURCE_PROVENANCE_SHA,
                              digest, tensor_digest, write_new)


def contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(contains_tensor(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(contains_tensor(v) for v in value)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    source = json.loads(args.source_summary.read_text())
    assert manifest["n"] == len(manifest["per_sequence"]) == 136
    assert manifest["scenes"] == len({x["scene"] for x in manifest["per_sequence"]}) == 68
    assert manifest["model_state_unchanged"] is True and manifest["model_state_sha256"] == STATE_SHA
    assert manifest["weight_sha256"] == WEIGHT_SHA
    assert manifest["source_provenance_sha256"] == SOURCE_PROVENANCE_SHA
    assert manifest["source_summary_sha256"] == digest(args.source_summary)
    assert manifest["quality_scores_computed"] is False and manifest["gt_storage_entries_read"] == 0
    assert [x["name"] for x in manifest["per_sequence"]] == [x["name"] for x in source["per_sequence"]]
    expected_files = {x["file"] for x in manifest["per_sequence"]}
    assert {p.name for p in args.cache.glob("*.pt")} == expected_files
    checked = []
    for row, original in zip(manifest["per_sequence"], source["per_sequence"]):
        path = args.cache / row["file"]
        assert digest(path) == row["file_sha256"]
        payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
        assert set(payload) == {"name", "scene", "base", "lineage"}
        assert payload["name"] == row["name"] == original["name"]
        assert payload["scene"] == row["scene"] == original["scene"]
        base, lineage = payload["base"], payload["lineage"]
        assert base.dtype == torch.float32 and tuple(base.shape) == (1, 3, 16, 256, 256)
        assert torch.isfinite(base).all() and base.min() >= 0 and base.max() <= 1
        assert tensor_digest(base) == row["base_tensor_sha256"] == lineage["base_tensor_sha256"]
        assert lineage["source_cache_sha256"] == original["cache_sha256"] == row["source_cache_sha256"]
        assert lineage["lq_tensor_sha256"] == row["lq_tensor_sha256"]
        assert lineage["weight_sha256"] == WEIGHT_SHA and lineage["model_state_sha256"] == STATE_SHA
        assert lineage["source_provenance_sha256"] == SOURCE_PROVENANCE_SHA
        assert lineage["config_sha256"] == digest(args.release / "config.json")
        assert lineage["release_manifest_sha256"] == digest(args.release / "release_manifest.json")
        assert lineage["input_reencoded_to_uint8"] is False
        assert not contains_tensor(lineage)
        storage_entries = [name for name in lineage["archive_entries_read"] if "/data/" in name]
        assert storage_entries == [lineage["lq_storage_entry"]]
        checked.append({"name": row["name"], "file": row["file"], "file_sha256": row["file_sha256"],
                        "bytes": path.stat().st_size, "shape_dtype_range_hash_and_lineage_verified": True})
    report = {"verified": True, "n": len(checked), "scenes": 68,
        "per_sequence": checked, "total_output_bytes": sum(x["bytes"] for x in checked),
        "manifest_sha256": digest(args.cache / "manifest.json"),
        "source_original_tensor_storages_read_by_verifier": 0,
        "scores_computed": False, "verifier_sha256": digest(__file__)}
    write_new(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_sequence"}, indent=2))


if __name__ == "__main__":
    main()
