"""Independent receipt-only source gate; no Torch, pixels or predictions."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--none", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    paths = {"pretrained": a.pretrained, "none": a.none}
    rows = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    assert len({row["protocol_sha256"] for row in rows.values()}) == 1
    assert len({row["release_manifest_sha256"] for row in rows.values()}) == 1
    for arm, row in rows.items():
        assert row["passed"] and row["stage"] == "source" and row["arm"] == arm
        assert row["state_unchanged"] and row["n"] == 2 and row["scene"] == "SDSD_outdoor/MVI_0975"
        assert row["gt_storage_reads"] == row["target_pixel_reads"] == 0
        assert row["quality_scores_computed"] is False
        assert [clip["index"] for clip in row["per_sequence"]] == [108, 109]
        assert row["model"]["strict_pixel_keys"]
        for load in row["model"]["load_key_receipts"]:
            assert load["missing"] == []
            assert not set(load["unexpected"]) - {"patch_embed_img.proj.weight", "patch_embed_img.proj.bias"}
        for clip in row["per_sequence"]:
            assert clip["wrapper_input_exact"] and clip["wrapper_output_permutation_exact"]
            diff = clip["old_base_parity"]
            assert diff["passed"] and diff["max_abs_limit"] == 1e-4 and diff["mean_abs_limit"] == 1e-5
            assert 0 <= diff["mean_abs"] <= diff["max_abs"] <= 1e-4 and diff["mean_abs"] <= 1e-5
    for left, right in zip(rows["pretrained"]["per_sequence"], rows["none"]["per_sequence"]):
        assert left["name"] == right["name"]
        assert left["native_lq_sha256"] == right["native_lq_sha256"]
    result = {"passed": True, "scope": "receipt-only independent verification; four fixed FIT predictions",
        "protocol_sha256": rows["none"]["protocol_sha256"],
        "source_parity_sha256": {arm: sha(path) for arm, path in paths.items()},
        "lq_shared_exact": True, "target_pixel_reads": 0, "gt_storage_reads": 0,
        "verification_source_sha256": sha(__file__)}
    with a.out.open("x") as out:
        json.dump(result, out, indent=2); out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
