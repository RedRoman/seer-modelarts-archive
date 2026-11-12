"""Exclusive UTF-8 evidence backup; reads no dataset tensors or images."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--backup", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    a = p.parse_args()
    assert not a.backup.exists() and not a.archive.exists()
    a.backup.mkdir(parents=True)
    mapping, excluded_metadata = {}, {}
    # Preserve the complete frozen runtime source release in the SFS backup.
    for path in sorted((a.root / "release_v2").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            if path.name.startswith("._"):
                excluded_metadata[str(path)] = sha(path)
                continue
            mapping["release_v2/" + str(path.relative_to(a.root / "release_v2"))] = path
    for path in sorted((a.root / "release_v1/phase").glob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            mapping["failed_prepare_v1/" + path.name] = path
    for path in sorted(a.root.glob("*")):
        if path.is_file() and path.suffix in (".py", ".sh", ".json", ".log", ".txt"):
            mapping["checks/" + path.name] = path
    mapping["checks/none_fixed_fit_transfer_receipt.json"] = a.root / "inputs/none_fixed_fit/transfer_receipt.json"
    records = {}
    for relative, source in mapping.items():
        raw = source.read_bytes(); raw.decode("utf8")
        destination = a.backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as out:
            out.write(raw)
        expected = hashlib.sha256(raw).hexdigest()
        assert sha(destination) == expected
        records[relative] = {"source": str(source), "sha256": expected, "bytes": len(raw)}
    failure = {"stage": "freeze_release_before_forward", "exception": "FileNotFoundError",
        "missing_path": "/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/transfers/none_cache_payload_v1/cache/clip_108.pt",
        "resolution": "V2 copied fixed original 108/109 N clips from producer 32193; old SHA verified",
        "real_forward_before_failure": False, "target_pixels_before_failure": False,
        "note": "This is a retrospective failure record, not reconstructed original stderr."}
    (a.backup / "failed_prepare_v1/failure_record.json").write_text(json.dumps(failure, indent=2) + "\n")
    manifest = {"scope": "Exclusive final UTF-8 code/metadata/results evidence; no dataset/prediction tensors",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": records, "excluded_nonexecutable_appledouble_metadata_sha256": excluded_metadata,
        "exclusive_directory": str(a.backup), "source_sha256": sha(__file__)}
    (a.backup / "backup_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Download only new phase source and textual evidence; existing code_full
    # sources remain in SFS and are bound by the complete release manifest.
    selected = []
    for path in sorted(a.backup.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(a.backup))
            if rel.startswith(("release_v2/code_full/", "release_v2/baselines/")):
                continue
            selected.append((path, rel))
    with tarfile.open(a.archive, "x") as archive:
        for path, relative in selected:
            archive.add(path, arcname=relative, recursive=False)
    result = {"backup_manifest_sha256": sha(a.backup / "backup_manifest.json"),
        "backup_files": len(records), "archive_sha256": sha(a.archive),
        "archive_bytes": a.archive.stat().st_size, "archive_files": len(selected),
        "all_utf8_verified": True, "no_binary_tensors": True}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
