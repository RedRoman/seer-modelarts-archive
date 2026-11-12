"""Preserve new compact HVI outputs only, with exclusive verified SFS writes."""
import argparse
import datetime
import json
from pathlib import Path
import tarfile
from cache_hvi_source import digest, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root, dest = args.root, args.destination
    assert json.loads((root / "checks_v1/outputs_verification.json").read_text())["verified"] is True
    dest.mkdir(parents=True, exist_ok=False)
    paths = []
    for name in ("cache_v1", "release_v1", "checks_v1", "logs_v1"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(root / name for name in ("run_source_v1.sh", "verify_outputs.py", "archive_source.py"))
    files = []
    for path in sorted(paths):
        target = dest / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        sha = digest(path)
        with path.open("rb") as source, target.open("xb") as output:
            for block in iter(lambda: source.read(1 << 20), b""):
                output.write(block)
        assert digest(target) == sha
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha})
    record = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "destination": str(dest), "files": files, "files_count": len(files),
        "total_bytes": sum(x["bytes"] for x in files), "all_copies_sha_verified": True,
        "contains_original_lq_or_gt": False, "quality_scores_computed": False}
    write_new(dest / "backup_manifest.json", record)
    archive = root / "text_evidence_v1.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for path in sorted(paths):
            if path.suffix not in (".pt", ".pth", ".bin"):
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
        tar.add(dest / "backup_manifest.json", arcname="backup_manifest.json", recursive=False)
    receipt = {"destination": str(dest), "files_count": len(files),
        "total_bytes": record["total_bytes"], "all_copies_sha_verified": True,
        "backup_manifest_sha256": digest(dest / "backup_manifest.json"),
        "text_archive_bytes": archive.stat().st_size, "text_archive_sha256": digest(archive)}
    write_new(root / "backup_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
