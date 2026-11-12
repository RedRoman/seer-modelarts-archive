"""Exclusive, verified SFS preservation; never rename or overwrite shared files."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root, dest = args.root, args.destination
    assert json.loads((root / "checks_v1/all_done.json").read_text())["methods"] == 3
    dest.mkdir(parents=True, exist_ok=False)
    paths = []
    for directory in ("release_v1", "smoke_v1", "logs_v1", "results_v1", "checks_v1"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(root / name for name in ("run_replay_v1.sh", "verify_replay.py", "archive_replay.py"))
    entries = []
    for source in sorted(paths):
        rel = source.relative_to(root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(source)
        with source.open("rb") as inp, target.open("xb") as out:
            for block in iter(lambda: inp.read(1 << 20), b""):
                out.write(block)
        assert digest(target) == expected
        entries.append({"path": str(rel), "bytes": source.stat().st_size, "sha256": expected})
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_root": str(root), "destination": str(dest), "files": entries,
        "files_count": len(entries), "total_bytes": sum(x["bytes"] for x in entries),
        "all_copies_verified": True, "contains_original_images": False}
    with (dest / "backup_manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    text_archive = root / "text_and_sources_v1.tar.gz"
    with tarfile.open(text_archive, "x:gz") as archive:
        for source in sorted(paths):
            if source.suffix not in (".pt", ".pth", ".bin"):
                archive.add(source, arcname=str(source.relative_to(root)), recursive=False)
        archive.add(dest / "backup_manifest.json", arcname="backup_manifest.json", recursive=False)
    receipt = {"destination": str(dest), "files_count": len(entries),
        "total_bytes": manifest["total_bytes"], "all_copies_verified": True,
        "backup_manifest_sha256": digest(dest / "backup_manifest.json"),
        "local_text_archive": str(text_archive), "archive_bytes": text_archive.stat().st_size,
        "archive_sha256": digest(text_archive)}
    with (root / "backup_receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
