"""Back up final CVPR checkpoints and small evidence without copying datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--host", required=True)
    args = ap.parse_args()
    base = args.root.resolve()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"seer_converged_{args.host}_20261111.tar"
    if archive.exists():
        raise SystemExit(f"refusing to replace existing archive: {archive}")
    files = []
    for group in ("runs_e17", "runs_e18", "runs_e20", "runs_e26"):
        for run in sorted((base / group).glob("*")):
            if not run.is_dir():
                continue
            checkpoint = run / "final_cpu.pt"
            if not checkpoint.is_file():
                checkpoint = run / "final.pt"
            if checkpoint.is_file():
                files.append(checkpoint)
            files.extend(sorted(run.glob("*.log")))
    for relative in ("results", "qual_regen"):
        files.extend(sorted(p for p in (base / relative).rglob("*") if p.is_file()))
    files = sorted(set(files))
    manifest = {
        "source_host": args.host,
        "source_root": str(base),
        "created_unix": time.time(),
        "selection": "Prefer final_cpu.pt over final.pt; include run logs, results, qual_regen.",
        "files": [
            {"path": str(p.relative_to(base)), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in files
        ],
    }
    manifest_path = dest / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary = archive.with_suffix(".tar.partial")
    with tarfile.open(temporary, "w") as tar:
        tar.add(manifest_path, arcname="artifact_manifest.json", recursive=False)
        for path in files:
            tar.add(path, arcname=str(path.relative_to(base)), recursive=False)
    os.replace(temporary, archive)
    summary = {
        "archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": digest(archive), "file_count": len(files),
        "checkpoint_count": sum(p.suffix == ".pt" for p in files),
    }
    (dest / "archive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
