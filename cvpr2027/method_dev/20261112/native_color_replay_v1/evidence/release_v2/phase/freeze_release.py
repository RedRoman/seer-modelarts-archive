"""Freeze dependencies without reading target pixels or launching prediction."""
import argparse
import json
from pathlib import Path
import shutil
from replay_native import digest, read_json, write_new, SHAS, SPLIT_SHA, ENCODER_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    release = args.release.resolve()
    config = read_json(release / "phase/config.json")
    root = Path("/temp/zihaohe/seer_cvpr")
    common = root / "code/baselines/common"
    for source in sorted(common.glob("*.py")):
        target = release / "baselines/common" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(source, target)
    externals = {}
    for arm, entry in config["arms"].items():
        checkpoint = Path(entry["checkpoint"])
        if digest(checkpoint) != SHAS[arm]:
            raise ValueError("Fixed final checkpoint differs")
        externals[str(checkpoint)] = SHAS[arm]
        cache = Path(entry["cache"])
        for name in ("provenance.json", "cache_summary.json"):
            externals[str(cache / name)] = digest(cache / name)
        summary = read_json(cache / "cache_summary.json")
        for index in (108, 109):
            row = summary["per_sequence"][index]
            path = cache / row["cache_path"]
            if digest(path) != row["cache_sha256"]:
                raise ValueError("Fixed FIT cache bytes differ")
            externals[str(path)] = row["cache_sha256"]
    for relative, sha in (("datasets/splits_cvpr.json", SPLIT_SHA),
                          ("weights/vjepa2_1_vitl_dist_vitG_384.pt", ENCODER_SHA)):
        path = root / relative
        if digest(path) != sha:
            raise ValueError("Split/encoder differs")
        externals[str(path)] = sha
    # Freeze every Python source in the local upstream checkout, beyond the
    # smaller transitive import set. This reads code only, never dataset pixels.
    for path in sorted((root / "modelzoo/vjepa2-main").rglob("*.py")):
        if ".git" not in path.parts:
            externals[str(path)] = digest(path)
    runtime = root / "method_dev_20261111/method_runtime_env.sh"
    externals[str(runtime)] = digest(runtime)
    metric_weight = Path("/home/ma-user/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth")
    if not metric_weight.is_file():
        raise FileNotFoundError(metric_weight)
    externals[str(metric_weight)] = digest(metric_weight)
    files = {}
    for top in ("phase", "code_full", "baselines"):
        for path in sorted((release / top).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(release))] = digest(path)
    manifest = {"schema_version": 1, "files": files, "external_files": externals,
        "target_pixel_reads": 0, "predictions_computed": 0,
        "protocol_sha256": digest(release / "phase/PLAN.md"),
        "dependency_scope": "all release Python/data metadata; all upstream V-JEPA Python sources; fixed weights/split/FIT cache files"}
    write_new(release / "release_manifest.json", manifest)
    print(json.dumps({"release_manifest_sha256": digest(release / "release_manifest.json"),
        "protocol_sha256": manifest["protocol_sha256"], "release_files": len(files), "external_files": len(externals)}))


if __name__ == "__main__":
    main()
