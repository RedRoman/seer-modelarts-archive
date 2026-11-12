"""Hash LPIPS dependencies before target authorization; no image/score access."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(importlib.util.find_spec("pyiqa").origin).parent
    paths = list(root.rglob("*.py"))
    paths += [Path("/home/ma-user/.cache/torch/hub/pyiqa/LPIPS_v0.1_alex-df73285e.pth"),
              Path("/home/ma-user/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth")]
    result = {"scope": "pre-target LPIPS code/weights hash supplement; no images, predictions or scores",
        "files": {str(path): sha(path) for path in sorted(paths)},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "pyiqa", "torchvision", "numpy")},
        "source_sha256": sha(__file__), "target_pixel_reads": 0, "quality_scores_computed": False}
    with args.out.open("x") as out:
        json.dump(result, out, indent=2); out.write("\n")
    print(json.dumps({"manifest_sha256": sha(args.out), "files": len(result["files"])}))


if __name__ == "__main__":
    main()
