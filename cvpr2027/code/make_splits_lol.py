"""Freeze the LOL splits for E4, before any LOL training run exists.

The downloaded releases carry the canonical train/test partitions used in the
literature (LOL-v1 our485/eval15, LOLv2-Real 689/100, LOLv2-Synthetic 900/100),
verified by file count, so published baseline numbers stay comparable. The only
choice left is the validation split, and it follows the same rule the SDSD splits
use: every 10th name in sorted order, held out of training.

Writes datasets/splits_lol.json. Test lists are recorded but never read by the
training scripts.
"""
import json
import os

ROOT = os.environ.get("SEER_ROOT", "/temp/zihaohe/seer_cvpr")
LOL = f"{ROOT}/datasets/lol"

SPEC = {
    "LOL_v1": {
        "root": f"{LOL}/lol_v1/lol_dataset",
        "train": ("our485/low", "our485/high"),
        "test": ("eval15/low", "eval15/high"),
    },
    "LOLv2_real": {
        "root": f"{LOL}/lolv2_real",
        "train": ("Train/Input", "Train/GT"),
        "test": ("Test/Input", "Test/GT"),
    },
    "LOLv2_syn": {
        "root": f"{LOL}/lolv2_syn",
        "train": ("Train/Input", "Train/GT"),
        "test": ("Test/Input", "Test/GT"),
    },
}
EXT = (".png", ".jpg", ".jpeg", ".bmp")


def names_in(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXT))


out = {"doc": ("LOL splits for E4. Canonical train/test from the public releases; "
               "val = every 10th training name in sorted order, removed from training. "
               "Test lists recorded for the single final evaluation only.")}
for key, spec in SPEC.items():
    root = spec["root"]
    tr_lq, tr_gt = spec["train"]
    te_lq, te_gt = spec["test"]
    tr = names_in(os.path.join(root, tr_lq))
    gt = set(names_in(os.path.join(root, tr_gt)))
    missing = [n for n in tr if n not in gt]
    if missing:
        raise RuntimeError(f"{key}: {len(missing)} training inputs have no GT, "
                           f"e.g. {missing[:3]}")
    val = tr[::10]
    train = [n for n in tr if n not in set(val)]
    te = names_in(os.path.join(root, te_lq))
    out[key] = {
        "root": root,
        "splits": {
            "train": {"lq": tr_lq, "gt": tr_gt, "names": train},
            "val": {"lq": tr_lq, "gt": tr_gt, "names": val},
            "test": {"lq": te_lq, "gt": te_gt, "names": te},
        },
    }
    print(f"{key:<12} train {len(train):4d}  val {len(val):3d}  test {len(te):3d}")

dst = f"{ROOT}/datasets/splits_lol.json"
json.dump(out, open(dst, "w"), indent=1)
print(f"\nwrote {dst}")
print(f"total LOL train {sum(len(out[k]['splits']['train']['names']) for k in SPEC)}, "
      f"val {sum(len(out[k]['splits']['val']['names']) for k in SPEC)}, "
      f"test {sum(len(out[k]['splits']['test']['names']) for k in SPEC)}")
