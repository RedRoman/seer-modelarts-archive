"""Freeze the CVPR split lists (train / val / test) for SDSD and BVI-RLV.

Rules are those pre-registered in notes/CVPR_PREREG.md:
  - test: unchanged from the ICLR round (SDSDDataset sorted-name 80/20; BVI 80/20 by scene)
  - SDSD-val: every 10th scene of the train split, in sorted order of base name
    (pairX and pairX_2, MVI_Y and MVI_Y_2 always travel together)
  - BVI-val: every 4th scene of the BVI train scenes
Writes datasets/splits_cvpr.json; never overwrites an existing file without --force.
"""
import argparse, json, os

D = "/temp/zihaohe/seer_cvpr/datasets"
OUT = os.path.join(D, "splits_cvpr.json")


def base(name):
    """pair5_2 -> pair5 ; MVI_1007_2 -> MVI_1007 ; S33_mix5__10fps -> S33_mix5"""
    for suf in ("__10fps", "__20fps"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name[:-2] if name.endswith("_2") else name


def split_80_20(seqs):
    k = int(round(0.8 * len(seqs)))
    return seqs[:k], seqs[k:]


def pick_val(train_seqs, every):
    """Hold out every n-th *scene* (keeping _2 variants with their base)."""
    scenes = sorted({base(s) for s in train_seqs})
    val_scenes = set(scenes[::every])
    val = sorted(s for s in train_seqs if base(s) in val_scenes)
    tr = sorted(s for s in train_seqs if base(s) not in val_scenes)
    return tr, val


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        raise SystemExit(f"{OUT} exists; pass --force to regenerate")

    out = {"doc": __doc__}
    for key, root, every in (("SDSD_indoor", f"{D}/indoor_static_np", 10),
                             ("SDSD_outdoor", f"{D}/outdoor_static_np", 10)):
        seqs = sorted(os.listdir(os.path.join(root, "input")))
        tr0, te = split_80_20(seqs)
        tr, va = pick_val(tr0, every)
        out[key] = {"root": root, "train": tr, "val": va, "test": te}
        print(f"{key}: all {len(seqs)} -> train {len(tr)} val {len(va)} test {len(te)}")

    bvi = f"{D}/BVI-Lowlight-videos"
    scenes = sorted(d for d in os.listdir(bvi) if os.path.isdir(os.path.join(bvi, d)))
    tr_sc, te_sc = split_80_20(scenes)
    val_sc = sorted(tr_sc[::4])
    tr_sc = [s for s in tr_sc if s not in set(val_sc)]
    clips = lambda sc: [f"{s}__{f}fps" for s in sc for f in ("10", "20")
                        if os.path.isdir(os.path.join(bvi, s, f"low_light_{f}"))]
    out["BVI_RLV"] = {"root": bvi, "train_scenes": tr_sc, "val_scenes": val_sc,
                      "test_scenes": te_sc, "train": clips(tr_sc),
                      "val": clips(val_sc), "test": clips(te_sc)}
    print(f"BVI: scenes {len(scenes)} -> train {len(tr_sc)} val {len(val_sc)} test {len(te_sc)}"
          f" | clips train {len(out['BVI_RLV']['train'])} val {len(out['BVI_RLV']['val'])}"
          f" test {len(out['BVI_RLV']['test'])}")

    # sanity: no leakage between any two subsets of the same dataset
    for k, v in out.items():
        if not isinstance(v, dict) or "train" not in v:
            continue
        tr, va, te = set(v["train"]), set(v["val"]), set(v["test"])
        assert not (tr & va) and not (tr & te) and not (va & te), f"leak in {k}"
        tb, vb, eb = ({base(x) for x in s} for s in (tr, va, te))
        print(f"  {k}: scene-level overlap train/val {len(tb&vb)} train/test {len(tb&eb)} val/test {len(vb&eb)}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("[SAVE]", OUT)


if __name__ == "__main__":
    main()
