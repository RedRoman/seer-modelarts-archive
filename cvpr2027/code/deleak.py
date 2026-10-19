"""Remove scene-level leakage from splits_cvpr.json.

Test lists stay identical to the ICLR round, so a sequence in train/val whose
scene (pairX for pairX_2, MVI_Y for MVI_Y_2) also appears in test is dropped.
Train/val scene overlap is removed the same way, keeping val intact.
"""
import json

P = "/temp/zihaohe/seer_cvpr/datasets/splits_cvpr.json"
d = json.load(open(P))


def base(n):
    for s in ("__10fps", "__20fps"):
        if n.endswith(s):
            return n[:-len(s)]
    return n[:-2] if n.endswith("_2") else n


d["doc"] += ("\n\nScene-level de-leak (2026-10-18): test lists are kept identical to the ICLR round; "
             "train/val sequences whose scene also occurs in test (or, for train, in val) are dropped. "
             "See _dropped_for_scene_leak.")
drop_all = {}
for k in ("SDSD_indoor", "SDSD_outdoor", "BVI_RLV"):
    v = d[k]
    eb = {base(x) for x in v["test"]}
    dropped = []
    for sub in ("train", "val"):
        dropped += [x for x in v[sub] if base(x) in eb]
        v[sub] = [x for x in v[sub] if base(x) not in eb]
    vb = {base(x) for x in v["val"]}
    dropped += [x for x in v["train"] if base(x) in vb]
    v["train"] = [x for x in v["train"] if base(x) not in vb]
    drop_all[k] = sorted(dropped)
    tb, vb2, eb2 = ({base(x) for x in v[s]} for s in ("train", "val", "test"))
    print(f"{k}: train {len(v['train'])} val {len(v['val'])} test {len(v['test'])}  "
          f"scene overlap tr/va={len(tb & vb2)} tr/te={len(tb & eb2)} va/te={len(vb2 & eb2)}")

d["_dropped_for_scene_leak"] = drop_all
json.dump(d, open(P, "w"), indent=1)
print("dropped:", json.dumps(drop_all))
