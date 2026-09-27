"""Training-set membership of the evaluation sequences for every checkpoint used in the ICLR tables.

Status values per sequence:
  "train"            in the checkpoint's training set (by its official split rule + data release)
  "train_if_present" the split rule would put it in training; whether the authors' data copy
                     contains this folder cannot be verified on disk (counted as seen in n_seen_upper)
  "held_out"         excluded by the official split rule (test / val list)
  "not_in_release"   folder not in the data release the checkpoint was trained on
  "none_sdsd"        checkpoint trained on non-SDSD data only
  "unknown"          cannot be determined
Scene-level: pairX and pairX_2 (MVI_x and MVI_x_2) are two recordings of the SAME physical scene
(checked visually, /cache/zihaohe/stcd_unified/chk/x_vs_x2.png). "scene_seen" lists test seqs whose
sibling (X <-> X_2) is in training although the seq itself is not.
"""
import json
import os
import sys

sys.path.insert(0, "/temp/zihaohe/workspace/vjepa_llve")
from data.sdsd_dataset import SDSDDataset  # noqa: E402

OUT = "/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals/train_membership.json"
REF = "/temp/zihaohe/cache/checkpoint/runs/encswap_evals/reference_v35stage4_L0to5_full_metrics_v2.json"
IN_ROOT = "/cache/zihaohe/datasets/SDSD_indoor"
OUT_ROOT = "/cache/zihaohe/datasets/SDSD_outdoor"
RF = "/temp/zihaohe/cache/ext_methods/Retinexformer"
ST = "/temp/zihaohe/cache/stcd_repro/LLVE_STCD"
UR = "/temp/zihaohe/workspace/baselines/URCSA"

ref = json.load(open(REF))["per_seq"]
L0 = list(ref["L0_SDSD-indoor"]["HMP-raw"].keys())
L1 = list(ref["L1_SDSD-outdoor"]["HMP-raw"].keys())
STCD12 = ["pair13", "pair15", "pair21", "pair23", "pair31", "pair33",
          "pair50", "pair52", "pair58", "pair60", "pair68", "pair70"]
SPLITS = {"L0": L0, "STCD12": STCD12, "L1": L1}

# folders of the SNR-Aware static_np release (local copy dated 2022-03-22); pair50..70 were added
# 2026-05-03 from the original SDSD indoor_png release (VJEPA_LLVE_CODE_REVIEW.md, section BX0)
SNR_INDOOR_RELEASE = sorted(d for d in os.listdir(f"{IN_ROOT}/indoor_static_np/output"))
LOCAL_INDOOR = sorted(os.listdir(f"{IN_ROOT}/indoor_static_np/input"))
LOCAL_OUTDOOR = sorted(os.listdir(f"{OUT_ROOT}/outdoor_static_np/input"))
ADDED_LATER = sorted(set(LOCAL_INDOOR) - set(SNR_INDOOR_RELEASE))


def base(s):
    return s[:-2] if s.endswith("_2") else s


def sibling(s):
    return s[:-2] if s.endswith("_2") else s + "_2"


def testing_dir_rule(testing, release, strip_2):
    """train = every folder of `release` not in testing (and, if strip_2, whose base is not in testing)."""
    def f(s):
        if s not in release:
            return "not_in_release"
        if s in testing or (strip_2 and base(s) in testing):
            return "held_out"
        return "train"
    return f


def exact_name_rule_unknown_release(testing):
    def f(s):
        return "held_out" if s in testing else "train_if_present"
    return f


def summarize(status, train_set=frozenset(), maybe_train_set=frozenset()):
    """train_set: folders known to be trained on; maybe_train_set: folders trained on if present in
    the authors' copy. Used for the scene-level (X <-> X_2 sibling) check."""
    out = {}
    for sp, seqs in SPLITS.items():
        st = {s: status[sp](s) for s in seqs}
        seen = [s for s, v in st.items() if v == "train"]
        maybe = [s for s, v in st.items() if v == "train_if_present"]
        unk = [s for s, v in st.items() if v == "unknown"]
        scene = [s for s in seqs if st[s] not in ("train", "train_if_present") and sibling(s) in train_set]
        scene_m = [s for s in seqs if st[s] not in ("train", "train_if_present") and sibling(s) in maybe_train_set]
        out[sp] = {"n": len(seqs), "n_seen": len(seen), "n_seen_upper": len(seen) + len(maybe),
                   "n_unknown": len(unk), "seen": seen, "seen_if_present": maybe,
                   "scene_seen_via_sibling": scene, "scene_seen_via_sibling_if_present": scene_m,
                   "per_seq": st}
    return out


M = {}

# ---- Retinexformer official SDSD_indoor.pth / SDSD_outdoor.pth ----
rf_in_test = ["pair11", "pair21", "pair1", "pair19", "pair4", "pair9"]
rf_out_test = ["MVI_0898", "MVI_0928", "MVI_0906", "MVI_0975", "MVI_1001", "MVI_0997", "MVI_1003",
               "MVI_1026", "MVI_1030", "MVI_1032"]
rf_in_rule = testing_dir_rule(rf_in_test, set(SNR_INDOOR_RELEASE), strip_2=True)
rf_out_rule = testing_dir_rule(rf_out_test, set(LOCAL_OUTDOOR), strip_2=True)
rf_in_train = {s for s in SNR_INDOOR_RELEASE if rf_in_rule(s) == "train"}
rf_out_train = {s for s in LOCAL_OUTDOOR if rf_out_rule(s) == "train"}
M["Retinexformer"] = {
    "checkpoints": {"L0": "baseline_ckpts/Retinexformer/SDSD_indoor.pth",
                    "STCD12": "baseline_ckpts/Retinexformer/SDSD_indoor.pth",
                    "L1": "baseline_ckpts/Retinexformer/SDSD_outdoor.pth"},
    "sources": [f"{RF}/Options/RetinexFormer_SDSD_indoor.yml (datasets.train.testing_dir: {','.join(rf_in_test)})",
                f"{RF}/Options/RetinexFormer_SDSD_outdoor.yml (datasets.train.testing_dir: {','.join(rf_out_test)})",
                f"{RF}/basicsr/data/SDSD_image_dataset.py L44-49: phase=train skips folder if name or name.split('_2')[0] in testing_dir; all other folders of dataroot are trained on",
                f"{RF}/README.md: SDSD indoor_static_np/outdoor_static_np from the SNR-Aware release",
                "release contents: local SNR-Aware copy (indoor_static_np/output, dated 2022-03-22, 68 folders pair1..pair34 + _2); "
                "pair50/52/58/60/68/70 are NOT in it (added locally 2026-05-03 from original SDSD indoor_png, VJEPA_LLVE_CODE_REVIEW.md BX0); "
                "outdoor_static_np local copy (126 folders) assumed identical to the release"],
    "assumption": "local 2022-03-22 static_np copies == the release Retinexformer was trained on",
    "splits": summarize({"L0": rf_in_rule, "STCD12": rf_in_rule, "L1": rf_out_rule},
                        train_set=rf_in_train | rf_out_train),
}
MAYBE_IN = frozenset(s for s in LOCAL_INDOOR if s not in STCD12)

# ---- STCD indoor_G.pth (also used cross-domain on outdoor / BVI) ----
M["STCD"] = {
    "checkpoints": {"L0": "stcd_repro/LLVE_STCD/pretrained_model/SDSD_indoor_model/indoor_G.pth",
                    "STCD12": "same", "L1": "same (indoor-only checkpoint, cross-domain on outdoor)"},
    "sources": [f"{ST}/options/train/SDSD_indoor.yml (testing_dir: {','.join(STCD12)}; dataroot .../indoor_np)",
                f"{ST}/data/dataset_SDSD_train.py L41: skips folder only if its exact name is in testing_dir (no _2 handling)",
                f"{ST}/README.md: data = SDSD indoor_np (original SDSD release, baidu/google link); training on SDSD-indoor only"],
    "assumption": "authors' indoor_np contents not on disk (the 12 GB indoor_np download was rejected/deleted); "
                  "every non-test folder present there is trained on",
    "splits": summarize({"L0": exact_name_rule_unknown_release(STCD12),
                         "STCD12": exact_name_rule_unknown_release(STCD12),
                         "L1": lambda s: "none_sdsd_outdoor"}, maybe_train_set=MAYBE_IN),
}
M["STCD"]["splits"]["L1"].update({"n_seen": 0, "n_seen_upper": 0,
                                  "note": "indoor_G.pth trained on SDSD-indoor only -> 0 outdoor seqs seen"})

# ---- URCSA RCSAindoor.pth / RCSAoutdoor.pth ----
M["URCSA"] = {
    "checkpoints": {"L0": "URCSA/sdsd/RCSAindoor.pth", "STCD12": "URCSA/sdsd/RCSAindoor.pth",
                    "L1": "URCSA/sdsd/RCSAoutdoor.pth"},
    "sources": [f"{UR}/sdsd/train_sdsd_model_begin_newloss.py all_list() L84-102 (and _continue_ variant): "
                f"train = every folder of --trainset (default /root/autodl-tmp/SDSD_224/indoor/) whose exact name "
                f"is not in new_test = STCD-12; test_list() uses the same 12",
                f"{UR}/README.md + released code: no outdoor split / list anywhere -> RCSAoutdoor.pth membership unknown"],
    "assumption": "authors' SDSD_224/indoor copy not on disk",
    "splits": summarize({"L0": exact_name_rule_unknown_release(STCD12),
                         "STCD12": exact_name_rule_unknown_release(STCD12),
                         "L1": lambda s: "unknown"}, maybe_train_set=MAYBE_IN),
}

# ---- non-SDSD checkpoints ----
def none_rule(s):
    return "none_sdsd"


for name, ck, src in [
    ("DarkIR", "baseline_ckpts/DarkIR/DarkIR_384.pt",
     ["/temp/zihaohe/workspace/baselines/DarkIR/README.md L58-69, L96: LOL-Blur-only training for the LOLBlur model; datasets list has no SDSD",
      "/temp/zihaohe/workspace/baselines/run_darkir_v2.py note: DarkIR_384.pt = LOL-Blur trained"]),
    ("HVI-CIDNet", "baseline_ckpts/HVI-CIDNet/Generalization/pytorch_model.bin",
     ["/temp/zihaohe/workspace/baselines/HVI-CIDNet/Readme.md L144-153: generalization weights trained on LOLv2-Syn with random gamma; "
      "HF-demo 'Generalization' weights' exact training data not fully documented (no SDSD mentioned)"]),
    ("Zero-DCE", "ext_methods/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth",
     ["/temp/zihaohe/cache/ext_methods/Zero-DCE/README.md L50-55: zero-reference training on the released train_data (SICE Part1), no SDSD"]),
    ("SCI", "ext_methods/SCI/CVPR/weights/medium.pt",
     ["/temp/zihaohe/cache/ext_methods/SCI/CVPR/README.md L27-33: medium.pt mainly trained on LOL and LSRW (unsupervised); no SDSD"]),
]:
    M[name] = {"checkpoints": {"L0": ck, "STCD12": ck, "L1": ck}, "sources": src,
               "splits": summarize({"L0": none_rule, "STCD12": none_rule, "L1": none_rule})}

# ---- Seer (reference): SDSDDataset sorted 80/20 train split on indoor + outdoor ----
seer_tr_in = {s[0] for s in SDSDDataset(root=IN_ROOT, subset="direct", split="train", clip_len=16,
                                         crop_size=256, is_train=False).seqs}
seer_tr_out = {s[0] for s in SDSDDataset(root=OUT_ROOT, subset="direct", split="train", clip_len=16,
                                          crop_size=256, is_train=False).seqs}
seer_rule = lambda s: "train" if s in (seer_tr_in | seer_tr_out) else "held_out"  # noqa: E731
M["Seer (v35 stage4, reference)"] = {
    "checkpoints": {"L0": "runs/v35_stage4/final.pt", "STCD12": "same", "L1": "same"},
    "sources": ["/temp/zihaohe/workspace/vjepa_llve/tools/v35_stage4_train.py L155-160: SDSDDataset(split='train') on SDSD_indoor and SDSD_outdoor",
                "/temp/zihaohe/workspace/vjepa_llve/data/sdsd_dataset.py L155-163: sorted-name 80/20 split (computed here from the local folders)"],
    "splits": summarize({"L0": seer_rule, "STCD12": seer_rule, "L1": seer_rule},
                        train_set=seer_tr_in | seer_tr_out),
}

out = {
    "doc": __doc__,
    "test_splits": {"L0 (Seer SDSD-indoor test, 15)": L0, "STCD12": STCD12, "L1 (Seer SDSD-outdoor test, 25)": L1},
    "local_release_info": {"snr_static_np_indoor_folders": len(SNR_INDOOR_RELEASE),
                           "indoor_folders_added_2026_05_03": ADDED_LATER,
                           "outdoor_folders": len(LOCAL_OUTDOOR)},
    "methods": M,
}
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
print(f"{'method':32s} {'L0-15':>10s} {'STCD-12':>10s} {'L1-25':>10s}")
for m, v in M.items():
    row = []
    for sp in ["L0", "STCD12", "L1"]:
        s = v["splits"][sp]
        if s.get("n_unknown"):
            row.append("unknown")
        elif s["n_seen"] == s["n_seen_upper"]:
            row.append(str(s["n_seen"]))
        else:
            row.append(f"{s['n_seen']}-{s['n_seen_upper']}")
        if s["scene_seen_via_sibling"]:
            row[-1] += f" (+scene:{','.join(s['scene_seen_via_sibling'])})"
        if s["scene_seen_via_sibling_if_present"]:
            row[-1] += f" (+scene?:{','.join(s['scene_seen_via_sibling_if_present'])})"
    print(f"{m:32s} " + " | ".join(row))
