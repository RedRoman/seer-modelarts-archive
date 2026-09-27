"""Sanity: STCD on pair13/pair50 under several protocols, to explain old numbers.
old comprehensive_eval.json sdsd.per_seq.STCD: pair13 22.111, pair50 29.315
old stcd_repro (STCD-exact, first30, 960x512): pair13 24.327, pair50 22.956
"""
import os, sys, json
import numpy as np, torch, cv2
sys.path.insert(0, "/temp/zihaohe/workspace/baselines")
import run_stcd_unified as R
from common.eval_lib import _load_img_any, center_crop, rgb_psnr_uint8, iter_sdsd_indoor_stcd
from seer_eval_unified import iter_sdsd_indoor_seer, score_frames

ROOT = "/cache/zihaohe/datasets/SDSD_indoor/indoor_static_np"
dev = torch.device("cuda")
M = R.STCD(dev)


def lists(seq):
    lq = sorted(os.listdir(f"{ROOT}/input/{seq}")); gt = sorted(os.listdir(f"{ROOT}/GT/{seq}"))
    return lq, gt


def load(seq, pairing, first, n, crop=None, resize=None):
    lq, gt = lists(seq)
    if pairing == "basename":
        c = sorted(set(lq) & set(gt)); lqn, gtn = c, c
    else:
        m = min(len(lq), len(gt)); lqn, gtn = lq[:m], gt[:m]
    if first:
        s = 0
    else:
        s = max(0, (len(lqn) - n) // 2)
    lqn, gtn = lqn[s:s + n], gtn[s:s + n]
    def rd(d, f):
        a = np.load(f"{ROOT}/{d}/{seq}/{f}")
        if resize is not None:
            a = cv2.resize(a, resize)
        if crop is not None:
            a = center_crop(a, crop)
        return a
    return (np.stack([rd("input", f) for f in lqn]), np.stack([rd("GT", f) for f in gtn]),
            (lqn[0], lqn[-1], gtn[0], gtn[-1]))


def psnr_only(lq_u8, gt_u8, swap=True, round_u8=False, chunk=8):
    lq, _ = R.to_dev(lq_u8, gt_u8, dev)
    pred = M.enhance_clip(lq, chunk=chunk, swap_bgr=swap)
    p = pred.permute(0, 2, 3, 1).cpu().numpy() * 255
    p = np.round(p) if round_u8 else p
    p = p.clip(0, 255).astype(np.uint8)
    return float(np.mean([rgb_psnr_uint8(p[t], gt_u8[t]) for t in range(len(p))]))


res = {}
for seq in ["pair13", "pair50"]:
    r = {}
    # V1 = exact unified path (iterator + score_frames), swap on/off
    for name, lq_u8, gt_u8 in iter_sdsd_indoor_stcd(16, 512):
        if name != seq: continue
        lq, gt = R.to_dev(lq_u8, gt_u8, dev)
        r["V1_unified_STCD12iter_swap"] = score_frames(M.enhance_clip(lq), gt)["psnr_rgb"]
        r["V1b_unified_STCD12iter_noswap"] = score_frames(M.enhance_clip(lq, swap_bgr=False), gt)["psnr_rgb"]
        r["V1c_LQ_identity"] = score_frames(lq, gt)["psnr_rgb"]
    for name, lq_u8, gt_u8 in iter_sdsd_indoor_seer(16, 512):
        if name != seq: continue
        lq, gt = R.to_dev(lq_u8, gt_u8, dev)
        r["V1d_unified_SeerL0iter_swap"] = score_frames(M.enhance_clip(lq), gt)["psnr_rgb"]
    for pairing in ["basename", "index"]:
        a, b, ids = load(seq, pairing, first=True, n=16, crop=512)
        r[f"V2_first16_{pairing}_crop512"] = psnr_only(a, b); r[f"V2_ids_{pairing}"] = ids
        a, b, ids = load(seq, pairing, first=True, n=16, crop=None)
        r[f"V3_first16_{pairing}_fullres"] = psnr_only(a, b, chunk=2)
        a, b, ids = load(seq, pairing, first=False, n=16, crop=None)
        r[f"V3b_center16_{pairing}_fullres"] = psnr_only(a, b, chunk=2)
        a, b, ids = load(seq, pairing, first=True, n=30, resize=(960, 512))
        r[f"V4_STCDexact_first30_{pairing}_960x512_round"] = psnr_only(a, b, round_u8=True)
        a, b, ids = load(seq, pairing, first=True, n=16, resize=(960, 512))
        r[f"V4b_first16_{pairing}_960x512"] = psnr_only(a, b)
    res[seq] = r
    for k, v in r.items():
        print(seq, k, v if isinstance(v, tuple) else f"{v:.3f}", flush=True)

old = json.load(open("/temp/zihaohe/cache/checkpoint/runs/comprehensive_eval.json"))["sdsd"]["per_seq"]["STCD"]
print("OLD comprehensive:", {s: round(old[s]["psnr_rgb"], 3) for s in ["pair13", "pair50"]})
json.dump(res, open("/cache/zihaohe/stcd_unified/sanity.json", "w"), indent=1)
