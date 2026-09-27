"""
Output-level temporal consistency metrics on L0/L1/L2 video splits.

For each seq and each method, produces a [T,512,512,3] uint8 output array
then computes:
  - tLP: LPIPS-alex between adjacent frames, mean
  - MABD: |mean(Y_t) - mean(Y_{t+1})| in luma [0,1], mean
  - flicker: pixelwise std of luma across T, mean over pixels

Methods:
  - LQ, GT: raw center-crop of dataset frames
  - Retinexformer, HVI-CIDNet, DarkIR, URCSA: fresh inference on raw crops
  - STCD, Seer (HMP-raw), Seer+CP (HMP+CP): extracted from
    /home/ma-user/workspace/zihaohe/visualizations/final_HMP_*/<seq>/<seq>_compare.mp4
    (5-col: LQ | HMP-raw | HMP+CP | STCD | GT; each col crop then center-crop to 512)

Sequences: intersection of eval-lib test set and available compare MP4s
  L0 indoor: 12 pair* (all)
  L1 outdoor: 5 (subset available in final_HMP_outdoor)
  L2 BVI: 6 (subset available in final_HMP_bvi)
"""
from __future__ import annotations
import os, sys, json, time, gc, argparse, traceback
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/home/ma-user/workspace/zihaohe/vjepa_llve")

from common.eval_lib import iter_sdsd_indoor_stcd, iter_sdsd_outdoor, iter_bvi

DEVICE = torch.device("cuda")
T_FRAMES = 13   # match Seer output MP4s (T=13)
CROP = 512

VIZ_ROOT = "/home/ma-user/workspace/zihaohe/visualizations"
MP4_MAP = {
    "L0": os.path.join(VIZ_ROOT, "final_HMP_sdsd_indoor"),
    "L1": os.path.join(VIZ_ROOT, "final_HMP_outdoor"),
    "L2": os.path.join(VIZ_ROOT, "final_HMP_bvi"),
}

# Available seqs (intersection with MP4s existence)
SEQS = {
    "L0": ["pair13","pair15","pair21","pair23","pair31","pair33",
           "pair50","pair52","pair58","pair60","pair68","pair70"],
    "L1": ["MVI_1007_2","MVI_1009","MVI_1009_2","MVI_1011","MVI_1011_2"],
    "L2": ["S33_mix5__10fps","S35_Bunny__10fps","S37_Mario3__10fps",
           "S39_Books4__10fps","S40_Books5__10fps","S40_Books5__20fps"],
}

METHODS_MP4 = {"STCD": 3, "Seer": 1, "Seer+CP": 2}  # column index in 5-col compare
METHODS_BASELINE = ["Retinexformer", "HVI-CIDNet", "DarkIR", "URCSA"]
METHODS_ORDER = ["LQ", "GT", "STCD", "Retinexformer", "HVI-CIDNet", "DarkIR", "URCSA", "Seer", "Seer+CP"]


# ---------------- Metric functions ----------------

_LPIPS = None
def _get_lpips():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net='alex').to(DEVICE).eval()
        for p in _LPIPS.parameters():
            p.requires_grad = False
    return _LPIPS


@torch.no_grad()
def compute_temporal_metrics(frames_u8: np.ndarray) -> Dict[str, float]:
    """
    frames_u8: (T, H, W, 3) uint8 RGB
    Returns dict with tLP_mean, MABD_mean, flicker_mean
    Also returns per-pair tLP list for variance.
    """
    assert frames_u8.dtype == np.uint8
    T = frames_u8.shape[0]
    # Luminance stack in [0,1]
    y = (0.299 * frames_u8[..., 0] + 0.587 * frames_u8[..., 1] + 0.114 * frames_u8[..., 2]).astype(np.float32) / 255.0  # (T, H, W)

    # MABD across adjacent pairs
    mabd_pairs = []
    for t in range(T - 1):
        mabd_pairs.append(abs(y[t].mean() - y[t+1].mean()))
    mabd_mean = float(np.mean(mabd_pairs))

    # Flicker: per-pixel std across T then mean
    flicker = float(y.std(axis=0).mean())

    # tLP with LPIPS-alex
    lpips_model = _get_lpips()
    # Convert to [-1,1] tensor NCHW
    x = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0 * 2.0 - 1.0
    tlp_pairs = []
    for t in range(T - 1):
        v = lpips_model(x[t:t+1], x[t+1:t+2])
        tlp_pairs.append(float(v.item()))
    tlp_mean = float(np.mean(tlp_pairs))
    tlp_std = float(np.std(tlp_pairs))
    return {
        "tLP_mean": tlp_mean, "tLP_std": tlp_std,
        "MABD_mean": mabd_mean, "flicker_mean": flicker,
        "n_pairs": T - 1, "T": T,
    }


# ---------------- MP4 extraction ----------------

def extract_mp4_col(mp4_path: str, col_idx: int, n_cols: int = 5,
                    header_h: int = 36, T_target: int = T_FRAMES, crop: int = CROP) -> np.ndarray:
    """Load MP4, take column col_idx (0-based), crop below header, center-crop 'crop'."""
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)  # BGR
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"empty mp4 {mp4_path}")
    H, W = frames[0].shape[:2]
    cw = W // n_cols
    col = np.stack([f[header_h:, col_idx*cw:(col_idx+1)*cw] for f in frames], axis=0)  # (T, H-hb, cw, 3) BGR
    # center-crop to (crop, crop)
    Th, ch, cw2, _ = col.shape
    if cw2 > crop:
        x0 = (cw2 - crop) // 2
        col = col[:, :, x0:x0+crop, :]
    if ch > crop:
        y0 = (ch - crop) // 2
        col = col[:, y0:y0+crop, :, :]
    # trim/pad to T_target
    if col.shape[0] > T_target:
        col = col[:T_target]
    elif col.shape[0] < T_target:
        # pad by repeating last frame
        pad = np.repeat(col[-1:], T_target - col.shape[0], axis=0)
        col = np.concatenate([col, pad], axis=0)
    # BGR -> RGB
    col_rgb = col[..., ::-1].copy()
    return col_rgb


# ---------------- Baseline model loaders ----------------

def load_retinexformer(split: str):
    """split in {L0, L1, L2}"""
    sys.path.insert(0, "/cache/zihaohe/ext_methods/Retinexformer/basicsr/models/archs")
    from RetinexFormer_arch import RetinexFormer
    m = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2])
    if split == "L0":
        p = "/cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_indoor.pth"
    elif split == "L1":
        p = "/cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_outdoor.pth"
    else:  # L2 (BVI) cross-domain uses indoor per existing eval
        p = "/cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_indoor.pth"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("params", ck.get("params_ema", ck))
    m.load_state_dict(sd, strict=True)
    m = m.to(DEVICE).eval()
    @torch.no_grad()
    def infer(lq):
        return m(lq).clamp(0,1)
    return infer, m


def load_hvi_cidnet(split: str):
    sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/HVI-CIDNet")
    from net.CIDNet import CIDNet
    m = CIDNet()
    # Cross-domain: use Generalization for all L0/L1/L2 (matches existing v2 run)
    p = "/cache/zihaohe/baseline_ckpts/HVI-CIDNet/Generalization/pytorch_model.bin"
    sd = torch.load(p, map_location="cpu", weights_only=False)
    m.load_state_dict(sd, strict=True)
    m = m.to(DEVICE).eval()
    @torch.no_grad()
    def infer(lq):
        _, _, H, W = lq.shape
        ph = (16 - H % 16) % 16
        pw = (16 - W % 16) % 16
        if ph or pw:
            lq = F.pad(lq, (0, pw, 0, ph), mode="reflect")
        y = m(lq)[:, :, :H, :W]
        return y.clamp(0,1)
    return infer, m


def load_darkir(split: str):
    sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/DarkIR")
    from archs.DarkIR import DarkIR
    m = DarkIR(img_channel=3, width=32,
               middle_blk_num_enc=2, middle_blk_num_dec=2,
               enc_blk_nums=[1,2,3], dec_blk_nums=[3,1,1],
               dilations=[1,4,9], extra_depth_wise=True)
    p = "/cache/zihaohe/baseline_ckpts/DarkIR/DarkIR_384.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["params"], strict=True)
    m = m.to(DEVICE).eval()
    @torch.no_grad()
    def infer(lq):
        _, _, H, W = lq.shape
        ph = (8 - H % 8) % 8; pw = (8 - W % 8) % 8
        if ph or pw:
            lq = F.pad(lq, (0, pw, 0, ph), mode="reflect")
        y = m(lq)[:, :, :H, :W]
        return y.clamp(0,1)
    return infer, m


def load_urcsa(split: str):
    """SDSD indoor/outdoor variant on L0/L1, indoor cross-domain on L2 (BVI)."""
    _URCSA = "/home/ma-user/workspace/zihaohe/baselines/URCSA"
    sys.path.insert(0, os.path.join(_URCSA, "sdsd"))
    from sdsd_basemodel import lowlightnet3 as SDSDNet
    from sdsd_basemodel import low_block as SDSD_low_block, ConvBlock as SDSDConvBlock

    def _patch(low_block_cls, ConvBlock_cls):
        orig_init = low_block_cls.__init__
        def new_init(self, dim=64, device="cuda:0"):
            orig_init(self, dim, device)
            self.ende_quarres_up_deline = ConvBlock_cls(4*dim, 2*dim, 1, 1, 0, bias=True)
            self.ende_quarres_up_up_deline = ConvBlock_cls(2*dim, dim, 1, 1, 0, bias=True)
        def _single(self, f_endecoder):
            fullres = f_endecoder
            halfres = self.downsample(fullres)
            quarres = self.downsample2x(halfres)
            e1 = self.endecoder1x(fullres)
            e2 = self.endecoder2x(halfres)
            e4 = self.endecoder4x(quarres)
            u1 = self.upsample4x(e4)
            eu = self.ende_quarres_up_deline(torch.cat([u1, e2], dim=1))
            u2 = self.upsample2x(eu)
            euu = self.ende_quarres_up_up_deline(torch.cat([u2, e1], dim=1))
            cat_all = torch.cat((euu, fullres), dim=1)
            fusion_all = self.attention(cat_all)
            fusion_all = self.outlayer(fusion_all) + cat_all
            fusion_out = self.out_conv2(fusion_all)
            return fusion_out + fullres
        def new_fwd(self, f_endecoder, f_endecoder2=None, mode="train"):
            o = _single(self, f_endecoder); return o, o
        low_block_cls.__init__ = new_init
        low_block_cls.forward = new_fwd

    # only patch once
    if not getattr(SDSD_low_block, "_patched_by_temporal", False):
        _patch(SDSD_low_block, SDSDConvBlock)
        SDSD_low_block._patched_by_temporal = True

    m = SDSDNet()
    if split == "L1":
        p = os.path.join(_URCSA, "sdsd", "RCSAoutdoor.pth")
    else:  # L0, L2 both use indoor per existing eval convention
        p = os.path.join(_URCSA, "sdsd", "RCSAindoor.pth")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    m.load_state_dict(sd, strict=False)
    m = m.to(DEVICE).eval()
    @torch.no_grad()
    def infer(lq):
        _, _, H, W = lq.shape
        ph = (8 - H % 8) % 8; pw = (8 - W % 8) % 8
        if ph or pw:
            lq = F.pad(lq, (0, pw, 0, ph), mode="reflect")
        out, _ = m(lq, lq, mode="test")
        return out[:, :, :H, :W].clamp(0,1)
    return infer, m


BASELINE_LOADERS = {
    "Retinexformer": load_retinexformer,
    "HVI-CIDNet":    load_hvi_cidnet,
    "DarkIR":        load_darkir,
    "URCSA":         load_urcsa,
}


# ---------------- Data loading per split ----------------

def load_split_lq_gt(split: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Returns {seq_name: (lq_u8[T,H,W,3], gt_u8[T,H,W,3])} for the target seqs."""
    if split == "L0":
        it = iter_sdsd_indoor_stcd(n_frames=T_FRAMES, crop=CROP)
    elif split == "L1":
        it = iter_sdsd_outdoor(n_frames=T_FRAMES, crop=CROP)
    else:
        it = iter_bvi(n_frames=T_FRAMES, crop=CROP)
    wanted = set(SEQS[split])
    got = {}
    for seq, lq, gt in it:
        if seq in wanted:
            got[seq] = (lq, gt)
    missing = wanted - set(got.keys())
    if missing:
        print(f"[WARN] {split} missing seqs: {missing}", flush=True)
    return got


def run_baseline_on_seq(infer, lq_u8: np.ndarray) -> np.ndarray:
    """lq_u8: (T,H,W,3) uint8. Returns (T,H,W,3) uint8 model output."""
    T = lq_u8.shape[0]
    outs = []
    for t in range(T):
        lq_t = torch.from_numpy(lq_u8[t]).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
        y = infer(lq_t)
        y_u8 = (y[0].permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
        outs.append(y_u8)
    return np.stack(outs, axis=0)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ma-user/workspace/zihaohe/temporal_consistency_v1.json")
    ap.add_argument("--baselines", nargs="+", default=None,
                    help="Only run these baselines (default: all)")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="Skip baseline inference (only LQ/GT/STCD/Seer/Seer+CP)")
    args = ap.parse_args()

    t0 = time.time()
    baselines = args.baselines or METHODS_BASELINE
    if args.skip_baselines:
        baselines = []

    # per_seq[method][split][seq] = metrics dict
    per_seq: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for m in METHODS_ORDER:
        per_seq[m] = {"L0": {}, "L1": {}, "L2": {}}

    # ============== Phase 1: for each split, get LQ/GT + MP4 methods ==============
    for split in ["L0", "L1", "L2"]:
        print(f"\n#### {split} ####", flush=True)
        seqs_data = load_split_lq_gt(split)
        for seq_name in SEQS[split]:
            if seq_name not in seqs_data:
                print(f"[SKIP] {split}/{seq_name}: no raw data")
                continue
            lq_u8, gt_u8 = seqs_data[seq_name]
            print(f"  {split}/{seq_name}: LQ shape={lq_u8.shape}", flush=True)
            per_seq["LQ"][split][seq_name] = compute_temporal_metrics(lq_u8)
            per_seq["GT"][split][seq_name] = compute_temporal_metrics(gt_u8)

            # MP4 methods
            mp4_dir = os.path.join(MP4_MAP[split], seq_name)
            mp4_path = os.path.join(mp4_dir, f"{seq_name}_compare.mp4")
            if not os.path.exists(mp4_path):
                print(f"  [WARN] no MP4 for {split}/{seq_name}: {mp4_path}")
            else:
                for mname, col_idx in METHODS_MP4.items():
                    frames = extract_mp4_col(mp4_path, col_idx)
                    per_seq[mname][split][seq_name] = compute_temporal_metrics(frames)

    # ============== Phase 2: baselines (fresh inference) ==============
    for bname in baselines:
        print(f"\n#### Baseline: {bname} ####", flush=True)
        for split in ["L0", "L1", "L2"]:
            print(f"  loading {bname} for {split} ...", flush=True)
            try:
                infer, model = BASELINE_LOADERS[bname](split)
            except Exception as e:
                print(f"  [ERROR] loading {bname}/{split}: {e}")
                traceback.print_exc()
                continue
            # inference on each seq
            seqs_data = load_split_lq_gt(split)
            for seq_name in SEQS[split]:
                if seq_name not in seqs_data:
                    continue
                lq_u8, gt_u8 = seqs_data[seq_name]
                try:
                    out_u8 = run_baseline_on_seq(infer, lq_u8)
                except Exception as e:
                    print(f"  [ERROR] {bname}/{split}/{seq_name}: {e}")
                    traceback.print_exc()
                    continue
                per_seq[bname][split][seq_name] = compute_temporal_metrics(out_u8)
                print(f"    {split}/{seq_name}: tLP={per_seq[bname][split][seq_name]['tLP_mean']:.4f} "
                      f"MABD={per_seq[bname][split][seq_name]['MABD_mean']:.5f} "
                      f"flick={per_seq[bname][split][seq_name]['flicker_mean']:.4f}", flush=True)
            # free memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # ============== Aggregate ==============
    summary: Dict[str, Dict[str, dict]] = {}
    for m in METHODS_ORDER:
        summary[m] = {}
        for split in ["L0", "L1", "L2"]:
            entries = per_seq[m][split]
            if not entries:
                summary[m][split] = {"n_seqs": 0, "seqs": []}
                continue
            tLPs  = [v["tLP_mean"]  for v in entries.values()]
            MABDs = [v["MABD_mean"] for v in entries.values()]
            FLKs  = [v["flicker_mean"] for v in entries.values()]
            summary[m][split] = {
                "tLP_mean":     float(np.mean(tLPs)),
                "tLP_std":      float(np.std(tLPs)),
                "MABD_mean":    float(np.mean(MABDs)),
                "MABD_std":     float(np.std(MABDs)),
                "flicker_mean": float(np.mean(FLKs)),
                "flicker_std":  float(np.std(FLKs)),
                "n_seqs":       len(entries),
                "seqs":         sorted(entries.keys()),
            }

    out = {
        "meta": {
            "metric_definitions": {
                "tLP":     "LPIPS-alex between adjacent frames, mean over T-1 pairs then over sequences",
                "MABD":    "|Y_t.mean() - Y_{t+1}.mean()| in luma [0,1], mean over T-1 pairs then over sequences",
                "flicker": "per-pixel std of luma across T frames, then mean over pixels, then mean over sequences",
            },
            "seqs_per_split": SEQS,
            "n_frames_per_seq": T_FRAMES,
            "crop": f"{CROP}x{CROP} center",
            "date": datetime.now(timezone.utc).isoformat(),
            "notes": (
                "LQ, GT, and image baselines (Retinexformer, HVI-CIDNet, DarkIR, URCSA) are run "
                "on RAW 512x512 center-crop of dataset frames. STCD, Seer (HMP-raw), and Seer+CP "
                "(HMP+CP) are extracted from the 5-column compare MP4 (LQ|HMP-raw|HMP+CP|STCD|GT) "
                "under /home/ma-user/workspace/zihaohe/visualizations/final_HMP_*. MP4 columns are "
                "cropped below the 36px label bar and center-cropped to 512x512. H264 compression "
                "in those MP4s could slightly smooth temporal high frequencies; this affects Seer "
                "and STCD equally and is a mild-conservative bias against them on the flicker metric. "
                "Temporal metrics are self-referential (adjacent frames within one method), so "
                "cross-method pixel alignment is not required."
            ),
            "baselines_run": baselines,
        },
        "per_method_per_split": summary,
        "per_seq_details":      per_seq,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {args.out}  (elapsed {time.time()-t0:.1f}s)", flush=True)

    # Print compact table
    print("\n" + "=" * 118)
    print("COMPACT TABLE (tLP / MABD / flicker)")
    print("=" * 118)
    hdr = f"{'Method':14s}"
    for sp in ["L0", "L1", "L2"]:
        hdr += f"| {sp} tLP    MABD     flick    "
    print(hdr)
    for m in METHODS_ORDER:
        row = f"{m:14s}"
        for sp in ["L0", "L1", "L2"]:
            s = summary[m][sp]
            if s.get("n_seqs", 0) == 0:
                row += "| ---      ---      ---      "
            else:
                row += f"| {s['tLP_mean']:.4f}  {s['MABD_mean']:.5f}  {s['flicker_mean']:.4f}  "
        print(row)
    print("=" * 118)


if __name__ == "__main__":
    main()
