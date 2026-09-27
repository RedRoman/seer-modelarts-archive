"""URCSA eval v2 -- adds DISTS + CLIPIQA. Same monkey-patch as v1."""
from __future__ import annotations
import os, sys, argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")

_URCSA = "/home/ma-user/workspace/zihaohe/baselines/URCSA"
sys.path.insert(0, os.path.join(_URCSA, "sdsd"))

from common.eval_lib_v2 import evaluate_method_v2
from sdsd_basemodel import lowlightnet3 as SDSDNet
from sdsd_basemodel import low_block as SDSD_low_block, ConvBlock as SDSDConvBlock


def _make_new_forward_single(ConvBlock_cls):
    def _single(self, f_endecoder):
        fullres = f_endecoder
        halfres = self.downsample(fullres)
        quarres = self.downsample2x(halfres)
        ende_fullres_out = self.endecoder1x(fullres)
        ende_halfres = self.endecoder2x(halfres)
        ende_quarres = self.endecoder4x(quarres)
        u1 = self.upsample4x(ende_quarres)
        ende_quarres_up = self.ende_quarres_up_deline(torch.cat([u1, ende_halfres], dim=1))
        u2 = self.upsample2x(ende_quarres_up)
        ende_quarres_up_up = self.ende_quarres_up_up_deline(torch.cat([u2, ende_fullres_out], dim=1))
        cat_all = torch.cat((ende_quarres_up_up, fullres), dim=1)
        fusion_all = self.attention(cat_all)
        fusion_all = self.outlayer(fusion_all) + cat_all
        fusion_out = self.out_conv2(fusion_all)
        fusion_out = fusion_out + fullres
        return fusion_out
    return _single


def _patch_low_block(low_block_cls, ConvBlock_cls, returns_tuple):
    orig_init = low_block_cls.__init__
    def new_init(self, dim=64, device="cuda:0"):
        orig_init(self, dim, device)
        self.ende_quarres_up_deline = ConvBlock_cls(4 * dim, 2 * dim, 1, 1, 0, bias=True)
        self.ende_quarres_up_up_deline = ConvBlock_cls(2 * dim, dim, 1, 1, 0, bias=True)
    single = _make_new_forward_single(ConvBlock_cls)
    if returns_tuple:
        def new_forward(self, f_endecoder, f_endecoder2=None, mode="train"):
            o = single(self, f_endecoder); return o, o
    else:
        def new_forward(self, f_endecoder):
            return single(self, f_endecoder)
    low_block_cls.__init__ = new_init
    low_block_cls.forward = new_forward


_patch_low_block(SDSD_low_block, SDSDConvBlock, returns_tuple=True)


def load_sdsd(weights_path, device):
    m = SDSDNet()
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    r = m.load_state_dict(sd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        print(f"[URCSA-SDSD load] missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}")
    return m.to(device).eval()


def load_lol(weights_path, device):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "urcsa_lol_base_model",
        os.path.join(_URCSA, "lol", "base_model.py"))
    mod = importlib.util.module_from_spec(spec)
    lol_dir = os.path.join(_URCSA, "lol")
    sys.path.insert(0, lol_dir)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(lol_dir)

    _patch_low_block(mod.low_block, mod.ConvBlock, returns_tuple=False)
    m = mod.lowlightnet3()
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    r = m.load_state_dict(sd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        print(f"[URCSA-LOL load] missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}")
    return m.to(device).eval(), mod


def make_infer_sdsd(model):
    @torch.no_grad()
    def infer(lq):
        _, _, H, W = lq.shape
        ph = (8 - H % 8) % 8; pw = (8 - W % 8) % 8
        if ph or pw:
            lq = F.pad(lq, (0, pw, 0, ph), mode="reflect")
        out, _ = model(lq, lq, mode="test")
        return out[:, :, :H, :W].clamp(0, 1)
    return infer


def make_infer_lol(model):
    @torch.no_grad()
    def infer(lq):
        _, _, H, W = lq.shape
        ph = (8 - H % 8) % 8; pw = (8 - W % 8) % 8
        if ph or pw:
            lq = F.pad(lq, (0, pw, 0, ph), mode="reflect")
        out = model(lq)
        return out[:, :, :H, :W].clamp(0, 1)
    return infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["SDSD_indoor", "SDSD_outdoor",
                             "BVI_RLV", "LOL_v1", "LOLv2_Syn", "LOLv2_Real"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="/cache/zihaohe/baseline_ckpts/URCSA_eval_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda")

    ind = load_sdsd(os.path.join(_URCSA, "sdsd", "RCSAindoor.pth"), device)
    out_m = load_sdsd(os.path.join(_URCSA, "sdsd", "RCSAoutdoor.pth"), device)
    lol_m, _ = load_lol(os.path.join(_URCSA, "lol", "checkpoint.pth"), device)

    infer_ind = make_infer_sdsd(ind)
    infer_out = make_infer_sdsd(out_m)
    infer_lol = make_infer_lol(lol_m)

    weight_map = {
        "SDSD_indoor":  (infer_ind, "RCSAindoor.pth (matched, low_block patched with deline convs)"),
        "SDSD_outdoor": (infer_out, "RCSAoutdoor.pth (matched, low_block patched with deline convs)"),
        "BVI_RLV":      (infer_ind, "RCSAindoor.pth (cross-domain)"),
        "LOL_v1":       (infer_lol, "LOL checkpoint.pth (matched, single-frame variant)"),
        "LOLv2_Syn":    (infer_lol, "LOL checkpoint.pth (cross-domain within LOL)"),
        "LOLv2_Real":   (infer_lol, "LOL checkpoint.pth (cross-domain within LOL)"),
    }

    combined_per_seq = {}
    combined_summary = {}
    for ds in args.datasets:
        infer_fn, note = weight_map[ds]
        print(f"\n########## URCSA @ {ds}  ({note}) ##########", flush=True)
        r = evaluate_method_v2(
            method_name=f"URCSA[{ds}]",
            infer_fn=infer_fn,
            datasets=[ds],
            out_path=f"/tmp/urcsa_v2_{ds}.json",
            notes=note,
            limit_seqs=args.limit,
        )
        combined_per_seq.update(r["per_seq"])
        combined_summary[ds] = r["summary"][ds]

    out = {
        "method": "URCSA",
        "notes": ("Uses published RCSAindoor.pth / RCSAoutdoor.pth on SDSD splits (matched); "
                  "RCSAindoor.pth on BVI_RLV (cross-domain); LOL checkpoint.pth on all three LOL "
                  "splits. Released source is missing two 1x1 deline conv modules present in the "
                  "checkpoints; we monkey-patch low_block accordingly. v2 adds DISTS+CLIPIQA."),
        "protocol": "video: center T=16, 512x512 center-crop; images: full-image. PSNR uint8 RGB, LPIPS+DISTS+CLIPIQA pyiqa.",
        "per_seq": combined_per_seq,
        "summary": combined_summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {args.out}")

    print("\n===== URCSA v2 summary =====")
    for k, v in combined_summary.items():
        print(f"  {k:14s} PSNR={v['psnr_rgb']:.3f}  LPIPS={v['lpips']:.4f}  DISTS={v['dists']:.4f}  CLIPIQA={v['clipiqa']:.4f}  (n={v['n_seqs']})")


if __name__ == "__main__":
    main()
