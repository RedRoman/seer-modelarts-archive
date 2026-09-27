"""Run 4 baselines on 3 LOL LQ patches; save per-baseline output PNGs.

Inputs:
  /home/ma-user/workspace/zihaohe/HMP-Net/tmp/lol_lq_patches/
      lol_v1_146_lq.png
      lolv2syn_r00816405t_lq.png
      lolv2real_00690_lq.png

Outputs:
  /home/ma-user/workspace/zihaohe/HMP-Net/tmp/lol_baseline_outputs/
      Retinexformer/{lol_v1_146,lolv2syn_r00816405t,lolv2real_00690}.png
      HVI_CIDNet/{...}
      DarkIR/{...}
      URCSA/{...}

Weight choice (matched-domain if available, else best cross-domain used in v2 scripts):
  Retinexformer -> SDSD_indoor.pth (cross; LOL weight unreachable in v2 scripts)
  HVI-CIDNet    -> LOLv1-wperc / LOLv2_syn-wperc / LOLv2_real-bestPSNR (matched)
  DarkIR        -> DarkIR_384.pt (LOL-Blur; cross)
  URCSA         -> LOL checkpoint.pth (matched, single-frame)
"""
from __future__ import annotations
import os, sys, traceback
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")

LQ_DIR = "/home/ma-user/workspace/zihaohe/HMP-Net/tmp/lol_lq_patches"
OUT_ROOT = "/home/ma-user/workspace/zihaohe/HMP-Net/tmp/lol_cherrypick/baselines"

# name -> (input_filename, output_basename, split)
TARGETS = [
    ("lol_v1_748",          "lol_v1_748_lq.png",          "LOLv1"),
    ("lolv2syn_r08873591t", "lolv2syn_r08873591t_lq.png", "LOLv2_syn"),
    ("lolv2real_00713",     "lolv2real_00713_lq.png",     "LOLv2_real"),
]

device = torch.device("cuda")


def load_lq_tensor(path):
    """Return (u8_hwc, tensor_1x3xHxW in [0,1]) from PNG."""
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return img, t.to(device)


def tensor_to_uint8(y_1x3xHxW):
    return (y_1x3xHxW.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def pad_to_mult(x, m):
    _, _, H, W = x.shape
    ph = (m - H % m) % m
    pw = (m - W % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, H, W


# ---------------------------------------------------------------- Retinexformer
def run_retinexformer():
    print("\n===== Retinexformer =====", flush=True)
    out_dir = os.path.join(OUT_ROOT, "Retinexformer")
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, "/cache/zihaohe/ext_methods/Retinexformer/basicsr/models/archs")
    from RetinexFormer_arch import RetinexFormer

    net_kwargs = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    m = RetinexFormer(**net_kwargs)
    ck = torch.load("/cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_indoor.pth",
                    map_location="cpu", weights_only=False)
    sd = ck.get("params", ck.get("params_ema", ck))
    m.load_state_dict(sd, strict=True)
    m = m.to(device).eval()

    ok = 0
    for name, lqfile, split in TARGETS:
        try:
            _, lq_t = load_lq_tensor(os.path.join(LQ_DIR, lqfile))
            with torch.no_grad():
                xp, H, W = pad_to_mult(lq_t, 8)
                y = m(xp)[:, :, :H, :W]
            out_p = os.path.join(out_dir, f"{name}.png")
            imageio.imwrite(out_p, tensor_to_uint8(y))
            print(f"  {name}: OK  ({lq_t.shape[-2]}x{lq_t.shape[-1]} -> {y.shape[-2]}x{y.shape[-1]})", flush=True)
            ok += 1
        except Exception as e:
            print(f"  {name}: FAIL -- {e}", flush=True)
            traceback.print_exc()
    del m
    torch.cuda.empty_cache()
    return ok


# ---------------------------------------------------------------- HVI-CIDNet
def run_hvi_cidnet():
    print("\n===== HVI-CIDNet =====", flush=True)
    out_dir = os.path.join(OUT_ROOT, "HVI_CIDNet")
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/HVI-CIDNet")
    from net.CIDNet import CIDNet

    ckpts = {
        "LOLv1":      "/cache/zihaohe/baseline_ckpts/HVI-CIDNet/LOLv1-wperc/pytorch_model.bin",
        "LOLv2_real": "/cache/zihaohe/baseline_ckpts/HVI-CIDNet/LOLv2_real-bestPSNR/pytorch_model.bin",
        "LOLv2_syn":  "/cache/zihaohe/baseline_ckpts/HVI-CIDNet/LOLv2_syn-wperc/pytorch_model.bin",
    }
    models = {}
    for split, p in ckpts.items():
        mm = CIDNet()
        sd = torch.load(p, map_location="cpu", weights_only=False)
        mm.load_state_dict(sd, strict=True)
        models[split] = mm.to(device).eval()

    ok = 0
    for name, lqfile, split in TARGETS:
        try:
            _, lq_t = load_lq_tensor(os.path.join(LQ_DIR, lqfile))
            mm = models[split]
            with torch.no_grad():
                xp, H, W = pad_to_mult(lq_t, 16)
                y = mm(xp)[:, :, :H, :W]
            out_p = os.path.join(out_dir, f"{name}.png")
            imageio.imwrite(out_p, tensor_to_uint8(y))
            print(f"  {name}[{split}]: OK", flush=True)
            ok += 1
        except Exception as e:
            print(f"  {name}[{split}]: FAIL -- {e}", flush=True)
            traceback.print_exc()
    for mm in models.values():
        del mm
    torch.cuda.empty_cache()
    return ok


# ---------------------------------------------------------------- DarkIR
def run_darkir():
    print("\n===== DarkIR =====", flush=True)
    out_dir = os.path.join(OUT_ROOT, "DarkIR")
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/DarkIR")
    from archs.DarkIR import DarkIR

    m = DarkIR(img_channel=3, width=32,
               middle_blk_num_enc=2, middle_blk_num_dec=2,
               enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
               dilations=[1, 4, 9], extra_depth_wise=True)
    ck = torch.load("/cache/zihaohe/baseline_ckpts/DarkIR/DarkIR_384.pt",
                    map_location="cpu", weights_only=False)
    m.load_state_dict(ck["params"], strict=True)
    m = m.to(device).eval()

    ok = 0
    for name, lqfile, split in TARGETS:
        try:
            _, lq_t = load_lq_tensor(os.path.join(LQ_DIR, lqfile))
            with torch.no_grad():
                xp, H, W = pad_to_mult(lq_t, 8)
                y = m(xp)[:, :, :H, :W]
            out_p = os.path.join(out_dir, f"{name}.png")
            imageio.imwrite(out_p, tensor_to_uint8(y))
            print(f"  {name}: OK", flush=True)
            ok += 1
        except Exception as e:
            print(f"  {name}: FAIL -- {e}", flush=True)
            traceback.print_exc()
    del m
    torch.cuda.empty_cache()
    return ok


# ---------------------------------------------------------------- URCSA (LOL variant)
def run_urcsa():
    print("\n===== URCSA =====", flush=True)
    out_dir = os.path.join(OUT_ROOT, "URCSA")
    os.makedirs(out_dir, exist_ok=True)

    URCSA_ROOT = "/home/ma-user/workspace/zihaohe/baselines/URCSA"
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "urcsa_lol_base_model",
        os.path.join(URCSA_ROOT, "lol", "base_model.py"))
    mod = importlib.util.module_from_spec(spec)
    lol_dir = os.path.join(URCSA_ROOT, "lol")
    sys.path.insert(0, lol_dir)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(lol_dir)

    # Apply same monkey-patch as v2 script for the LOL low_block.
    ConvBlock_cls = mod.ConvBlock
    low_block_cls = mod.low_block

    def _make_single(ConvBlock_cls):
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

    orig_init = low_block_cls.__init__
    def new_init(self, dim=64, device="cuda:0"):
        orig_init(self, dim, device)
        self.ende_quarres_up_deline = ConvBlock_cls(4 * dim, 2 * dim, 1, 1, 0, bias=True)
        self.ende_quarres_up_up_deline = ConvBlock_cls(2 * dim, dim, 1, 1, 0, bias=True)
    single = _make_single(ConvBlock_cls)
    def new_forward(self, f_endecoder):
        return single(self, f_endecoder)
    low_block_cls.__init__ = new_init
    low_block_cls.forward = new_forward

    m = mod.lowlightnet3()
    ckpt_path = os.path.join(URCSA_ROOT, "lol", "checkpoint.pth")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    r = m.load_state_dict(sd, strict=False)
    print(f"  [URCSA LOL load] missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}", flush=True)
    m = m.to(device).eval()

    ok = 0
    for name, lqfile, split in TARGETS:
        try:
            _, lq_t = load_lq_tensor(os.path.join(LQ_DIR, lqfile))
            with torch.no_grad():
                xp, H, W = pad_to_mult(lq_t, 8)
                y = m(xp)[:, :, :H, :W].clamp(0, 1)
            out_p = os.path.join(out_dir, f"{name}.png")
            imageio.imwrite(out_p, tensor_to_uint8(y))
            print(f"  {name}: OK", flush=True)
            ok += 1
        except Exception as e:
            print(f"  {name}: FAIL -- {e}", flush=True)
            traceback.print_exc()
    del m
    torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = {}
    for fn, key in [(run_retinexformer, "Retinexformer"),
                    (run_hvi_cidnet,    "HVI_CIDNet"),
                    (run_darkir,        "DarkIR"),
                    (run_urcsa,         "URCSA")]:
        try:
            results[key] = fn()
        except Exception as e:
            print(f"\n!!! {key} crashed: {e}", flush=True)
            traceback.print_exc()
            results[key] = -1

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"  {k}: {v}/3 OK")
