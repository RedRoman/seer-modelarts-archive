"""Model complexity comparison for Seer and 5 baselines.

For each method:
  - Params (M): from state_dict / model.parameters()
  - GFLOPs: fvcore FlopCountAnalysis on one representative input
  - Peak GPU memory (MB): torch.cuda.max_memory_allocated during single fwd
  - Wall-clock (ms): mean over N=20 fwds after 5 warmup, GPU-synced

Inputs:
  - Video methods (Seer full): (1, 3, 16, 512, 512)
  - Image methods (Retinexformer, HVI-CIDNet, DarkIR, URCSA): (1, 3, 512, 512)

Task A (M1/M3 LOO): SKIPPED — source code for v35 model class was deleted
from vjepa_llve/models. Only checkpoint state_dict remains; the class had
custom `dec_igab`, `illu_estimator`, `illu_proj`, `router_logits`, `cf_refiner`
modules whose init signatures we cannot recover reliably. See REPORT.md.

Seer complexity: we can only report parameter count (from state_dict, which
we do have). GFLOPs / memory / timing require instantiating the model, which
we cannot. We report V-JEPA-2.1 ViT-L encoder + summed checkpoint tensors,
with clear caveat.
"""
from __future__ import annotations

import os, sys, json, time, math, gc
from typing import Callable, Dict, Any, Optional

import torch
# Break torch._dynamo <-> einops circular-import bug: import _dynamo first
# so that einops's _torch_specific.allow_ops_in_compiled_graph() finds a
# fully-initialised module.
try:
    import torch._dynamo  # noqa: F401
except Exception:
    pass
# Now warm up einops so subsequent rearrange() calls skip the compiled-graph patch.
try:
    import einops  # noqa: F401
    _ = einops.rearrange(torch.zeros(1, 2), "a b -> b a")
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_VJ_REPO = "/home/ma-user/workspace/zihaohe/vjepa_llve"
_BASE = "/home/ma-user/workspace/zihaohe/baselines"
sys.path.insert(0, _VJ_REPO)
sys.path.insert(0, _BASE)

DEVICE = torch.device("cuda")
OUT = "/home/ma-user/workspace/zihaohe/quick_ablations/model_complexity.json"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def count_params(module_or_sd) -> int:
    if isinstance(module_or_sd, dict):
        return sum(v.numel() for v in module_or_sd.values() if torch.is_tensor(v))
    return sum(p.numel() for p in module_or_sd.parameters())


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure(model_fn: Callable[[], Any],
            forward: Callable[[Any, torch.Tensor], torch.Tensor],
            input_tensor: torch.Tensor,
            warmup: int = 5, iters: int = 20,
            use_fvcore: bool = True) -> Dict[str, Any]:
    """Build model, run FLOPs + memory + timing."""
    torch.cuda.empty_cache(); gc.collect()

    model = model_fn()
    model = model.to(DEVICE).eval()
    x = input_tensor.to(DEVICE)

    # ---- Params ----
    params = count_params(model)

    result: Dict[str, Any] = {
        "params": params,
        "params_M": params / 1e6,
    }

    # ---- FLOPs ----
    gflops = None
    flop_err = None
    if use_fvcore:
        try:
            from fvcore.nn import FlopCountAnalysis
            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            flops = fca.total()
            gflops = flops / 1e9
        except Exception as e:
            flop_err = f"fvcore: {type(e).__name__}: {e}"
            try:
                from thop import profile
                macs, _ = profile(model, inputs=(x,), verbose=False)
                gflops = 2 * macs / 1e9  # rough FLOPs=2*MACs
                flop_err += " ; thop-fallback OK"
            except Exception as e2:
                flop_err += f" ; thop: {type(e2).__name__}: {e2}"

    result["gflops"] = gflops
    if flop_err:
        result["flop_error"] = flop_err

    # ---- Memory + Timing ----
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = forward(model, x)
        torch.cuda.synchronize()

        # peak mem: fresh measure on single fwd
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        _ = forward(model, x)
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()

        # timing
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            _ = forward(model, x)
            ends[i].record()
        torch.cuda.synchronize()
        times_ms = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
        # discard 20% highest+lowest for robustness
        times_ms.sort()
        k = max(1, iters // 5)
        core = times_ms[k:-k] if iters > 2 * k else times_ms
        mean_ms = sum(core) / len(core)

    result["peak_mem_bytes"] = int(peak_bytes)
    result["peak_mem_MB"] = peak_bytes / 1024 / 1024
    result["time_ms_mean"] = mean_ms
    result["time_ms_min"] = min(times_ms)
    result["time_ms_max"] = max(times_ms)
    result["iters"] = iters

    # cleanup
    del model
    torch.cuda.empty_cache(); gc.collect()
    return result


# ------------------------------------------------------------------
# 1. Retinexformer
# ------------------------------------------------------------------
def build_retinexformer():
    sys.path.insert(0, "/cache/zihaohe/ext_methods/Retinexformer/basicsr/models/archs")
    from RetinexFormer_arch import RetinexFormer
    m = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                      num_blocks=[1, 2, 2])
    return m


def fwd_retinexformer(model, x):
    return model(x)


# ------------------------------------------------------------------
# 2. HVI-CIDNet
# ------------------------------------------------------------------
def build_hvi_cidnet():
    sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/HVI-CIDNet")
    from net.CIDNet import CIDNet
    m = CIDNet()
    return m


def fwd_hvi_cidnet(model, x):
    # HVI-CIDNet expects 16-multiple input; 512 is already 16-multiple.
    return model(x)


# ------------------------------------------------------------------
# 3. DarkIR
# ------------------------------------------------------------------
def build_darkir():
    sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/DarkIR")
    from archs.DarkIR import DarkIR
    m = DarkIR(img_channel=3, width=32,
               middle_blk_num_enc=2, middle_blk_num_dec=2,
               enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
               dilations=[1, 4, 9], extra_depth_wise=True)
    return m


def fwd_darkir(model, x):
    # padder=8, 512 is fine
    return model(x)


# ------------------------------------------------------------------
# 4. URCSA (SDSD variant - uses (lq, lq, mode="test") signature)
# ------------------------------------------------------------------
def build_urcsa_sdsd():
    _URCSA = "/home/ma-user/workspace/zihaohe/baselines/URCSA"
    sys.path.insert(0, os.path.join(_URCSA, "sdsd"))
    from sdsd_basemodel import lowlightnet3, low_block, ConvBlock
    # replicate the deline patch from run_urcsa.py so state_dict *would* load
    orig_init = low_block.__init__
    def new_init(self, dim=64, device="cuda:0"):
        orig_init(self, dim, device)
        self.ende_quarres_up_deline = ConvBlock(4 * dim, 2 * dim, 1, 1, 0, bias=True)
        self.ende_quarres_up_up_deline = ConvBlock(2 * dim, dim, 1, 1, 0, bias=True)
    def new_forward(self, f):
        fullres = f
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
    def new_wrapper(self, f_endecoder, f_endecoder2=None, mode="train"):
        o = new_forward(self, f_endecoder); return o, o
    low_block.__init__ = new_init
    low_block.forward = new_wrapper
    m = lowlightnet3()

    # Wrap to unify signature for FLOPs (single-input, single-output).
    class URCSAWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out, _ = self.base(x, x, mode="test")
            return out
    return URCSAWrapper(m)


def fwd_urcsa(model, x):
    return model(x)


# ------------------------------------------------------------------
# 5. MambaLLIE (arch only — no weights, but arch instantiable)
# ------------------------------------------------------------------
def build_mambaliie():
    ML = "/home/ma-user/workspace/zihaohe/baselines/MambaLLIE"
    sys.path.insert(0, ML)
    sys.path.insert(0, os.path.join(ML, "basicsr"))
    from basicsr.archs.MambaLLIE_arch import MambaLLIE
    m = MambaLLIE()
    return m


def fwd_mambaliie(model, x):
    return model(x)


# ------------------------------------------------------------------
# 6. BVI-Mamba (arch only) — class name is STASUNet, input is (B, T, C, H, W)
# ------------------------------------------------------------------
def build_bvi_mamba():
    BM = "/home/ma-user/workspace/zihaohe/baselines/BVI-Mamba"
    sys.path.insert(0, BM)
    sys.path.insert(0, os.path.join(BM, "arch"))
    from arch.BVIMamba import STASUNet
    # Match BVIMamba.yml config (paper's official settings): num_feat=embed_dim=16,
    # depths=[8,8,8,8], num_heads=[8,8,8,8], window_size=8, patch_norm=True,
    # num_frames=5, hr_in=True, img_size=512
    m = STASUNet(
        num_in_ch=3, num_out_ch=3, num_feat=16, embed_dim=16,
        num_frame=5, deformable_groups=8, num_extract_block=5,
        num_reconstruct_block=10, hr_in=True, img_size=[512, 512],
        patch_size=4, depths=[8, 8, 8, 8], num_heads=[8, 8, 8, 8],
        window_size=8, patch_norm=True,
    )
    return m


def fwd_bvi_mamba(model, x):
    return model(x)


# ------------------------------------------------------------------
# Seer: params only (from state_dict) + VJEPA-2.1 encoder from separate weight
# ------------------------------------------------------------------
def measure_seer_partial() -> Dict[str, Any]:
    ck = torch.load("/cache/zihaohe/checkpoint/runs/v35_S4_indoor_ft/best_y512.pt",
                    map_location="cpu", weights_only=False)
    pb = ck["pixel_branch"]
    cf = ck["cf_refiner"]

    pb_params = sum(v.numel() for v in pb.values() if torch.is_tensor(v))
    cf_params = sum(v.numel() for v in cf.values() if torch.is_tensor(v))
    trainable = pb_params + cf_params

    # V-JEPA 2.1 ViT-L: fixed 302M frozen (as per PROJECT MEMORY)
    # Try to load actual weight file to verify
    frozen_params = None
    vjepa_note = "302M nominal (V-JEPA 2.1 ViT-L, frozen)"
    for cand in ["/cache/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt",
                 "/cache/zihaohe/modelzoo/vjepa2/vitl-2p1.pth",
                 "/cache/zihaohe/modelzoo/vjepa2/vjepa2.1_vitl.pth"]:
        if os.path.exists(cand):
            try:
                vc = torch.load(cand, map_location="cpu", weights_only=False)
                if isinstance(vc, dict):
                    # Try each top-level dict that's stateful
                    for key in ("encoder", "target_encoder", "ema_encoder", "state_dict"):
                        if key in vc and isinstance(vc[key], dict):
                            frozen_params = sum(v.numel() for v in vc[key].values()
                                                if torch.is_tensor(v))
                            vjepa_note = f"measured from {cand}[{key}]"
                            break
                    if frozen_params is None:
                        # Assume top-level is the state dict
                        tv = [v for v in vc.values() if torch.is_tensor(v)]
                        if tv:
                            frozen_params = sum(v.numel() for v in tv)
                            vjepa_note = f"measured from {cand} top-level"
                break
            except Exception as e:
                vjepa_note = f"load-fail {cand}: {e}"
    if frozen_params is None:
        # fallback: V-JEPA 2 ViT-L has ~303M params
        frozen_params = 303_000_000
        vjepa_note += " (fallback constant)"

    total = trainable + frozen_params
    return {
        "params": total,
        "params_M": total / 1e6,
        "trainable_params": trainable,
        "trainable_params_M": trainable / 1e6,
        "frozen_params": frozen_params,
        "frozen_params_M": frozen_params / 1e6,
        "pixel_branch_params_M": pb_params / 1e6,
        "cf_refiner_params_M": cf_params / 1e6,
        "gflops": None,
        "peak_mem_MB": None,
        "time_ms_mean": None,
        "note": ("Only params measured. v35 model source deleted from vjepa_llve; "
                 "cannot instantiate for GFLOPs/memory/timing. "
                 f"V-JEPA encoder: {vjepa_note}."),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
CFG = [
    ("Retinexformer", build_retinexformer, fwd_retinexformer,
     (1, 3, 512, 512), "1x3x512x512 (1 frame)"),
    ("HVI-CIDNet", build_hvi_cidnet, fwd_hvi_cidnet,
     (1, 3, 512, 512), "1x3x512x512 (1 frame)"),
    ("DarkIR", build_darkir, fwd_darkir,
     (1, 3, 512, 512), "1x3x512x512 (1 frame)"),
    ("URCSA", build_urcsa_sdsd, fwd_urcsa,
     (1, 3, 512, 512), "1x3x512x512 (1 frame; single-image variant)"),
    ("MambaLLIE", build_mambaliie, fwd_mambaliie,
     (1, 3, 512, 512), "1x3x512x512 (1 frame; no weights on host)"),
    ("BVI-Mamba", build_bvi_mamba, fwd_bvi_mamba,
     (1, 5, 3, 512, 512), "1x5x3x512x512 (5-frame clip, native BVI-Mamba stride; no weights on host)"),
]


def main():
    results = {}
    method_names = [c[0] for c in CFG]
    print(f"[main] measuring {len(CFG)} baselines on {torch.cuda.get_device_name(0)}")
    for name, build, fwd, shape, input_desc in CFG:
        print(f"\n===== {name}  input={input_desc} =====", flush=True)
        try:
            x = torch.randn(*shape)
            r = measure(build, fwd, x)
            r["input"] = input_desc
            results[name] = r
            print(f"  params={r['params_M']:.2f}M  "
                  f"GFLOPs={('%.2f'%r['gflops']) if r['gflops'] is not None else 'NaN'}  "
                  f"mem={r['peak_mem_MB']:.1f}MB  time={r['time_ms_mean']:.2f}ms")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"error": f"{type(e).__name__}: {e}",
                             "input": input_desc,
                             "params": None, "params_M": None,
                             "gflops": None, "peak_mem_MB": None,
                             "time_ms_mean": None}
            print(f"  FAILED: {e}")

    # Seer partial
    print("\n===== Seer (params only) =====", flush=True)
    seer = measure_seer_partial()
    seer["input"] = "1x3x16x512x512 (16-frame clip)"
    results["Seer"] = seer
    print(f"  total={seer['params_M']:.2f}M  "
          f"(frozen={seer['frozen_params_M']:.2f}M + trainable={seer['trainable_params_M']:.2f}M)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
