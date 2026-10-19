"""End-to-end NPU test of the full Seer training step with synthetic data.

Exercises exactly the Stage-4 pipeline of tools/v35_stage4_ctrl.py:
  frozen V-JEPA on LQ and HQ -> M1 refiner (LoRA predictor) -> pixel branch
  -> charbonnier + retinex + TV + M2, backward, AdamW step.
No dataset needed: LQ/HQ are random clips. Reports per-step time and memory so
the E1 schedule can be budgeted before SDSD arrives.
"""
import os
import sys
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

CODE = "/temp/zihaohe/seer_cvpr/code/vjepa_llve"
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "tools"))

from losses.v6 import charbonnier_loss  # noqa: E402
from losses.v7 import illumination_smoothness_loss, retinex_compose_loss  # noqa: E402
from models.pixel_branch_v10 import VJepaPixelBranch  # noqa: E402
from models.predictor_lora import CounterfactualRefiner  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402

REPO = "/temp/zihaohe/seer_cvpr/modelzoo/vjepa2-main"
CKPT = "/temp/zihaohe/seer_cvpr/weights/vjepa2_1_vitl_dist_vitG_384.pt"
W_PIX, W_RET, W_SMO, W_JEP = 1.5, 0.15, 0.02, 0.05
CROP, T, ACCUM = 256, 16, 8

dev = torch.device("npu:0")
torch.manual_seed(0)

bb = VJepaBackboneLocal21(local_repo=REPO, local_ckpt=CKPT, feature_layers=(5, 11, 17, 23),
                          image_size=256, frames_per_clip=T, patch_size=16, tubelet_size=2,
                          freeze=True).to(dev).eval()
for p in bb.parameters():
    p.requires_grad_(False)
rf = CounterfactualRefiner(local_repo=REPO, ckpt_path=CKPT, encoder_embed_dim=1024,
                           image_size=256, num_frames=T, patch_size=16, tubelet_size=2,
                           predictor_embed_dim=384, depth=12, num_heads=12,
                           lora_rank=16, freeze_body=True).to(dev)
pb = VJepaPixelBranch(in_channels=3, vjepa_dim=1024, vjepa_grid=(T // 2, 16, 16),
                      num_vjepa_layers=4, cross_attn_at_scales=(1, 2, 3, 4),
                      use_cfim_bottleneck=True, use_illumination_cues=True,
                      use_gradient_checkpoint=True, router_init_c=4.0).to(dev)
trainable = [p for p in rf.parameters() if p.requires_grad] + \
            [p for p in pb.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=1e-4, betas=(0.9, 0.99))
pb.train(); rf.train()

ts = []
for it in range(ACCUM * 2):
    t0 = time.time()
    lq = torch.rand(1, 3, T, CROP, CROP, device=dev)
    gt = torch.rand(1, 3, T, CROP, CROP, device=dev)
    with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
        zl = bb(lq)["feats"]
        zh = bb(gt)["feats"][3]
    zl = [z.float() for z in zl]; zh = zh.float()
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        ref = rf(zl[3].bfloat16())
    ref_f = ref.float()
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        out = pb(lq, list(zl[:3]) + [ref_f])
    pred, R, L = out["pred_pixels"].float(), out["R_hat"].float(), out["L_hat"].float()
    loss = (W_PIX * charbonnier_loss(pred, gt) + W_RET * retinex_compose_loss(R, L, gt)
            + W_SMO * illumination_smoothness_loss(L) + W_JEP * F.l1_loss(ref_f, zh))
    (loss / ACCUM).backward()
    if (it + 1) % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.npu.synchronize()
    ts.append(time.time() - t0)
    print(f"micro {it}: {ts[-1]:.2f}s loss={loss.item():.4f} "
          f"mem={torch.npu.max_memory_allocated()/2**30:.1f}GiB", flush=True)

avg = sum(ts[2:]) / len(ts[2:])
print(f"E2E_OK avg micro {avg:.2f}s -> opt step {ACCUM*avg:.1f}s "
      f"peak {torch.npu.max_memory_allocated()/2**30:.1f}GiB", flush=True)
print(f"1500 steps ~= {ACCUM*avg*1500/3600:.1f} h/run", flush=True)
