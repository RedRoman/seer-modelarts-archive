import sys, os, time, torch, numpy as np
sys.path.insert(0, '/temp/zihaohe/workspace/vjepa_llve'); sys.path.insert(0, '/temp/zihaohe/workspace/vjepa_llve/tools')
import v35_stage4_train as T
from models.vjepa_backbone import VJepaBackboneLocal21
from models.predictor_lora import CounterfactualRefiner
from models.pixel_branch_v10 import VJepaPixelBranch
key = sys.argv[1]
dev = torch.device('cuda')
bb = VJepaBackboneLocal21(local_repo=T.VJEPA_REPO, local_ckpt='', feature_layers=(5,11,17,23), image_size=256, frames_per_clip=16, patch_size=16, tubelet_size=2, freeze=True)
s = torch.load(T.VJEPA_CKPT, map_location='cpu', weights_only=False)[key]
s = {k.replace('module.backbone.',''):v for k,v in s.items()}
print('load', key, bb.model.load_state_dict(s, strict=False))
bb = bb.to(dev).eval()
rf = CounterfactualRefiner(local_repo=T.VJEPA_REPO, ckpt_path=T.VJEPA_CKPT, encoder_embed_dim=1024, image_size=256, num_frames=16, patch_size=16, tubelet_size=2, predictor_embed_dim=384, depth=12, num_heads=12, lora_rank=16, freeze_body=True).to(dev)
pb = VJepaPixelBranch(in_channels=3, vjepa_dim=1024, vjepa_grid=(8,16,16), num_vjepa_layers=4, cross_attn_at_scales=(1,2,3,4), use_cfim_bottleneck=True, use_illumination_cues=True, use_gradient_checkpoint=True, router_init_c=4.0).to(dev)
st = torch.load('/cache/zihaohe/checkpoint/runs/v35_stage3/final.pt', map_location='cpu', weights_only=False)
print(pb.load_state_dict(st['pixel_branch'], strict=False)[0][:3], rf.load_state_dict(st['cf_refiner'], strict=False)[0][:3])
loader = T.build_train_loader(256, 16, 2)
it = iter(loader); ls = []
with torch.no_grad():
    for _ in range(4):
        b = next(it); lq, gt = b['lq'].to(dev), b['hq'].to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            zl = bb(lq)['feats'][3].float(); zh = bb(gt)['feats'][3].float()
        ls.append(torch.nn.functional.l1_loss(zl, zh).item())
print(key, 'jepa L1 (gate=0) on 4 train clips:', np.round(ls,3), 'mean', np.mean(ls))
t=time.time(); rgb, y = T.audit_indoor_512(bb, rf, pb, T.build_audit_seqs(), dev); print(key, f'zero-iter audit RGB-512={rgb:.3f} Y-512={y:.3f} ({time.time()-t:.0f}s)')
