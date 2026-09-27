"""Rerun Retinexformer on exact-crop LQ patches supplied by the local process."""
import os, sys
sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/cache/zihaohe/ext_methods/Retinexformer/basicsr/models/archs")
import numpy as np, torch, imageio.v2 as imageio
from RetinexFormer_arch import RetinexFormer

NET = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])

def load(w):
    m = RetinexFormer(**NET)
    ck = torch.load(w, map_location="cpu", weights_only=False)
    sd = ck.get("params", ck.get("params_ema", ck))
    m.load_state_dict(sd, strict=True)
    return m.cuda().eval()

indoor = load("/cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_indoor.pth")
outdoor = load("/cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_outdoor.pth")

@torch.no_grad()
def infer(model, hwc_u8):
    x = torch.from_numpy(hwc_u8).float().permute(2,0,1).unsqueeze(0)/255.0
    # RetinexFormer needs H,W divisible; pad to next multiple of 4
    _, _, H, W = x.shape
    ph = (4 - H % 4) % 4; pw = (4 - W % 4) % 4
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')
    y = model(x.cuda())[:, :, :H, :W]
    return (y.clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

TARGETS = [
    ("/tmp/seer_lq_patches/outdoor_MVI_1007_2_t08_lq_matched.png",
     "/tmp/retinex_viz/outdoor_MVI_1007_2_t08_retinex.png", outdoor),
    ("/tmp/seer_lq_patches/bvi_S40_Books5__10fps_t08_lq_matched.png",
     "/tmp/retinex_viz/bvi_S40_Books5__10fps_t08_retinex.png", indoor),
    ("/tmp/seer_lq_patches/lolv2real_00690_lq_matched.png",
     "/tmp/retinex_viz/lolv2real_00690_retinex.png", indoor),
]

for lq_p, out_p, model in TARGETS:
    lq = imageio.imread(lq_p)
    if lq.shape[-1] == 4: lq = lq[..., :3]
    pred = infer(model, lq)
    imageio.imwrite(out_p, pred)
    print(f"{os.path.basename(lq_p)}  {lq.shape} → {pred.shape}")
