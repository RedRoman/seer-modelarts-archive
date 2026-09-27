"""Dump Retinexformer outputs on 3 representative frames for qualitative
comparison. Saves PNG triples (LQ + Retinex + GT) to /tmp/retinex_viz/."""
import os, sys, json
sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/cache/zihaohe/ext_methods/Retinexformer/basicsr/models/archs")
import numpy as np, torch, imageio.v2 as imageio
from RetinexFormer_arch import RetinexFormer
from common import eval_lib as E

OUT = "/tmp/retinex_viz"
os.makedirs(OUT, exist_ok=True)
NET = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])

def load(w):
    m = RetinexFormer(**NET)
    ck = torch.load(w, map_location="cpu", weights_only=False)
    sd = ck.get("params", ck.get("params_ema", ck))
    m.load_state_dict(sd, strict=True)
    return m.cuda().eval()

@torch.no_grad()
def infer(model, hwc_u8):
    x = torch.from_numpy(hwc_u8).float().permute(2,0,1).unsqueeze(0)/255.0
    y = model(x.cuda())
    return (y.clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

indoor = load("/cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_indoor.pth")
outdoor = load("/cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_outdoor.pth")

targets = []

# 1) SDSD-outdoor MVI_1007 frame 8 → outdoor ckpt (matched)
print(">>> SDSD-outdoor MVI_1007 f8", flush=True)
for seq, lqs, gts in E.iter_sdsd_outdoor():
    if seq in ("MVI_1007", "MVI_1007_2"):
        t = 8
        pred = infer(outdoor, lqs[t])
        imageio.imwrite(f"{OUT}/outdoor_{seq}_t{t:02d}_lq.png", lqs[t])
        imageio.imwrite(f"{OUT}/outdoor_{seq}_t{t:02d}_retinex.png", pred)
        imageio.imwrite(f"{OUT}/outdoor_{seq}_t{t:02d}_gt.png", gts[t])
        targets.append(f"outdoor_{seq}_t{t:02d}")
        break

# 2) BVI Books5 10fps frame 8 → indoor ckpt (cross-domain)
print(">>> BVI Books5 f8", flush=True)
try:
    for seq, lqs, gts in E.iter_bvi():
        if "Books5" in seq and "10fps" in seq:
            t = 8
            pred = infer(indoor, lqs[t])
            imageio.imwrite(f"{OUT}/bvi_{seq}_t{t:02d}_lq.png", lqs[t])
            imageio.imwrite(f"{OUT}/bvi_{seq}_t{t:02d}_retinex.png", pred)
            imageio.imwrite(f"{OUT}/bvi_{seq}_t{t:02d}_gt.png", gts[t])
            targets.append(f"bvi_{seq}_t{t:02d}")
            break
except Exception as e:
    print("bvi iter failed:", e)

# 3) LOL-v2 Real first image → indoor ckpt (cross-domain)
print(">>> LOL-v2 Real first", flush=True)
try:
    for name, lq_stk, gt_stk in E.iter_lol_v2("Real"):
        lq = lq_stk[0]; gt = gt_stk[0]
        pred = infer(indoor, lq)
        stem = name.replace("/", "_")
        imageio.imwrite(f"{OUT}/lolv2real_{stem}_lq.png", lq)
        imageio.imwrite(f"{OUT}/lolv2real_{stem}_retinex.png", pred)
        imageio.imwrite(f"{OUT}/lolv2real_{stem}_gt.png", gt)
        targets.append(f"lolv2real_{stem}")
        break
except Exception as e:
    print("lolv2 iter failed:", e)

print("wrote:", targets)
print("files in", OUT, ":", os.listdir(OUT))
