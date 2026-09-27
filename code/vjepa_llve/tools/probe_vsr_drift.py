"""Measure latent drift for VSR (bicubic x4) in the Wan latent space, to be
compared against the LLVE numbers from probe_latent_scale.py:
    LLVE:  std(z_lq)=0.741  std(z_hq)=0.649  std(drift)=0.537
           RMS(drift)/RMS(z_hq) = 0.828
If VSR drift is an order of magnitude smaller, sigma_0 must be rescaled
rather than transplanted.
"""
import os, sys, glob
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.abspath("."))
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKLWan

WAN = "/cache/zihaohe/checkpoint/wan21_t2v_1_3b"
REDS = "/temp/zihaohe/cache/datasets/REDS/train_sharp"
N_CLIPS, T, CROP, SCALE = 12, 16, 256, 4
dev = "cuda"

def enc(vae, x, m, s):
    return (vae.encode(x*2-1).latent_dist.mode() - m) * s

def main():
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae",
                                           torch_dtype=torch.float32).to(dev).eval()
    m = torch.tensor(vae.config.latents_mean, device=dev).view(1,16,1,1,1).float()
    s = (1.0/torch.tensor(vae.config.latents_std, device=dev).view(1,16,1,1,1)).float()

    clips = sorted(os.listdir(REDS))
    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("z_lq","z_hq","drift")}
    print(f"REDS clips: {len(clips)}")

    for i in range(N_CLIPS):
        c = clips[int(rng.integers(0, len(clips)))]
        fs = sorted(glob.glob(f"{REDS}/{c}/*.png"))
        st = int(rng.integers(0, max(1, len(fs)-T)))
        fr = []
        for f in fs[st:st+T]:
            im = np.array(Image.open(f).convert("RGB"), dtype=np.float32)/255.
            fr.append(im)
        hq = torch.from_numpy(np.stack(fr)).permute(3,0,1,2).unsqueeze(0).to(dev)
        # random crop to CROP x CROP
        _,_,_,H,W = hq.shape
        y = int(rng.integers(0, H-CROP)); x = int(rng.integers(0, W-CROP))
        hq = hq[:,:,:,y:y+CROP, x:x+CROP]
        # standard BI degradation: down x4 then bicubic back up
        B,C,Tn,H2,W2 = hq.shape
        flat = hq.permute(0,2,1,3,4).reshape(B*Tn,C,H2,W2)
        lr  = F.interpolate(flat, scale_factor=1/SCALE, mode="bicubic", align_corners=False, antialias=True)
        up  = F.interpolate(lr, size=(H2,W2), mode="bicubic", align_corners=False).clamp(0,1)
        lq  = up.reshape(B,Tn,C,H2,W2).permute(0,2,1,3,4)

        with torch.no_grad():
            zl, zh = enc(vae,lq,m,s), enc(vae,hq,m,s)
        d = zh - zl
        acc["z_lq"].append(zl.std().item()); acc["z_hq"].append(zh.std().item())
        acc["drift"].append(d.std().item())
        print(f"  [{i:2d}] clip {c} shape={tuple(zl.shape)} "
              f"std(z_lq)={zl.std():.3f} std(z_hq)={zh.std():.3f} std(drift)={d.std():.3f}")

    print("\n===== VSR x4 (REDS) =====")
    for k,v in acc.items():
        print(f"  {k:6s} std: mean={np.mean(v):.4f} min={np.min(v):.4f} max={np.max(v):.4f}")
    r = np.mean(acc["drift"])/np.mean(acc["z_hq"])
    print(f"\n  RMS(drift)/RMS(z_hq) = {r:.4f}     [LLVE was 0.8279]")
    print(f"  ratio vs LLVE        = {r/0.8279:.3f}x")
    print(f"\n  LLVE best sigma_0 = 0.1 -> scaled VSR sigma_0 ~= {0.1*np.mean(acc['drift'])/0.5371:.4f}")
    print(f"  suggested sweep: {[round(0.1*np.mean(acc['drift'])/0.5371*f,4) for f in (0.5,1,2,4)]}")

main()
