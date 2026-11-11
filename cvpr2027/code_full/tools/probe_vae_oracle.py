"""VAE reconstruction ceiling on the 11 SDSD held-out sequences.

The question: our best model sits at ~21.5 dB. How much of that is the model
failing, and how much is the frozen Wan VAE simply being unable to represent
the GT?

ORACLE = decode(encode(GT)) vs GT. No model, no LQ. This is the hard upper
bound for ANY method operating in this latent space with 1-NFE.

Uses the same eval protocol as the trainer (load_clip_eval, causal-VAE frame
trim, compute_iqa) so the numbers are directly comparable to the 21.5 dB.
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.abspath("."))
import numpy as np, torch
from diffusers import AutoencoderKLWan
from tools.v11_oracle_delta_probe import load_clip_eval, build_split_seqs, mean_psnr, to_u8
from tools.iqa_metrics import compute_iqa

WAN = "/cache/zihaohe/checkpoint/wan21_t2v_1_3b"
dev = "cuda"

vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae",
                                       torch_dtype=torch.float32).to(dev).eval()
m = torch.tensor(vae.config.latents_mean, device=dev).view(1,16,1,1,1).float()
s = (1.0/torch.tensor(vae.config.latents_std, device=dev).view(1,16,1,1,1)).float()

_, test_seqs = build_split_seqs()
print(f"held-out seqs: {len(test_seqs)}\n")

rows = []
for name, lq_p, gt_p in test_seqs:
    lq_t, gt_t = load_clip_eval(lq_p, gt_p, crop_size=256)
    gt = gt_t.to(dev)
    with torch.no_grad():
        z = (vae.encode(gt*2-1).latent_dist.mode() - m) * s     # encode GT
        rec = vae.decode(z/s + m).sample                         # decode back
        rec = ((rec+1)/2).clamp(0,1)
    T_out = rec.shape[2]; drop = (gt.shape[2]-T_out)//2
    gt_trim = gt[:, :, drop:drop+T_out]
    iqa = compute_iqa(rec, gt_trim)
    iqa['psnr'] = float(mean_psnr(to_u8(rec), to_u8(gt_trim)))
    rows.append((name, iqa))
    print(f"  {name}: PSNR={iqa['psnr']:.3f} SSIM={iqa['ssim']:.4f} "
          f"LPIPS={iqa['lpips']:.4f} DISTS={iqa['dists']:.4f}")

print("\n===== VAE ORACLE CEILING (11 SDSD held-out) =====")
for k in ("psnr","ssim","lpips","dists","clipiqa"):
    v=[r[1][k] for r in rows if k in r[1]]
    if v: print(f"  {k.upper():8s} = {np.mean(v):.4f}")
print(f"\n  our best model  = 21.53 dB")
print(f"  oracle ceiling  = {np.mean([r[1]['psnr'] for r in rows]):.3f} dB")
print(f"  headroom        = {np.mean([r[1]['psnr'] for r in rows])-21.53:.3f} dB")
