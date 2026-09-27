"""Inspect captured attention maps for real spatial structure (sanity check
before building the paper figure). Prints statistics; does NOT fabricate
anything -- if maps look uniform/noisy we report that honestly.
"""
import json
import numpy as np
import torch

save_dir = "/home/ma-user/workspace/zihaohe/visualizations/fig_cross_attention"

meta = json.load(open(save_dir + "/meta.json"))
print(json.dumps(meta, indent=2))

for tag in ["enc_s1", "enc_s2", "enc_s3", "enc_s4", "dec_s3", "dec_s2", "dec_s1"]:
    attn = torch.load(f"{save_dir}/attn_{tag}.pt")  # [BT, heads, HW_query, M_kv=256]
    BT, h, HW, M = attn.shape
    # average over heads and over batch*time (we used B=1,T=16 -> BT=16)
    a = attn.mean(dim=1)  # [BT, HW, M]
    # for frame 8 (middle), check entropy / max-vs-uniform of attention distributions
    frame = a[8]  # [HW, M]
    uniform = 1.0 / M
    max_val = frame.max(dim=-1).values.mean().item()
    mean_val = frame.mean().item()
    std_over_kv = frame.std(dim=-1).mean().item()
    entropy = (-frame.clamp_min(1e-12) * frame.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
    max_entropy = np.log(M)
    print(f"{tag}: shape={tuple(attn.shape)}  mean_attn={mean_val:.6f} (uniform={uniform:.6f})  "
          f"max_attn(avg over queries)={max_val:.6f}  std_over_kv={std_over_kv:.6f}  "
          f"entropy={entropy:.3f}/{max_entropy:.3f} ({100*entropy/max_entropy:.1f}% of uniform-entropy)")
