"""Deep-dive on enc_s4 (coarsest scale, most concentrated attention) and
enc_s1 (finest xattn scale) -- check whether attention argmax varies
spatially across query positions (real localization) vs is dominated by
one or two global tokens for every query (degenerate / not spatially
informative despite being non-uniform).
"""
import json
import numpy as np
import torch

save_dir = "/home/ma-user/workspace/zihaohe/visualizations/fig_cross_attention"

for tag in ["enc_s4", "enc_s1", "dec_s3"]:
    attn = torch.load(f"{save_dir}/attn_{tag}.pt")  # [BT, heads, HW_q, M=256]
    BT, h, HW, M = attn.shape
    a = attn.mean(dim=1)  # avg over heads -> [BT, HW, M]
    frame = a[8]  # middle frame [HW, M]
    Hq = Wq = int(HW ** 0.5)
    argmax = frame.argmax(dim=-1).reshape(Hq, Wq)  # which of 256 V-JEPA tokens each query picks
    n_unique = len(torch.unique(argmax))
    # spatial correlation: does argmax vary smoothly / match a spatial pattern,
    # or is it dominated by 1-2 tokens everywhere?
    vals, counts = torch.unique(argmax, return_counts=True)
    top_frac = counts.max().item() / argmax.numel()
    print(f"{tag}: query grid {Hq}x{Wq}, argmax uses {n_unique}/{M} unique V-JEPA tokens, "
          f"most popular token used by {100*top_frac:.1f}% of queries")
    # Also: correlate argmax spatial position (as 16x16 vjepa grid coords) with query position
    vjepa_r = (argmax // 16).float()
    vjepa_c = (argmax % 16).float()
    q_r = torch.arange(Hq).view(-1, 1).expand(Hq, Wq).float()
    q_c = torch.arange(Wq).view(1, -1).expand(Hq, Wq).float()
    # normalize both to [0,1] and compute correlation
    q_r_n = q_r / (Hq - 1); q_c_n = q_c / (Wq - 1)
    v_r_n = vjepa_r / 15.0; v_c_n = vjepa_c / 15.0
    def corr(x, y):
        x = x.flatten(); y = y.flatten()
        x = x - x.mean(); y = y - y.mean()
        return (x*y).sum() / (x.norm()*y.norm() + 1e-8)
    cr = corr(q_r_n, v_r_n).item()
    cc = corr(q_c_n, v_c_n).item()
    print(f"    spatial corr: row corr(query_row, argmax_vjepa_row)={cr:.3f}, "
          f"col corr={cc:.3f}  (near 1.0 = argmax tracks query position = spatially localized attn)")
