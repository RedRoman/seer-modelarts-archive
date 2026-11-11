"""Check enc_s4 attention consistency across frames within the clip, and
check per-head variation (is the concentrated pattern a property of most
heads, or just 1-2 heads dominating the post-mean-average)?
"""
import torch

save_dir = "/home/ma-user/workspace/zihaohe/visualizations/fig_cross_attention"
attn = torch.load(f"{save_dir}/attn_enc_s4.pt")  # [BT=16, h=16, HW_q=256, M=256]
BT, h, HW, M = attn.shape
Hq = Wq = 16

# Check argmax stability for the "bright query" and "dark query" cell across all 16 frames
# From debug run: bright query was top-right area, dark query bottom-right.
# We'll just check a fixed query index e.g. row=2,col=10 (bright-ish, top area) across frames.
for label, (r, c) in [("bright_top", (2, 10)), ("dark_bottomright", (13, 14)), ("center_obj", (8, 8))]:
    qidx = r * Wq + c
    argmaxes = []
    maxvals = []
    for t in range(BT):
        head_avg = attn[t].mean(dim=0)  # [HW,M]
        dist = head_avg[qidx]
        argmaxes.append(dist.argmax().item())
        maxvals.append(dist.max().item())
    print(f"{label} (q={r},{c}): argmax per frame = {argmaxes}")
    print(f"   max attn val per frame = {[f'{v:.3f}' for v in maxvals]}")

print()
print("Per-head check at frame 8, query=(13,14) [dark_bottomright]:")
qidx = 13*16+14
for head in range(h):
    dist = attn[8, head, qidx]
    print(f"  head {head}: argmax={dist.argmax().item()}, max={dist.max().item():.4f}, "
          f"entropy_frac={(-dist.clamp_min(1e-9)*dist.clamp_min(1e-9).log()).sum().item()/torch.log(torch.tensor(256.0)).item():.3f}")
