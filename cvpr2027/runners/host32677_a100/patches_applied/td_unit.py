"""Does the E24 form actually refuse the collapse that E22 took?

Fed the two states measured by td_probe.py -- the converged baseline's
(mag 1.063, cos +0.034) and E22 step 5000's (mag 0.202, cos +0.014) -- the old
term must prefer the second and the new term must prefer the first, or the fix
does not address the diagnosis.
"""
import sys
import torch
sys.path.insert(0, "/cache/zihaohe/seer_cvpr/code/vjepa_llve")
from losses.temporal_delta import temporal_delta_loss, temporal_delta_cosine_loss

g = torch.Generator().manual_seed(0)
grid = (8, 16, 16)
B, N, D = 1, 8 * 16 * 16, 1024

# build a target whose deltas are nonzero, then synthesise predictions with a
# chosen magnitude ratio and near-zero alignment, as measured
gt = torch.randn(B, N, D, generator=g)
gt = gt.reshape(B, 8, 256, D).cumsum(1).reshape(B, N, D)   # smooth in time


def synth(mag, align):
    """tokens whose deltas have |dp|/|dg| ~= mag and cos(dp,dg) ~= align"""
    noise = torch.randn(B, N, D, generator=g)
    noise = noise.reshape(B, 8, 256, D).cumsum(1).reshape(B, N, D)
    from losses.temporal_delta import tubelet_deltas
    dn = tubelet_deltas(noise, grid).abs().mean()
    dg = tubelet_deltas(gt, grid).abs().mean()
    mixed = align * gt + (1 - align) * noise * float(dg / dn)
    dm = tubelet_deltas(mixed, grid).abs().mean()
    return mixed * float(mag * dg / dm)


print("%-26s %8s %8s" % ("state", "relative", "cosine"))
for name, mag, align in (("baseline  mag 1.06 cos .03", 1.063, 0.03),
                         ("E22 @5k   mag 0.20 cos .01", 0.202, 0.01),
                         ("collapsed mag 0.00        ", 0.000, 0.00),
                         ("aligned   mag 1.00 cos .90", 1.000, 0.90)):
    p = synth(mag, align) if mag > 0 else torch.zeros(B, N, D)
    r = float(temporal_delta_loss(p, gt, grid))
    c = float(temporal_delta_cosine_loss(p, gt, grid))
    print("%-26s %8.3f %8.3f" % (name, r, c))
print("\nrelative must rank collapse best; cosine must rank it worst.")
