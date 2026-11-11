"""Is the target of the temporal-delta loss a stable quantity at all?

The baseline probe says the converged model's latent deltas have roughly the
right magnitude (1.06x the target's) and almost no alignment with it
(cos +0.034), and that the low-light input scores the same (+0.045). Two very
different situations produce that:

  A  the alignment is learnable and nothing has asked for it yet
  B  the delta field is dominated by a noise floor, in which case no two
     renderings of the same motion agree either, and the loss has no signal to
     offer however long it trains

B is testable without any model. The ground truth is re-encoded under changes
that leave the motion untouched and alter only appearance, and the deltas are
compared with the original's:

  noise   gt + N(0, 1/255)   a perturbation far smaller than the lq/gt gap
  gamma   gt ** 1.1          a mild photometric change, identical motion
  shift   gt rolled by 1 px  identical content, sub-token spatial offset

Whatever these score is the ceiling any model could reach. If they sit near the
+0.04 the model and the input both score, the ceiling is the floor.
"""
import argparse, json, os, sys
import numpy as np
import torch

CODE = "/cache/zihaohe/seer_cvpr/code/vjepa_llve"
for p in (CODE, os.path.join(CODE, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_e4 import held_out_sequences, REF_LAYER  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402

ROOT = "/cache/zihaohe/seer_cvpr"


def deltas(tok, grid):
    t_v, h_v, w_v = grid
    x = tok.reshape(tok.shape[0], t_v, h_v * w_v, -1)
    return x[:, 1:] - x[:, :-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device(a.device)
    amp = torch.bfloat16
    clip_len = 16
    grid = (clip_len // 2, a.crop // 16, a.crop // 16)

    bb = VJepaBackboneLocal21(
        local_repo=f"{ROOT}/modelzoo/vjepa2-main",
        local_ckpt=f"{ROOT}/weights/vjepa2_1_vitl_dist_vitG_384.pt",
        feature_layers=(5, 11, 17, 23), image_size=a.crop,
        frames_per_clip=clip_len, patch_size=16, tubelet_size=2,
        freeze=True).to(dev).eval()

    def tok(x01):
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp):
            return bb(x01)["feats"][REF_LAYER].float()

    g = torch.Generator(device="cpu").manual_seed(0)
    conds = ("noise", "gamma", "shift", "lq")
    acc = {c: {"td": [], "mag": [], "cos": []} for c in conds}

    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences("sdsd", "test", clip_len, a.crop)):
        if i >= a.clips:
            break
        def t(u8):
            x = torch.from_numpy(np.ascontiguousarray(u8)).float().div(255.)
            return x.permute(3, 0, 1, 2)[None].to(dev)
        x_gt = t(gt)
        variants = {
            "noise": (x_gt + torch.randn(x_gt.shape, generator=g).to(dev) / 255.).clamp(0, 1),
            "gamma": x_gt.clamp(0, 1) ** 1.1,
            "shift": torch.roll(x_gt, shifts=1, dims=-1),
            "lq":    t(lq),
        }
        dg = deltas(tok(x_gt), grid)
        gm = dg.abs().mean().clamp_min(1e-6)
        line = []
        for c in conds:
            dv = deltas(tok(variants[c]), grid)
            acc[c]["td"].append(float((dv - dg).abs().mean() / gm))
            acc[c]["mag"].append(float(dv.abs().mean() / gm))
            acc[c]["cos"].append(float(torch.nn.functional.cosine_similarity(
                dv.flatten(0, 2), dg.flatten(0, 2), dim=-1).mean()))
            line.append("%s %.3f/%+.3f" % (c, acc[c]["td"][-1], acc[c]["cos"][-1]))
        print("  %2d %-20s %s" % (i + 1, name[:20], "  ".join(line)), flush=True)

    print("\nreference ceilings on the delta field, SDSD test, %d clips" % a.clips)
    print("%-8s %7s %7s %7s" % ("vs gt", "td", "mag", "cos"))
    out = {}
    for c in conds:
        m = {k: float(np.mean(v)) for k, v in acc[c].items()}
        out[c] = m
        print("%-8s %7.3f %7.3f %+7.3f" % (c, m["td"], m["mag"], m["cos"]))
    print("%-8s %7.3f %7.3f %+7.3f    <- collapse to constant latents" %
          ("zero", 1.0, 0.0, 0.0))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
