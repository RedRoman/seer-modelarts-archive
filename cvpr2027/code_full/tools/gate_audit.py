"""Did E21's trust gate actually train, and what did it learn to do?

The lesson that keeps recurring in this project is that "installed" is not
"fired": `strict=False` loads have silently dropped modules before, and a gate
initialised to pass everything through is indistinguishable from no gate at all
unless its parameters are checked against their initial values.

Three things, none of which need training anything:

  1. the gate parameters exist in the checkpoint and have moved off their init
     (last conv weight zeros, bias 4.0 -- chosen so the module starts
     elementwise-identical to the ungated one)
  2. what the gate opens to on held-out data, per scale
  3. whether it closes when the tokens are made useless, which is H16's
     criterion (ii) and the half that does not depend on the gate helping
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

CODE = "/cache/zihaohe/seer_cvpr/code/vjepa_llve"
for p in (CODE, os.path.join(CODE, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
from eval_e4 import build_model, held_out_sequences, run_clip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device(a.device)

    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = st["pixel_branch"]
    keys = [k for k in sd if ".gate." in k]
    if not keys:
        raise SystemExit("no gate parameters in this checkpoint: the run did "
                         "not train the gated module")
    print("gate parameters found: %d tensors across %d cross-attention sites"
          % (len(keys), len({k.split('.gate.')[0] for k in keys})))
    print("\n%-46s %10s %10s" % ("tensor", "max |x|", "moved off init"))
    moved = []
    for k in sorted(keys):
        v = sd[k].float()
        if k.endswith("2.weight"):        # init zeros
            off = bool(v.abs().max() > 1e-8); ref = 0.0
        elif k.endswith("2.bias"):        # init 4.0
            off = bool((v - 4.0).abs().max() > 1e-6); ref = 4.0
        else:
            off = True; ref = float("nan")
        moved.append(off)
        print("%-46s %10.4f %10s" % (k[-46:], float(v.abs().max()),
                                     "yes" if off else "NO"))
    if not all(moved):
        print("\nsome gate tensors are still at their initial values")

    # 2 and 3: what the gate opens to, with real and with useless tokens
    report = {}
    for force in (None, "random"):
        pb, bb, refiner, exp_head, arm, clip_len, _ = build_model(
            a.ckpt, dev, force_arm=force)
        vals = []
        hooks = []

        def hook(mod, inp, out):
            x, kv = inp[0], inp[1]
            with torch.no_grad():
                import torch.nn.functional as F
                from einops import rearrange
                xn = mod.norm_q(x)
                Q = rearrange(mod.q(xn), "b (h d) y w -> b h (y w) d", h=mod.h)
                k2 = mod.kv(kv).reshape(x.shape[0], kv.shape[1], 2, mod.h, mod.d)
                K, V = k2.permute(2, 0, 3, 1, 4).unbind(0)
                o = F.scaled_dot_product_attention(Q, K, V)
                o = rearrange(o, "b h (y w) d -> b (h d) y w",
                              y=x.shape[2], w=x.shape[3])
                agree = F.cosine_similarity(o.float(), xn.float(), dim=1, eps=1e-6)
                mag = o.float().pow(2).mean(1).clamp_min(1e-12).sqrt() / mod.scale_ema.clamp_min(1e-6)
                g = torch.sigmoid(mod.gate(torch.stack([agree, mag], 1)))
                vals.append(float(g.mean()))

        for m in pb.modules():
            if type(m).__name__ == "VJepaCrossAttnGated":
                hooks.append(m.register_forward_hook(hook))
        for i, (name, lq, gt, kind) in enumerate(
                held_out_sequences("sdsd", "test", clip_len, a.crop)):
            if i >= a.clips:
                break
            run_clip(pb, bb, refiner, exp_head, arm, lq, clip_len, dev,
                     torch.bfloat16)
        for h in hooks:
            h.remove()
        label = "real tokens" if force is None else "random tokens"
        report[label] = float(np.mean(vals)) if vals else float("nan")
        print("\nmean gate value, %-14s %.4f  (1.0 = fully open, init 0.982)"
              % (label, report[label]))
        del pb, bb
        torch.cuda.empty_cache()

    print("\nH16 criterion (ii) expects the second to be clearly below the first.")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(ckpt=a.ckpt, gate_mean=report,
                       all_params_moved=bool(all(moved))),
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
