"""A fixed low-dimensional basis for the tokens, estimated once on the train split.

`token_sufficiency.py` projects onto each clip's own principal directions, so
it licenses "four clip-adaptive directions suffice" rather than "a fixed four-
dimensional code suffices". Only the second can be predicted by a small network
from the input, so only the second can be distilled, and the difference has to
be measured before E27 is defined rather than assumed away.

The basis is estimated on training-split clips only -- the test split is read
once, per ground rule 1, and estimating a basis on it would be reading it
twice. Mean and components are saved so inference is a fixed linear map.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

CODE = os.environ.get("SEER_CODE", "/cache/zihaohe/seer_cvpr/code/vjepa_llve")
for p in (CODE, os.path.join(CODE, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
from eval_e4 import held_out_sequences, REF_LAYER  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402

ROOT = os.environ.get("SEER_ROOT", "/cache/zihaohe/seer_cvpr")
TAPS = (5, 11, 17, 23)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--rank", type=int, default=64, help="components to keep")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device.startswith("npu"):
        import torch_npu  # noqa: F401
    dev = torch.device(a.device)
    amp = torch.bfloat16

    bb = VJepaBackboneLocal21(
        local_repo=f"{ROOT}/modelzoo/vjepa2-main",
        local_ckpt=f"{ROOT}/weights/vjepa2_1_vitl_dist_vitG_384.pt",
        feature_layers=TAPS, image_size=a.crop, frames_per_clip=16,
        patch_size=16, tubelet_size=2, freeze=True).to(dev).eval()

    # accumulate second moments per tapped layer rather than storing every token
    D = 1024
    n = np.zeros(len(TAPS), dtype=np.int64)
    s1 = [torch.zeros(D, dtype=torch.float64) for _ in TAPS]
    s2 = [torch.zeros(D, D, dtype=torch.float64) for _ in TAPS]

    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences("sdsd", "train", 16, a.crop)):
        if i >= a.clips:
            break
        x = torch.from_numpy(np.ascontiguousarray(lq)).float().div(255.)
        x = x.permute(3, 0, 1, 2)[None].to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp):
            feats = bb(x)["feats"]
        for k, f in enumerate(feats):
            z = f.float().reshape(-1, D).double().cpu()
            n[k] += z.shape[0]
            s1[k] += z.sum(0)
            s2[k] += z.T @ z
        if (i + 1) % 10 == 0:
            print("  %d/%d clips" % (i + 1, a.clips), flush=True)

    out = {"taps": list(TAPS), "rank": a.rank, "clips": int(min(a.clips, i + 1)),
           "split": "train", "mean": [], "components": [], "explained": []}
    for k in range(len(TAPS)):
        mu = (s1[k] / n[k])
        cov = s2[k] / n[k] - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)          # ascending
        evals = evals.flip(0).clamp_min(0)
        evecs = evecs.flip(1)
        tot = float(evals.sum())
        out["mean"].append(mu.float().tolist())
        out["components"].append(evecs[:, :a.rank].T.float().tolist())
        out["explained"].append([float(evals[:j].sum() / tot)
                                 for j in (1, 2, 4, 8, 16, 32, 64)])
        print("block %2d  variance explained by 1/2/4/8/16/32/64 comps: %s"
              % (TAPS[k], "  ".join("%.3f" % v for v in out["explained"][-1])))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
