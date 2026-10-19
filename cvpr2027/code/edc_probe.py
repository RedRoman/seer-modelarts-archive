"""Does the V-JEPA token help predict the target exposure?

Fits the ExposureHead alone (frozen backbone, no pixel branch) on the training
split and reports held-out error on the validation split and, as a transfer
check, on BVI-RLV *training* clips -- a different sensor with a different
exposure prior. Three variants: statistics only, V-JEPA only, both.
Pure diagnostic: no test data, no checkpoint selection.
"""
import json, os, sys, time
import numpy as np, torch, torch_npu  # noqa: F401
import torch.nn.functional as F

ROOT = "/temp/zihaohe/seer_cvpr"
sys.path.insert(0, f"{ROOT}/code/vjepa_llve")
sys.path.insert(0, f"{ROOT}/code/vjepa_llve/tools")
from models.exposure_head import ExposureHead, target_log_gain  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402
from seer_train_cvpr import ListedSDSD  # noqa: E402
from data.common import clip_to_tensor, list_images, read_image_01  # noqa: E402

dev = torch.device("npu:0")
sp = json.load(open(f"{ROOT}/datasets/splits_cvpr.json"))
bb = VJepaBackboneLocal21(local_repo=f"{ROOT}/modelzoo/vjepa2-main",
                          local_ckpt=f"{ROOT}/weights/vjepa2_1_vitl_dist_vitG_384.pt",
                          feature_layers=(5, 11, 17, 23), image_size=256, frames_per_clip=16,
                          patch_size=16, tubelet_size=2, freeze=True).to(dev).eval()


def cache(ds, n=None):
    """Return (stats-input clip, pooled token, target log-gain) per sequence."""
    out = []
    for i in range(len(ds) if n is None else min(n, len(ds))):
        b = ds[i]
        lq = b["lq"].unsqueeze(0).to(dev); gt = b["hq"].unsqueeze(0).to(dev)
        with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
            tok = bb(lq)["feats"][3].float().mean(dim=1)
        out.append((lq.cpu(), tok.cpu(), target_log_gain(lq, gt).cpu()))
    return out


def bvi_clips(n=16):
    root = sp["BVI_RLV"]["root"]; out = []
    for name in sp["BVI_RLV"]["train"][:n]:
        base, fps = name.split("__"); fps = fps.replace("fps", "")
        lqd = os.path.join(root, base, f"low_light_{fps}")
        gtd = os.path.join(root, base, f"normal_light_{fps}")
        lf, gf = list_images(lqd), list_images(gtd)
        k = min(len(lf), len(gf)); st = max(0, k // 2 - 8)
        idx = list(range(st, min(k, st + 16)))
        lq = clip_to_tensor(np.stack([read_image_01(lf[j]) for j in idx]))[None].to(dev)
        gt = clip_to_tensor(np.stack([read_image_01(gf[j]) for j in idx]))[None].to(dev)
        lq = lq[..., :512, :512]; gt = gt[..., :512, :512]
        with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
            tok = bb(lq)["feats"][3].float().mean(dim=1)
        out.append((lq.cpu(), tok.cpu(), target_log_gain(lq, gt).cpu()))
    return out


print("caching...", flush=True)
tr = cache(ListedSDSD(sp["SDSD_indoor"]["root"], sp["SDSD_indoor"]["train"], 16, 256, True)) + \
     cache(ListedSDSD(sp["SDSD_outdoor"]["root"], sp["SDSD_outdoor"]["train"], 16, 256, True))
va = cache(ListedSDSD(sp["SDSD_indoor"]["root"], sp["SDSD_indoor"]["val"], 16, 256, False)) + \
     cache(ListedSDSD(sp["SDSD_outdoor"]["root"], sp["SDSD_outdoor"]["val"], 16, 256, False))
bv = bvi_clips()
print(f"train {len(tr)}  val {len(va)}  bvi-train {len(bv)}", flush=True)
print(f"target log-gain: SDSD train mean {np.mean([float(t[2]) for t in tr]):.3f} "
      f"val {np.mean([float(t[2]) for t in va]):.3f}  BVI {np.mean([float(t[2]) for t in bv]):.3f}")

def run(use_stats, use_vjepa, seed=0, epochs=300):
    torch.manual_seed(seed)
    h = ExposureHead(use_vjepa=use_vjepa).to(dev)
    if not use_stats:
        with torch.no_grad():
            for p in h.stat_mlp.parameters():
                p.zero_()
        for p in h.stat_mlp.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in h.parameters() if p.requires_grad], lr=1e-3)
    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        for i in perm:
            lq, tok, y = tr[i]
            pred = h(lq.to(dev), tok.to(dev).unsqueeze(1) if use_vjepa else None)
            loss = F.l1_loss(pred, y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    h.eval()
    errs = {}
    with torch.no_grad():
        for tag, data in (("SDSD-val", va), ("BVI-train", bv)):
            e = [abs(float(h(lq.to(dev), tok.to(dev).unsqueeze(1) if use_vjepa else None)) - float(y))
                 for lq, tok, y in data]
            errs[tag] = float(np.mean(e))
    return errs

base = {t: float(np.mean([abs(float(y) - np.mean([float(x[2]) for x in tr]))
                          for _, _, y in d])) for t, d in (("SDSD-val", va), ("BVI-train", bv))}
print(f"\nbaseline (predict train mean): " + "  ".join(f"{k} {v:.3f}" for k, v in base.items()))
for name, (us, uv) in (("stats only", (True, False)), ("V-JEPA only", (False, True)),
                       ("stats+V-JEPA", (True, True))):
    r = [run(us, uv, seed=s) for s in (0, 1)]
    m = {k: np.mean([x[k] for x in r]) for k in r[0]}
    print(f"{name:14s} " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
