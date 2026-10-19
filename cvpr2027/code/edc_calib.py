"""Few-shot exposure calibration on top of EDC.

The probe showed the gain head is accurate in-domain (MAE 0.220 vs 0.405 baseline)
but worse than the baseline across datasets, because the scene->exposure mapping it
learns is the source domain's. Exposure is a property of how the *target* dataset
was captured, so it is not inferable from source data alone; what we can ask is how
cheaply it is correctable.

This measures exactly that: fit a single scalar bias b on k paired clips of the
target domain (b = median of true minus predicted log-gain) and apply log-gain + b
everywhere else in that domain. Reported as MAE on the held-out clips of the domain,
for k = 0 (no calibration), 1, 2, 3, 5. Leave-k-out with several random draws, so no
clip is ever scored with a bias fitted on itself.

Diagnostic on BVI/LOL *training* clips only; no test data is touched.
"""
import json, os, sys
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


def feat(lq, gt):
    with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
        tok = bb(lq)["feats"][3].float().mean(dim=1, keepdim=True)
    return lq.cpu(), tok.cpu(), target_log_gain(lq, gt).cpu()


def sdsd(key, sub, train):
    ds = ListedSDSD(sp[key]["root"], sp[key][sub], 16, 256, train)
    return [feat(ds[i]["lq"][None].to(dev), ds[i]["hq"][None].to(dev)) for i in range(len(ds))]


def video_dir(root, names, lq_t, gt_t, n, crop=512):
    out = []
    for nm in names[:n]:
        if "__" in nm:
            base, fps = nm.split("__"); fps = fps.replace("fps", "")
            lqd, gtd = os.path.join(root, base, lq_t % fps), os.path.join(root, base, gt_t % fps)
        else:
            lqd, gtd = os.path.join(root, lq_t, nm), os.path.join(root, gt_t, nm)
        lf, gf = list_images(lqd), list_images(gtd)
        k = min(len(lf), len(gf))
        if k < 16:
            continue
        st = max(0, k // 2 - 8); idx = list(range(st, min(k, st + 16)))
        lq = clip_to_tensor(np.stack([read_image_01(lf[j]) for j in idx]))[None].to(dev)[..., :crop, :crop]
        gt = clip_to_tensor(np.stack([read_image_01(gf[j]) for j in idx]))[None].to(dev)[..., :crop, :crop]
        out.append(feat(lq, gt))
    return out


print("caching source...", flush=True)
tr = sdsd("SDSD_indoor", "train", True) + sdsd("SDSD_outdoor", "train", True)
va = sdsd("SDSD_indoor", "val", False) + sdsd("SDSD_outdoor", "val", False)
print("caching targets...", flush=True)
bvi = video_dir(sp["BVI_RLV"]["root"], sp["BVI_RLV"]["train"], "low_light_%s", "normal_light_%s", 24)
print(f"train {len(tr)} val {len(va)} bvi {len(bvi)}", flush=True)

def fit(use_vjepa, seed, epochs=300):
    torch.manual_seed(seed)
    h = ExposureHead(use_vjepa=use_vjepa).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3)
    for _ in range(epochs):
        for i in np.random.permutation(len(tr)):
            lq, tok, y = tr[i]
            loss = F.l1_loss(h(lq.to(dev), tok.to(dev) if use_vjepa else None), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    h.eval()
    return h


@torch.no_grad()
def preds(h, data, use_vjepa):
    p = np.array([float(h(lq.to(dev), tok.to(dev) if use_vjepa else None)) for lq, tok, _ in data])
    y = np.array([float(t[2]) for t in data])
    return p, y


def calib_curve(p, y, ks=(0, 1, 2, 3, 5), draws=40, rng=None):
    rng = rng or np.random.default_rng(0)
    out = {}
    n = len(y)
    for k in ks:
        if k == 0:
            out[k] = float(np.abs(p - y).mean()); continue
        errs = []
        for _ in range(draws):
            idx = rng.choice(n, k, replace=False)
            b = float(np.median((y - p)[idx]))
            rest = np.setdiff1d(np.arange(n), idx)
            errs.append(np.abs(p[rest] + b - y[rest]).mean())
        out[k] = float(np.mean(errs))
    return out


for use_vjepa, name in ((True, "stats+V-JEPA"), (False, "stats only")):
    rows = []
    for seed in (0, 1):
        h = fit(use_vjepa, seed)
        pv, yv = preds(h, va, use_vjepa)
        pb_, yb = preds(h, bvi, use_vjepa)
        rows.append((np.abs(pv - yv).mean(), calib_curve(pb_, yb)))
    sd = float(np.mean([r[0] for r in rows]))
    cc = {k: float(np.mean([r[1][k] for r in rows])) for k in rows[0][1]}
    print(f"{name:14s} SDSD-val {sd:.3f} | BVI k=0 {cc[0]:.3f} k=1 {cc[1]:.3f} "
          f"k=2 {cc[2]:.3f} k=3 {cc[3]:.3f} k=5 {cc[5]:.3f}", flush=True)
