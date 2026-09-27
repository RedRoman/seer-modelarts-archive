"""Round 2 scan: try alternate sampling protocols to give V-JEPA the best
shot at winning temporal-coherence, and report both raw cosine and
mean-centered (Pearson) cosine of per-patch norm maps.

Protocols tried:
  A: "wide" — 8 evenly spaced frames across full seq (same as round 1).
  B: "consec" — 8 consecutive frames from the middle of the seq
     (adjacent motion is single-frame small, tubelet advantage max).
  C: "burst" — 8 evenly spaced frames from first 1/3 of seq only
     (medium motion between samples, whole clip fits in ~1 second).

For each, compute per-encoder:
  - adj_cos_raw  = raw cosine of adjacent per-patch norm vectors
  - adj_cos_cent = Pearson (mean-centered) cosine of same
  - long_cos_raw / long_cos_cent

Skip S37_Mario3. Only re-check the top-3 closest-margin seqs from
round 1 (S33_mix5, MVI_1007, MVI_1011) plus the SDSD car ones with
motion — since round 1 already sampled the wide protocol for all.
"""
from __future__ import annotations
import os, sys, json, tarfile, time
from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/cache/zihaohe/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/zihaohe/hf_cache")

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/vjepa_llve")
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa

DEVICE     = torch.device("cuda")
BVI_ROOT   = "/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos"
SDSD_ROOT  = "/cache/zihaohe/datasets/SDSD_outdoor/outdoor_static_np/input"
VJEPA_REPO = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT = "/cache/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"
SIGLIP2_ID = "google/siglip2-large-patch16-256"
DINO3_ID   = "vit_large_patch16_dinov3.lvd1689m"
CACHE_DIR  = "/cache/zihaohe/hf_cache"

CROP = 512; N_SAMPLE = 8; CLIP_LEN = 16; TUBELET = 2

SIGLIP_MEAN   = torch.tensor([0.5, 0.5, 0.5])
SIGLIP_STD    = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def center_crop(arr, size):
    H, W = arr.shape[:2]
    if H < size or W < size:
        pad_h = max(0, size - H); pad_w = max(0, size - W)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        H, W = arr.shape[:2]
    y0 = (H - size) // 2; x0 = (W - size) // 2
    return arr[y0:y0+size, x0:x0+size]


def _load_bvi(p):
    im = imageio.imread(p)
    if im.ndim == 2: im = np.stack([im, im, im], axis=-1)
    if im.shape[-1] == 4: im = im[..., :3]
    return im


def _load_sdsd(p):
    a = np.load(p)
    if a.dtype != np.uint8: a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


def list_bvi(scene, fps):
    d = os.path.join(BVI_ROOT, scene, f"low_light_{fps}")
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith((".png", ".jpg")))


def list_sdsd(seq):
    d = os.path.join(SDSD_ROOT, seq)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npy"))


def sample_indices(T, protocol):
    if protocol == "wide":
        return np.linspace(0, T - 1, N_SAMPLE, dtype=int).tolist()
    if protocol == "consec":
        # 8 consecutive from center
        start = max(0, (T - N_SAMPLE) // 2)
        return list(range(start, start + N_SAMPLE))
    if protocol == "burst":
        end = max(N_SAMPLE, T // 3)
        return np.linspace(0, end - 1, N_SAMPLE, dtype=int).tolist()
    raise ValueError(protocol)


def build_clip(idx, T):
    """Build 16-frame clip (pairs (i, i+delta)) for V-JEPA."""
    if len(idx) < 2: delta = 1
    else: delta = max(1, (idx[-1] - idx[0]) // (2 * (N_SAMPLE - 1)) )
    delta = max(1, delta)
    pair = []
    for i in idx:
        j = i + delta if i + delta < T else max(0, i - delta)
        pair.extend([i, j])
    return pair, delta


def load_frames(files, idx, loader):
    thumbs = np.stack([center_crop(loader(files[i]), CROP) for i in idx], axis=0)
    return thumbs


def build_vjepa():
    return VJepaBackboneLocal21(
        local_repo=VJEPA_REPO, local_ckpt=VJEPA_CKPT,
        feature_layers=(23,), image_size=CROP, frames_per_clip=CLIP_LEN,
        patch_size=16, tubelet_size=TUBELET, freeze=True,
    ).to(DEVICE).eval()


@torch.no_grad()
def vjepa_norm(m, clip):
    x = torch.from_numpy(clip).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    out = m(x); feats = out["feats"][0]
    Tp, Hp, Wp = out["token_grid"]; D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D).squeeze(0)
    return torch.linalg.norm(f, dim=-1).detach().float().cpu().numpy()


def _rn(x, size, mean, std):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = mean.to(x.device).view(1, 3, 1, 1); s = std.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def siglip_norm(m, th):
    x = torch.from_numpy(th).to(DEVICE).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    x = _rn(x, 256, SIGLIP_MEAN, SIGLIP_STD)
    tk = m(pixel_values=x).last_hidden_state
    B, N, D = tk.shape; g = int(round(N**0.5))
    return torch.linalg.norm(tk.view(B, g, g, D), dim=-1).detach().float().cpu().numpy()


@torch.no_grad()
def dino3_norm(m, th, npref):
    x = torch.from_numpy(th).to(DEVICE).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    x = _rn(x, 256, IMAGENET_MEAN, IMAGENET_STD)
    tk = m.forward_features(x)[:, npref:, :]
    B, N, D = tk.shape; g = int(round(N**0.5))
    return torch.linalg.norm(tk.view(B, g, g, D), dim=-1).detach().float().cpu().numpy()


def _raw_cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))


def _cent_cos(a, b):
    a = a.astype(np.float64) - a.mean(); b = b.astype(np.float64) - b.mean()
    return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))


def stats(nrm):
    N = nrm.shape[0]
    v = nrm.reshape(N, -1)
    adj_r = [_raw_cos(v[i], v[i+1]) for i in range(N-1)]
    adj_c = [_cent_cos(v[i], v[i+1]) for i in range(N-1)]
    return dict(
        adj_cos       = float(np.mean(adj_r)),
        adj_cos_cent  = float(np.mean(adj_c)),
        long_cos      = _raw_cos(v[0], v[-1]),
        long_cos_cent = _cent_cos(v[0], v[-1]),
        mean_pair_cos = float(np.mean([_raw_cos(v[i], v[j])
                                       for i, j in combinations(range(N), 2)])),
    )


def main():
    t_all = time.time()
    candidates = [
        ("SDSD_outdoor", "MVI_1007",         list_sdsd, "MVI_1007"),
        ("SDSD_outdoor", "MVI_1007_2",       list_sdsd, "MVI_1007_2"),
        ("SDSD_outdoor", "MVI_1009",         list_sdsd, "MVI_1009"),
        ("SDSD_outdoor", "MVI_1011",         list_sdsd, "MVI_1011"),
        ("SDSD_outdoor", "MVI_1011_2",       list_sdsd, "MVI_1011_2"),
        ("BVI",          "S33_mix5__10fps",  list_bvi,  ("S33_mix5",  "10")),
        ("BVI",          "S35_Bunny__10fps", list_bvi,  ("S35_Bunny", "10")),
        ("BVI",          "S39_Books4__10fps",list_bvi,  ("S39_Books4","10")),
        ("BVI",          "S40_Books5__10fps",list_bvi,  ("S40_Books5","10")),
        ("BVI",          "S40_Books5__20fps",list_bvi,  ("S40_Books5","20")),
    ]
    protocols = ["wide", "consec", "burst"]

    # prep frames for every (candidate, protocol)
    prepared = []
    for dset, name, lister, arg in candidates:
        files = lister(*arg) if isinstance(arg, tuple) else lister(arg)
        T = len(files)
        if T < N_SAMPLE:
            print(f"[SKIP] {name}: T={T}<8"); continue
        loader = _load_sdsd if dset == "SDSD_outdoor" else _load_bvi
        for proto in protocols:
            idx = sample_indices(T, proto)
            pair, delta = build_clip(idx, T)
            thumbs = load_frames(files, idx, loader)
            clip = load_frames(files, pair, loader)
            prepared.append(dict(dataset=dset, name=name, protocol=proto,
                                 T=T, idx=idx, delta=delta,
                                 thumbs=thumbs, clip=clip))
            print(f"[LOAD] {name:20s} {proto:6s}  T={T} idx={idx}  delta={delta}",
                  flush=True)

    # V-JEPA
    print("\n[*] V-JEPA 2 ViT-L ...", flush=True)
    t0 = time.time(); vj = build_vjepa()
    print(f"    +{time.time()-t0:.1f}s", flush=True)
    for p in prepared:
        p["v_norm"] = vjepa_norm(vj, p["clip"])
        p["v_stats"] = stats(p["v_norm"])
    del vj; torch.cuda.empty_cache()

    # DINOv3
    print("[*] DINOv3 ViT-L/16 ...", flush=True)
    import timm
    d3 = timm.create_model(DINO3_ID, pretrained=True, num_classes=0).to(DEVICE).eval()
    npref = int(getattr(d3, "num_prefix_tokens", 5))
    for p in prepared:
        p["d_norm"] = dino3_norm(d3, p["thumbs"], npref)
        p["d_stats"] = stats(p["d_norm"])
    del d3; torch.cuda.empty_cache()

    # SigLIP
    print("[*] SigLIP 2 ...", flush=True)
    from transformers import SiglipVisionModel
    sg = SiglipVisionModel.from_pretrained(SIGLIP2_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    for p in prepared:
        p["s_norm"] = siglip_norm(sg, p["thumbs"])
        p["s_stats"] = stats(p["s_norm"])
    del sg; torch.cuda.empty_cache()

    # summarize
    rows = []
    best_raw = (-1e9, None); best_cent = (-1e9, None)
    for p in prepared:
        vs, ds, ss = p["v_stats"], p["d_stats"], p["s_stats"]
        m_raw  = vs["adj_cos"]      - ds["adj_cos"]
        m_cent = vs["adj_cos_cent"] - ds["adj_cos_cent"]
        row = dict(
            seq_name=p["name"], dataset=p["dataset"], protocol=p["protocol"],
            T=p["T"], idx=p["idx"], delta=p["delta"],
            vjepa_adj_cos=vs["adj_cos"], dino3_adj_cos=ds["adj_cos"], siglip_adj_cos=ss["adj_cos"],
            vjepa_long_cos=vs["long_cos"], dino3_long_cos=ds["long_cos"],
            vjepa_mean_pair_cos=vs["mean_pair_cos"], dino3_mean_pair_cos=ds["mean_pair_cos"],
            vjepa_adj_cos_cent=vs["adj_cos_cent"], dino3_adj_cos_cent=ds["adj_cos_cent"],
            vjepa_long_cos_cent=vs["long_cos_cent"], dino3_long_cos_cent=ds["long_cos_cent"],
            adj_margin_raw=m_raw, adj_margin_cent=m_cent,
            vjepa_wins_adj_raw=bool(m_raw > 0),
            vjepa_wins_adj_cent=bool(m_cent > 0),
            vjepa_wins_long_raw=bool(vs["long_cos"] > ds["long_cos"]),
            vjepa_wins_long_cent=bool(vs["long_cos_cent"] > ds["long_cos_cent"]),
        )
        rows.append(row)
        if m_raw  > best_raw[0]:  best_raw  = (m_raw,  p)
        if m_cent > best_cent[0]: best_cent = (m_cent, p)

    # print table sorted by raw margin descending
    rows_sorted = sorted(rows, key=lambda r: -r["adj_margin_raw"])
    print("\n" + "=" * 110)
    print(f"{'seq':22s} {'proto':6s} {'V adj':>9s} {'D adj':>9s} {'margin':>8s} "
          f"{'V adj_c':>9s} {'D adj_c':>9s} {'m_cent':>8s}  win_r/win_c")
    print("-" * 110)
    for r in rows_sorted:
        wr = "V" if r["vjepa_wins_adj_raw"]  else "D"
        wc = "V" if r["vjepa_wins_adj_cent"] else "D"
        print(f"{r['seq_name']:22s} {r['protocol']:6s} "
              f"{r['vjepa_adj_cos']:9.5f} {r['dino3_adj_cos']:9.5f} "
              f"{r['adj_margin_raw']:+8.4f} "
              f"{r['vjepa_adj_cos_cent']:9.5f} {r['dino3_adj_cos_cent']:9.5f} "
              f"{r['adj_margin_cent']:+8.4f}   {wr}/{wc}")
    print("=" * 110)
    print(f"\n[BEST RAW ]  {best_raw[1]['name']}  proto={best_raw[1]['protocol']}"
          f"  margin={best_raw[0]:+.6f}")
    print(f"[BEST CENT]  {best_cent[1]['name']}  proto={best_cent[1]['protocol']}"
          f"  margin_cent={best_cent[0]:+.6f}")

    # ---- decide which "best" to save. Prefer V-JEPA-wins RAW; else V-JEPA-wins
    # CENTERED; else smallest raw gap.
    best = best_raw[1]
    save_key = "smallest_raw_gap"
    if best_raw[0] > 0:
        save_key = "vjepa_wins_raw"
    elif best_cent[0] > 0:
        best = best_cent[1]; save_key = "vjepa_wins_cent"

    summary = dict(
        sequences=rows,
        best_seq_for_vjepa=best["name"],
        best_protocol=best["protocol"],
        best_margin_raw=float(best_raw[0]),
        best_margin_cent=float(best_cent[0]),
        chosen_by=save_key,
        note=("Raw = cosine of adjacent per-patch feature-norm vectors as "
              "specified. Centered = mean-subtracted cosine (removes uniform "
              "offset that would otherwise dominate DINOv3's near-flat norm "
              "map).  Positive margin ⇒ V-JEPA is more temporally coherent."),
    )
    with open("/tmp/temporal_scan_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print("\n[SAVE] /tmp/temporal_scan_summary.json")

    all_cos = dict(vjepa=best["v_stats"], dino3=best["d_stats"], siglip=best["s_stats"])
    np.savez(
        "/tmp/temporal_best_vjepa.npz",
        seq_name=np.array(best["name"]),
        dataset=np.array(best["dataset"]),
        protocol=np.array(best["protocol"]),
        frame_thumbs=best["thumbs"].astype(np.uint8),
        vjepa_norm_frames=best["v_norm"].astype(np.float32),
        dino3_norm_frames=best["d_norm"].astype(np.float32),
        siglip_norm_frames=best["s_norm"].astype(np.float32),
        sample_indices=np.array(best["idx"], dtype=np.int32),
        all_cos_stats_json=np.array(json.dumps(all_cos)),
        adj_margin_v_minus_d=np.float32(best_raw[0]),
        adj_margin_v_minus_d_cent=np.float32(best_cent[0]),
    )
    print("[SAVE] /tmp/temporal_best_vjepa.npz")

    with tarfile.open("/tmp/temporal_best_vjepa.tar.gz", "w:gz") as tf:
        tf.add("/tmp/temporal_best_vjepa.npz", arcname="temporal_best_vjepa.npz")
        tf.add("/tmp/temporal_scan_summary.json", arcname="temporal_scan_summary.json")
    print("[SAVE] /tmp/temporal_best_vjepa.tar.gz")
    print(f"\n[DONE] {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
