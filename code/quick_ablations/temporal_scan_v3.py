"""temporal_scan_v3.py

Multi-sequence temporal-coherence analysis for the Seer paper's encoder
comparison. Extends temporal_scan_v2.py in three ways:

  1. Loops over ALL six BVI-RLV L2 sequences (S33 / S35 / S37 / S39 /
     S40@10fps / S40@20fps) plus five SDSD-outdoor sequences for
     cross-dataset robustness.
  2. Adds three encoders: DINOv2, CLIP, VideoMAE-large.
  3. Adds a "DINOv3-temporal-agg" baseline (image encoder run
     independently per frame) so the video-pretraining vs
     temporal-token-fusion axis is isolated.  For the temporal Pearson
     metric this uses the same per-frame maps as regular DINOv3 (a single
     image encoder cannot produce a genuinely different adjacent-frame
     signal), so we report it once and note the equivalence in the JSON.

Metric: mean-centered Pearson correlation between per-patch
feature-norm maps of adjacent frames, averaged over the (N-1) adjacent
pairs of the 8 burst-sampled frames.

Aggregation across sequences: mean, std, 95% bootstrap CI, paired
one-sided Wilcoxon signed-rank test (H1: V-JEPA more coherent than
competitor), Cohen's d for paired samples.

Output: /home/ma-user/workspace/zihaohe/quick_ablations/temporal_scan_v3.json
"""
from __future__ import annotations
import os, sys, json, time
from typing import Callable
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
CLIP_ID    = "openai/clip-vit-large-patch14"
DINO_ID    = "facebook/dinov2-large"
DINO3_ID   = "vit_large_patch16_dinov3.lvd1689m"
SIGLIP2_ID = "google/siglip2-large-patch16-256"
VMAE_ID    = "MCG-NJU/videomae-large"
CACHE_DIR  = "/cache/zihaohe/hf_cache"

OUT_JSON   = "/home/ma-user/workspace/zihaohe/quick_ablations/temporal_scan_v3.json"

CROP       = 512
N_SAMPLE   = 8      # temporal points at which we compare adjacent frames
CLIP_LEN   = 16     # V-JEPA / VideoMAE need 16 input frames (tubelet=2)
TUBELET    = 2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN     = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD      = torch.tensor([0.26862954, 0.26130258, 0.27577711])
SIGLIP_MEAN   = torch.tensor([0.5, 0.5, 0.5])
SIGLIP_STD    = torch.tensor([0.5, 0.5, 0.5])


# ============================================================= data
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


def burst_indices(T):
    """8 evenly spaced frames from the first 1/3 of the sequence."""
    end = max(N_SAMPLE, T // 3)
    return np.linspace(0, end - 1, N_SAMPLE, dtype=int).tolist()


def build_pair_clip(idx, T):
    """Build 16-frame clip (pairs (i, i+delta)) for tubelet-2 models."""
    delta = max(1, (idx[-1] - idx[0]) // (2 * (N_SAMPLE - 1)))
    pair = []
    for i in idx:
        j = i + delta if i + delta < T else max(0, i - delta)
        pair.extend([i, j])
    return pair, delta


def load_frames(files, indices, loader):
    return np.stack([center_crop(loader(files[i]), CROP)
                     for i in indices], axis=0)


# ============================================================= encoders
def _to_bchw01(frames_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


def _resize_norm(x, size, mean, std):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = mean.to(x.device).view(1, 3, 1, 1); s = std.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


def build_vjepa():
    return VJepaBackboneLocal21(
        local_repo=VJEPA_REPO, local_ckpt=VJEPA_CKPT,
        feature_layers=(23,), image_size=CROP, frames_per_clip=CLIP_LEN,
        patch_size=16, tubelet_size=TUBELET, freeze=True,
    ).to(DEVICE).eval()


@torch.no_grad()
def vjepa_per_frame_norm(model, clip16: np.ndarray) -> np.ndarray:
    """clip16: (16, H, W, 3) uint8. Returns (8, 32, 32) per-patch norm map,
    one per tubelet output timestep."""
    x = torch.from_numpy(clip16).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()   # (1,3,T,H,W)
    out = model(x)
    feats = out["feats"][0]                                # (1, N, D)
    Tp, Hp, Wp = out["token_grid"]
    D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D).squeeze(0)            # (Tp, Hp, Wp, D)
    return torch.linalg.norm(f, dim=-1).detach().float().cpu().numpy()


@torch.no_grad()
def clip_per_frame_norm(model, frames8: np.ndarray) -> np.ndarray:
    """(8, H, W, 3) uint8 -> (8, 16, 16)."""
    x = _to_bchw01(frames8); x = _resize_norm(x, 224, CLIP_MEAN, CLIP_STD)
    tk = model(pixel_values=x).last_hidden_state[:, 1:, :]  # drop CLS
    B, N, D = tk.shape; g = int(round(N ** 0.5))
    return torch.linalg.norm(tk.view(B, g, g, D), dim=-1).detach().float().cpu().numpy()


@torch.no_grad()
def dinov2_per_frame_norm(model, frames8: np.ndarray) -> np.ndarray:
    """(8, H, W, 3) uint8 -> (8, 16, 16)."""
    x = _to_bchw01(frames8); x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    tk = model(pixel_values=x).last_hidden_state[:, 1:, :]
    B, N, D = tk.shape; g = int(round(N ** 0.5))
    return torch.linalg.norm(tk.view(B, g, g, D), dim=-1).detach().float().cpu().numpy()


@torch.no_grad()
def dinov3_per_frame_norm(model, frames8: np.ndarray, n_prefix: int) -> np.ndarray:
    x = _to_bchw01(frames8); x = _resize_norm(x, 256, IMAGENET_MEAN, IMAGENET_STD)
    tk = model.forward_features(x)[:, n_prefix:, :]
    B, N, D = tk.shape; g = int(round(N ** 0.5))
    return torch.linalg.norm(tk.view(B, g, g, D), dim=-1).detach().float().cpu().numpy()


@torch.no_grad()
def siglip_per_frame_norm(model, frames8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames8); x = _resize_norm(x, 256, SIGLIP_MEAN, SIGLIP_STD)
    tk = model(pixel_values=x).last_hidden_state
    B, N, D = tk.shape; g = int(round(N ** 0.5))
    return torch.linalg.norm(tk.view(B, g, g, D), dim=-1).detach().float().cpu().numpy()


@torch.no_grad()
def videomae_per_frame_norm(model, clip16: np.ndarray) -> np.ndarray:
    """VideoMAE-large @ 224: 16 frames -> 8 temporal x 14x14 spatial tokens.
    Returns (8, 14, 14) per-patch norm map, one per tubelet timestep."""
    x = _to_bchw01(clip16)                    # (16, 3, H, W)
    x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    # VideoMAE expects (B, T, C, H, W)
    x = x.unsqueeze(0)                        # (1, 16, 3, 224, 224)
    tk = model(pixel_values=x).last_hidden_state     # (1, 1568, 1024)
    B, N, D = tk.shape
    Tp = CLIP_LEN // TUBELET                  # 8
    N_spatial = N // Tp                       # 196
    g = int(round(N_spatial ** 0.5))          # 14
    assert g * g * Tp == N, f"unexpected VideoMAE token count {N}"
    f = tk.view(B, Tp, g, g, D).squeeze(0)    # (8, 14, 14, 1024)
    return torch.linalg.norm(f, dim=-1).detach().float().cpu().numpy()


# ============================================================= metric
def _cent_cos(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def adjacent_pearson(norm_maps: np.ndarray) -> float:
    """norm_maps: (N, h, w). Returns mean over adjacent-pair Pearson."""
    v = norm_maps.reshape(norm_maps.shape[0], -1)
    return float(np.mean([_cent_cos(v[i], v[i + 1])
                          for i in range(v.shape[0] - 1)]))


# ============================================================= stats helpers
def bootstrap_ci_mean(vals, n_boot=10000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=np.float64)
    N = len(v)
    boots = rng.choice(v, size=(n_boot, N), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return lo, hi


def bootstrap_ci_delta(a, b, n_boot=10000, ci=95, seed=0):
    """Paired delta a-b, bootstrap over paired samples."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    N = len(a)
    idx = rng.integers(0, N, size=(n_boot, N))
    deltas = (a[idx] - b[idx]).mean(axis=1)
    lo = float(np.percentile(deltas, (100 - ci) / 2))
    hi = float(np.percentile(deltas, 100 - (100 - ci) / 2))
    return lo, hi


def paired_cohens_d(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


# ============================================================= main
def main():
    from scipy.stats import wilcoxon
    t_all = time.time()

    candidates = [
        # (dataset,        seq_name,             lister,      lister-arg)
        ("BVI", "S33_mix5__10fps",   list_bvi, ("S33_mix5",  "10")),
        ("BVI", "S35_Bunny__10fps",  list_bvi, ("S35_Bunny", "10")),
        ("BVI", "S37_Mario3__10fps", list_bvi, ("S37_Mario3","10")),
        ("BVI", "S39_Books4__10fps", list_bvi, ("S39_Books4","10")),
        ("BVI", "S40_Books5__10fps", list_bvi, ("S40_Books5","10")),
        ("BVI", "S40_Books5__20fps", list_bvi, ("S40_Books5","20")),
        ("SDSD_outdoor", "MVI_1007",   list_sdsd, "MVI_1007"),
        ("SDSD_outdoor", "MVI_1007_2", list_sdsd, "MVI_1007_2"),
        ("SDSD_outdoor", "MVI_1009",   list_sdsd, "MVI_1009"),
        ("SDSD_outdoor", "MVI_1011",   list_sdsd, "MVI_1011"),
        ("SDSD_outdoor", "MVI_1011_2", list_sdsd, "MVI_1011_2"),
    ]

    # ---- prep frames for each sequence ----
    prepared = []
    for dset, name, lister, arg in candidates:
        files = lister(*arg) if isinstance(arg, tuple) else lister(arg)
        T = len(files)
        if T < N_SAMPLE:
            print(f"[SKIP] {name}: T={T}<{N_SAMPLE}"); continue
        loader = _load_sdsd if dset.startswith("SDSD") else _load_bvi
        idx = burst_indices(T)
        pair, delta = build_pair_clip(idx, T)
        thumbs = load_frames(files, idx,  loader)   # (8, 512, 512, 3)
        clip16 = load_frames(files, pair, loader)   # (16, 512, 512, 3)
        prepared.append(dict(dataset=dset, name=name, T=T, idx=idx,
                              delta=delta, thumbs=thumbs, clip16=clip16))
        print(f"[LOAD] {name:22s} T={T} idx={idx} delta={delta}", flush=True)

    n_seq = len(prepared)
    print(f"\n[*] {n_seq} sequences prepared\n", flush=True)

    # ---- V-JEPA ----
    print("[*] V-JEPA 2.1 ViT-L ...", flush=True); t0 = time.time()
    vj = build_vjepa()
    print(f"    loaded +{time.time()-t0:.1f}s", flush=True)
    for p in prepared:
        p["vjepa_norm"] = vjepa_per_frame_norm(vj, p["clip16"])
        p["vjepa_r"]    = adjacent_pearson(p["vjepa_norm"])
        print(f"    {p['name']:22s} r={p['vjepa_r']:.5f}", flush=True)
    del vj; torch.cuda.empty_cache()

    # ---- DINOv3 ----
    import timm
    print("\n[*] DINOv3 ViT-L/16 (timm) ...", flush=True)
    d3 = timm.create_model(DINO3_ID, pretrained=True, num_classes=0).to(DEVICE).eval()
    n_prefix = int(getattr(d3, "num_prefix_tokens", 5))
    for p in prepared:
        p["dinov3_norm"] = dinov3_per_frame_norm(d3, p["thumbs"], n_prefix)
        p["dinov3_r"]    = adjacent_pearson(p["dinov3_norm"])
        print(f"    {p['name']:22s} r={p['dinov3_r']:.5f}", flush=True)
    del d3; torch.cuda.empty_cache()

    # ---- DINOv2 ----
    from transformers import Dinov2Model, CLIPVisionModel, SiglipVisionModel
    print("\n[*] DINOv2 ViT-L/14 ...", flush=True)
    d2 = Dinov2Model.from_pretrained(DINO_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    for p in prepared:
        p["dinov2_norm"] = dinov2_per_frame_norm(d2, p["thumbs"])
        p["dinov2_r"]    = adjacent_pearson(p["dinov2_norm"])
        print(f"    {p['name']:22s} r={p['dinov2_r']:.5f}", flush=True)
    del d2; torch.cuda.empty_cache()

    # ---- CLIP ----
    print("\n[*] CLIP ViT-L/14 ...", flush=True)
    cl = CLIPVisionModel.from_pretrained(CLIP_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    for p in prepared:
        p["clip_norm"] = clip_per_frame_norm(cl, p["thumbs"])
        p["clip_r"]    = adjacent_pearson(p["clip_norm"])
        print(f"    {p['name']:22s} r={p['clip_r']:.5f}", flush=True)
    del cl; torch.cuda.empty_cache()

    # ---- SigLIP 2 ----
    print("\n[*] SigLIP 2 ViT-L/16 ...", flush=True)
    sg = SiglipVisionModel.from_pretrained(SIGLIP2_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    for p in prepared:
        p["siglip_norm"] = siglip_per_frame_norm(sg, p["thumbs"])
        p["siglip_r"]    = adjacent_pearson(p["siglip_norm"])
        print(f"    {p['name']:22s} r={p['siglip_r']:.5f}", flush=True)
    del sg; torch.cuda.empty_cache()

    # ---- VideoMAE-large ----
    videomae_failed = None
    try:
        from transformers import VideoMAEModel
        print("\n[*] VideoMAE-large ...", flush=True)
        vm = VideoMAEModel.from_pretrained(VMAE_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
        for p in prepared:
            p["videomae_norm"] = videomae_per_frame_norm(vm, p["clip16"])
            p["videomae_r"]    = adjacent_pearson(p["videomae_norm"])
            print(f"    {p['name']:22s} r={p['videomae_r']:.5f}", flush=True)
        del vm; torch.cuda.empty_cache()
    except Exception as e:
        videomae_failed = f"{type(e).__name__}: {e}"
        print(f"[WARN] VideoMAE failed: {videomae_failed}", flush=True)
        for p in prepared:
            p["videomae_r"] = None

    # ---- DINOv3-temporal-agg (per-frame image encoder as "video baseline") ----
    # For the adjacent-frame Pearson metric, an image encoder applied
    # independently per frame IS the natural signal - there is no temporal
    # fusion in DINOv3 either way. So the temporal-agg value here equals
    # regular DINOv3. We mirror it in the JSON for consistency with the
    # invariance report where temporal-agg genuinely differs.
    for p in prepared:
        p["dinov3_tagg_r"] = p["dinov3_r"]

    # ============================================================= summary
    encoders = [
        ("vjepa",       "V-JEPA-2.1"),
        ("dinov3",      "DINOv3"),
        ("dinov2",      "DINOv2"),
        ("siglip",      "SigLIP2"),
        ("clip",        "CLIP"),
        ("videomae",    "VideoMAE-L"),
        ("dinov3_tagg", "DINOv3-temporal-agg"),
    ]

    per_seq_out = {}
    for p in prepared:
        per_seq_out[p["name"]] = {
            "dataset": p["dataset"], "T": int(p["T"]),
            "idx": p["idx"], "delta": int(p["delta"]),
        }
        for k, _ in encoders:
            per_seq_out[p["name"]][k] = p.get(f"{k}_r", None)

    # V-JEPA vector for pairwise tests
    v_vec = np.array([p["vjepa_r"] for p in prepared], dtype=np.float64)

    summary = {}
    for k, label in encoders:
        vals = [p.get(f"{k}_r", None) for p in prepared]
        if any(v is None for v in vals):
            summary[k] = {"label": label, "status": "missing",
                          "reason": videomae_failed if k == "videomae" else "N/A"}
            continue
        vals = np.array(vals, dtype=np.float64)
        m = float(vals.mean()); s = float(vals.std(ddof=1))
        lo, hi = bootstrap_ci_mean(vals)
        entry = {
            "label": label,
            "mean":  m,
            "std":   s,
            "ci95_lo": lo, "ci95_hi": hi,
            "n": int(len(vals)),
            "per_seq_values": vals.tolist(),
        }
        if k != "vjepa":
            # paired one-sided Wilcoxon: V-JEPA > competitor
            try:
                wres = wilcoxon(v_vec, vals, alternative="greater",
                                zero_method="pratt")
                entry["wilcoxon_p_vs_vjepa"] = float(wres.pvalue)
                entry["wilcoxon_stat"]       = float(wres.statistic)
            except ValueError as e:
                entry["wilcoxon_p_vs_vjepa"] = None
                entry["wilcoxon_error"]      = str(e)
            entry["cohens_d_paired_vs_vjepa"] = paired_cohens_d(v_vec, vals)
            entry["delta_mean_vjepa_minus"]   = float((v_vec - vals).mean())
            dlo, dhi = bootstrap_ci_delta(v_vec.tolist(), vals.tolist())
            entry["delta_ci95_lo"] = dlo
            entry["delta_ci95_hi"] = dhi
        summary[k] = entry

    out = {
        "meta": {
            "protocol": "burst_8_frames_first_third",
            "metric": "adjacent-frame Pearson r on per-patch feature-norm maps "
                      "(mean-centered cosine)",
            "n_seqs": n_seq,
            "sequences": [p["name"] for p in prepared],
            "encoders": [k for k, _ in encoders],
            "vjepa_ckpt": VJEPA_CKPT,
            "crop": CROP, "n_sample": N_SAMPLE, "clip_len": CLIP_LEN,
            "note": ("DINOv3-temporal-agg equals DINOv3 for this metric because "
                     "a per-frame image encoder yields the same adjacent-frame "
                     "signal either way; the two differ only for the "
                     "single-frame invariance analysis (invariance_v2.json)."),
        },
        "per_seq": per_seq_out,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=2)

    # ---- print a compact table ----
    print("\n" + "=" * 90)
    print(f"{'encoder':<22s} {'mean':>8s} {'95% CI':>18s} {'delta_v':>8s} {'p_wilc':>10s} {'cohen d':>8s}")
    print("-" * 90)
    for k, label in encoders:
        e = summary[k]
        if e.get("status") == "missing":
            print(f"{label:<22s} MISSING ({e.get('reason','')})")
            continue
        ci = f"[{e['ci95_lo']:.4f},{e['ci95_hi']:.4f}]"
        if k == "vjepa":
            print(f"{label:<22s} {e['mean']:>8.5f} {ci:>18s}   ---")
        else:
            print(f"{label:<22s} {e['mean']:>8.5f} {ci:>18s}"
                  f" {e['delta_mean_vjepa_minus']:>+8.5f}"
                  f" {e.get('wilcoxon_p_vs_vjepa', float('nan')):>10.4g}"
                  f" {e['cohens_d_paired_vs_vjepa']:>+8.4f}")
    print("=" * 90)
    print(f"[SAVE] {OUT_JSON}")
    print(f"[DONE] {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
