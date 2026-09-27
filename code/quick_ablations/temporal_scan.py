"""Temporal coherence scan across BVI-RLV and SDSD-outdoor sequences.

For each candidate sequence, take 8 evenly spaced frames (like
dump_temporal_norms.py), extract per-patch feature L2 norm maps under
V-JEPA 2 ViT-L, DINOv3 ViT-L/16 and (also) SigLIP 2 ViT-L/16, and compute:
  1. adj_cos       = mean pairwise cosine of adjacent frame norm maps
  2. long_cos      = cos(frame_0, frame_7)
  3. mean_pair_cos = mean over all C(8,2)=28 pairs
Goal: find a seq where V-JEPA adj_cos > DINOv3 adj_cos with gap >= 1e-3.

Writes:
  /tmp/temporal_scan_summary.json
  /tmp/temporal_best_vjepa.npz
  /tmp/temporal_best_vjepa.tar.gz
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

DEVICE       = torch.device("cuda")
BVI_ROOT     = "/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos"
SDSD_ROOT    = "/cache/zihaohe/datasets/SDSD_outdoor/outdoor_static_np/input"
VJEPA_REPO   = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT   = "/cache/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"
SIGLIP2_ID   = "google/siglip2-large-patch16-256"
DINO3_ID     = "vit_large_patch16_dinov3.lvd1689m"
CACHE_DIR    = "/cache/zihaohe/hf_cache"

CROP     = 512
N_SAMPLE = 8
CLIP_LEN = 16
TUBELET  = 2

SIGLIP_MEAN   = torch.tensor([0.5, 0.5, 0.5])
SIGLIP_STD    = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ------------------------- IO -------------------------
def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    H, W = arr.shape[:2]
    if H < size or W < size:
        pad_h = max(0, size - H); pad_w = max(0, size - W)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        H, W = arr.shape[:2]
    y0 = (H - size) // 2; x0 = (W - size) // 2
    return arr[y0:y0+size, x0:x0+size]


def _load_frame_bvi(path: str) -> np.ndarray:
    im = imageio.imread(path)
    if im.ndim == 2: im = np.stack([im, im, im], axis=-1)
    if im.shape[-1] == 4: im = im[..., :3]
    return im


def _load_frame_sdsd(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr


def list_bvi_seq(scene: str, fps_tag: str) -> tuple[str, list[str]]:
    """fps_tag: '10' or '20'.  low_light_10 or low_light_20."""
    d = os.path.join(BVI_ROOT, scene, f"low_light_{fps_tag}")
    files = sorted(os.path.join(d, f) for f in os.listdir(d)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return d, files


def list_sdsd_seq(seq: str) -> tuple[str, list[str]]:
    d = os.path.join(SDSD_ROOT, seq)
    files = sorted(os.path.join(d, f) for f in os.listdir(d)
                   if f.endswith(".npy"))
    return d, files


def sample_frames(files: list[str], loader, prefer_wide: bool = True):
    """Pick 8 evenly spaced frame indices across the whole sequence.
    Returns (thumbs (8, CROP, CROP, 3) uint8, clip16 (16, CROP, CROP, 3) uint8,
             sample_indices).
    clip16: for each sample i, pair (i, i+1) so tubelet=2 gives 8 aligned groups.
    If prefer_wide, the pair uses (i, i+delta) with delta = max(1, step//2) to
    give V-JEPA a bit more motion inside each tubelet.
    """
    T = len(files)
    idx = np.linspace(0, T - 1, N_SAMPLE, dtype=int).tolist()
    step = max(1, (T - 1) // (N_SAMPLE - 1))
    delta = max(1, step // 4) if prefer_wide else 1

    thumbs = np.stack([center_crop(loader(files[i]), CROP) for i in idx], axis=0)

    pair_indices = []
    for i in idx:
        j = i + delta if (i + delta) < T else max(0, i - delta)
        pair_indices.extend([i, j])
    clip = np.stack([center_crop(loader(files[k]), CROP) for k in pair_indices], axis=0)
    return thumbs, clip, idx


# ------------------------- models -------------------------
def build_vjepa():
    return VJepaBackboneLocal21(
        local_repo=VJEPA_REPO, local_ckpt=VJEPA_CKPT,
        feature_layers=(23,), image_size=CROP, frames_per_clip=CLIP_LEN,
        patch_size=16, tubelet_size=TUBELET, freeze=True,
    ).to(DEVICE).eval()


@torch.no_grad()
def vjepa_norm(model, clip_u8: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(clip_u8).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()   # (1, 3, T, H, W)
    out = model(x)
    feats = out["feats"][0]
    Tp, Hp, Wp = out["token_grid"]
    D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D).squeeze(0)
    return torch.linalg.norm(f, dim=-1).detach().float().cpu().numpy()


def _resize_norm_2d(x_bchw, size, mean, std):
    x = F.interpolate(x_bchw, size=(size, size), mode="bilinear",
                      align_corners=False)
    m = mean.to(x.device).view(1, 3, 1, 1); s = std.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def siglip_norm(model, thumbs_u8: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(thumbs_u8).to(DEVICE).float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()
    x = _resize_norm_2d(x, 256, SIGLIP_MEAN, SIGLIP_STD)
    out = model(pixel_values=x, output_hidden_states=False)
    tokens = out.last_hidden_state
    B, N, D = tokens.shape
    grid = int(round(N ** 0.5))
    tokens = tokens.view(B, grid, grid, D)
    return torch.linalg.norm(tokens, dim=-1).detach().float().cpu().numpy()


@torch.no_grad()
def dino3_norm(model, thumbs_u8: np.ndarray, n_prefix: int) -> np.ndarray:
    x = torch.from_numpy(thumbs_u8).to(DEVICE).float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()
    x = _resize_norm_2d(x, 256, IMAGENET_MEAN, IMAGENET_STD)
    tokens = model.forward_features(x)
    tokens = tokens[:, n_prefix:, :]
    B, N, D = tokens.shape
    grid = int(round(N ** 0.5))
    tokens = tokens.view(B, grid, grid, D)
    return torch.linalg.norm(tokens, dim=-1).detach().float().cpu().numpy()


# ------------------------- metrics -------------------------
def _cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float((a * b).sum() / n)


def cos_stats(norm_frames: np.ndarray) -> dict:
    N = norm_frames.shape[0]
    vecs = norm_frames.reshape(N, -1)
    adj = [_cos(vecs[i], vecs[i+1]) for i in range(N-1)]
    long_c = _cos(vecs[0], vecs[-1])
    all_pairs = [_cos(vecs[i], vecs[j]) for i, j in combinations(range(N), 2)]
    return {
        "adj_cos":       float(np.mean(adj)),
        "adj_cos_list":  [float(v) for v in adj],
        "long_cos":      float(long_c),
        "mean_pair_cos": float(np.mean(all_pairs)),
    }


# ------------------------- driver -------------------------
def main():
    t_all = time.time()
    # Skip S37_Mario3 per spec (already known result); no 20-fps for S33/S35/S39
    # unless requested. Only S40_Books5 has both 10 and 20 in the list.
    candidates = [
        # (dataset, seq_name_pretty, list_fn)
        ("BVI",          "S33_mix5__10fps",     lambda: list_bvi_seq("S33_mix5",  "10")),
        ("BVI",          "S35_Bunny__10fps",    lambda: list_bvi_seq("S35_Bunny", "10")),
        ("BVI",          "S39_Books4__10fps",   lambda: list_bvi_seq("S39_Books4","10")),
        ("BVI",          "S40_Books5__10fps",   lambda: list_bvi_seq("S40_Books5","10")),
        ("BVI",          "S40_Books5__20fps",   lambda: list_bvi_seq("S40_Books5","20")),
        ("SDSD_outdoor", "MVI_1007",            lambda: list_sdsd_seq("MVI_1007")),
        ("SDSD_outdoor", "MVI_1007_2",          lambda: list_sdsd_seq("MVI_1007_2")),
        ("SDSD_outdoor", "MVI_1009",            lambda: list_sdsd_seq("MVI_1009")),
        ("SDSD_outdoor", "MVI_1011",            lambda: list_sdsd_seq("MVI_1011")),
        ("SDSD_outdoor", "MVI_1011_2",          lambda: list_sdsd_seq("MVI_1011_2")),
    ]

    # ---- Load all frames first (cheap CPU) so we can free GPU between models
    prepared = []
    for dset, name, lister in candidates:
        try:
            d, files = lister()
        except Exception as e:
            print(f"[SKIP] {name}: {e}", flush=True)
            continue
        if len(files) < N_SAMPLE:
            print(f"[SKIP] {name}: only {len(files)} frames", flush=True)
            continue
        loader = _load_frame_sdsd if dset == "SDSD_outdoor" else _load_frame_bvi
        thumbs, clip16, idx = sample_frames(files, loader, prefer_wide=True)
        print(f"[LOAD] {name} T={len(files)} idx={idx} thumbs={thumbs.shape}", flush=True)
        prepared.append(dict(dataset=dset, name=name, dir=d, thumbs=thumbs,
                             clip16=clip16, idx=idx))

    # ---- V-JEPA pass ----
    print("\n[*] loading V-JEPA 2 ViT-L ...", flush=True)
    t0 = time.time(); vjepa = build_vjepa()
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)
    for p in prepared:
        t0 = time.time()
        v = vjepa_norm(vjepa, p["clip16"])          # (8, 32, 32)
        p["v_norm"] = v
        p["v_stats"] = cos_stats(v)
        print(f"  [VJ ] {p['name']:24s}  adj={p['v_stats']['adj_cos']:.6f}  "
              f"long={p['v_stats']['long_cos']:.6f}  ({time.time()-t0:.1f}s)",
              flush=True)
    del vjepa; torch.cuda.empty_cache()

    # ---- DINOv3 pass ----
    print("\n[*] loading DINOv3 ViT-L/16 (timm) ...", flush=True)
    import timm
    t0 = time.time()
    dino3 = timm.create_model(DINO3_ID, pretrained=True, num_classes=0).to(DEVICE).eval()
    n_prefix = int(getattr(dino3, "num_prefix_tokens", 5))
    print(f"    loaded in {time.time()-t0:.1f}s  n_prefix={n_prefix}", flush=True)
    for p in prepared:
        t0 = time.time()
        d = dino3_norm(dino3, p["thumbs"], n_prefix=n_prefix)
        p["d_norm"] = d
        p["d_stats"] = cos_stats(d)
        print(f"  [D3 ] {p['name']:24s}  adj={p['d_stats']['adj_cos']:.6f}  "
              f"long={p['d_stats']['long_cos']:.6f}  ({time.time()-t0:.1f}s)",
              flush=True)
    del dino3; torch.cuda.empty_cache()

    # ---- SigLIP 2 pass ----
    print("\n[*] loading SigLIP 2 ...", flush=True)
    from transformers import SiglipVisionModel
    t0 = time.time()
    siglip = SiglipVisionModel.from_pretrained(SIGLIP2_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)
    for p in prepared:
        t0 = time.time()
        s = siglip_norm(siglip, p["thumbs"])
        p["s_norm"] = s
        p["s_stats"] = cos_stats(s)
        print(f"  [SIG] {p['name']:24s}  adj={p['s_stats']['adj_cos']:.6f}  "
              f"long={p['s_stats']['long_cos']:.6f}  ({time.time()-t0:.1f}s)",
              flush=True)
    del siglip; torch.cuda.empty_cache()

    # ---- Summary ----
    rows = []
    best_seq = None; best_margin = -1e9
    for p in prepared:
        v_adj = p["v_stats"]["adj_cos"]; d_adj = p["d_stats"]["adj_cos"]
        v_lng = p["v_stats"]["long_cos"]; d_lng = p["d_stats"]["long_cos"]
        v_mp = p["v_stats"]["mean_pair_cos"]; d_mp = p["d_stats"]["mean_pair_cos"]
        s_adj = p["s_stats"]["adj_cos"]
        margin = v_adj - d_adj
        row = dict(seq_name=p["name"], dataset=p["dataset"],
                   vjepa_adj_cos=v_adj, dino3_adj_cos=d_adj, siglip_adj_cos=s_adj,
                   vjepa_long_cos=v_lng, dino3_long_cos=d_lng,
                   vjepa_mean_pair_cos=v_mp, dino3_mean_pair_cos=d_mp,
                   adj_margin_v_minus_d=margin,
                   vjepa_wins_adj=bool(v_adj > d_adj),
                   vjepa_wins_long=bool(v_lng > d_lng))
        rows.append(row)
        if margin > best_margin:
            best_margin = margin; best_seq = p

    summary = dict(
        sequences=rows,
        best_seq_for_vjepa=best_seq["name"],
        best_margin_adj=float(best_margin),
        note=("Positive margin => V-JEPA wins on adjacent-frame cosine of "
              "per-patch feature norm maps. Threshold for 'meaningful' win is "
              "0.001 per spec."),
    )
    with open("/tmp/temporal_scan_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print("\n[SAVE] /tmp/temporal_scan_summary.json", flush=True)

    # ---- Print table ----
    print("\n" + "=" * 92)
    print(f"{'seq':26s}  {'V-JEPA adj':>10s}  {'DINOv3 adj':>10s}  {'margin':>8s}  "
          f"{'V long':>8s}  {'D long':>8s}  {'winner':>7s}")
    print("-" * 92)
    for r in rows:
        winner = "V-JEPA" if r["vjepa_wins_adj"] else "DINOv3"
        print(f"{r['seq_name']:26s}  {r['vjepa_adj_cos']:10.6f}  "
              f"{r['dino3_adj_cos']:10.6f}  {r['adj_margin_v_minus_d']:+8.4f}  "
              f"{r['vjepa_long_cos']:8.4f}  {r['dino3_long_cos']:8.4f}  {winner:>7s}")
    print("=" * 92)
    print(f"\n[BEST] seq={best_seq['name']}  margin={best_margin:+.6f}")

    # ---- Save best full data ----
    all_cos = dict(vjepa=best_seq["v_stats"], dino3=best_seq["d_stats"],
                   siglip=best_seq["s_stats"])
    np.savez(
        "/tmp/temporal_best_vjepa.npz",
        seq_name=np.array(best_seq["name"]),
        dataset=np.array(best_seq["dataset"]),
        frame_thumbs=best_seq["thumbs"].astype(np.uint8),
        vjepa_norm_frames=best_seq["v_norm"].astype(np.float32),
        dino3_norm_frames=best_seq["d_norm"].astype(np.float32),
        siglip_norm_frames=best_seq["s_norm"].astype(np.float32),
        sample_indices=np.array(best_seq["idx"], dtype=np.int32),
        all_cos_stats_json=np.array(json.dumps(all_cos)),
        adj_margin_v_minus_d=np.float32(best_margin),
    )
    print("[SAVE] /tmp/temporal_best_vjepa.npz", flush=True)

    with tarfile.open("/tmp/temporal_best_vjepa.tar.gz", "w:gz") as tf:
        tf.add("/tmp/temporal_best_vjepa.npz", arcname="temporal_best_vjepa.npz")
        tf.add("/tmp/temporal_scan_summary.json", arcname="temporal_scan_summary.json")
    print("[SAVE] /tmp/temporal_best_vjepa.tar.gz", flush=True)
    print(f"\n[DONE] total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
