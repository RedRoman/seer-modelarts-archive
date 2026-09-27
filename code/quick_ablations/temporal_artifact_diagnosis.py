"""temporal_artifact_diagnosis.py

Diagnoses a possible methodological confound in temporal_scan_v3.py's
encoder temporal-coherence comparison: does DINOv3's inflated norm-map
adjacent-Pearson correlation on SDSD-outdoor sequences come from a small
set of "register-like" high-norm outlier PATCH tokens with spatially fixed
positions (Darcet et al., "Vision Transformers Need Registers", ICLR24),
rather than from genuine temporal coherence of ordinary content tokens?

This script reuses all data-loading / model-building / norm-extraction
code from temporal_scan_v3.py (imported directly, guarded by
`if __name__ == "__main__"` in that file so import is side-effect-free
except for the module-level HF_ENDPOINT env defaults, constants, and
function/class definitions).

Steps
-----
1. Outlier-token diagnostics on a subset of sequences (3 SDSD-outdoor +
   2 BVI) for VJEPA + DINOv3: max/mean ratio, excess kurtosis, outlier
   fraction (>3x per-frame median), and spatial position stability of the
   top-3 highest-norm patches across the 8 burst frames.
2. Outlier-excluded norm-Pearson metric (k=5%, k=10% patches excluded,
   union-of-both-frames rule) on the full 11 seqs x 6 encoders, reusing
   cached raw norm maps.
3. Content-based cosine similarity metric (raw feature vectors, not norm
   maps) on the same 11 seqs x 6 encoders, plus an outlier-excluded
   variant.
4. (optional / likely skipped) V-JEPA predictive coherence via the
   predictor module -- only attempted if it's a cheap, low-risk addition;
   otherwise explicitly marked attempted=false with a reason.

Output: /home/ma-user/workspace/zihaohe/quick_ablations/temporal_artifact_diagnosis.json
"""
from __future__ import annotations
import os, sys, json, time, traceback
import numpy as np
import torch

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/quick_ablations")
from temporal_scan_v3 import (  # noqa
    BVI_ROOT, SDSD_ROOT, VJEPA_REPO, VJEPA_CKPT, CLIP_ID, DINO_ID, DINO3_ID,
    SIGLIP2_ID, VMAE_ID, CACHE_DIR, CROP, N_SAMPLE, CLIP_LEN, TUBELET,
    IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD, SIGLIP_MEAN, SIGLIP_STD,
    center_crop, _load_bvi, _load_sdsd, list_bvi, list_sdsd, burst_indices,
    build_pair_clip, load_frames, _to_bchw01, _resize_norm, build_vjepa,
    _cent_cos, adjacent_pearson, bootstrap_ci_mean, bootstrap_ci_delta,
    paired_cohens_d, DEVICE,
)

OUT_JSON = "/home/ma-user/workspace/zihaohe/quick_ablations/temporal_artifact_diagnosis.json"

# ------------------------------------------------------------- candidates
CANDIDATES = [
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

STEP1_SEQS = {"MVI_1007", "MVI_1011", "MVI_1009", "S35_Bunny__10fps", "S39_Books4__10fps"}

ENCODERS = [
    ("vjepa",    "V-JEPA-2.1"),
    ("dinov3",   "DINOv3"),
    ("dinov2",   "DINOv2"),
    ("siglip",   "SigLIP2"),
    ("clip",     "CLIP"),
    ("videomae", "VideoMAE-L"),
]

K_LIST = [5, 10]   # percent excluded


# ============================================================= data prep
def prepare_sequences(candidates):
    prepared = []
    for dset, name, lister, arg in candidates:
        files = lister(*arg) if isinstance(arg, tuple) else lister(arg)
        T = len(files)
        if T < N_SAMPLE:
            print(f"[SKIP] {name}: T={T}<{N_SAMPLE}")
            continue
        loader = _load_sdsd if dset.startswith("SDSD") else _load_bvi
        idx = burst_indices(T)
        pair, delta = build_pair_clip(idx, T)
        thumbs = load_frames(files, idx,  loader)   # (8, 512, 512, 3)
        clip16 = load_frames(files, pair, loader)   # (16, 512, 512, 3)
        prepared.append(dict(dataset=dset, name=name, T=T, idx=idx,
                              delta=delta, thumbs=thumbs, clip16=clip16))
        print(f"[LOAD] {name:22s} T={T} idx={idx} delta={delta}", flush=True)
    return prepared


# ============================================================= feature extractors
# Variants of temporal_scan_v3's *_per_frame_norm that return raw (Tp,Hp,Wp,D)
# features (float32) instead of / in addition to the norm map. Norm maps are
# trivially derived via `.norm(dim=-1)` from the same tensor, so downstream
# code caches only the raw features and computes norms as needed.

@torch.no_grad()
def vjepa_per_frame_feats(model, clip16: np.ndarray) -> np.ndarray:
    """clip16: (16,H,W,3) uint8 -> (8, 32, 32, D) float32 features."""
    x = torch.from_numpy(clip16).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    out = model(x)
    feats = out["feats"][0]
    Tp, Hp, Wp = out["token_grid"]
    D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D).squeeze(0)
    return f.detach().float().cpu().numpy()


@torch.no_grad()
def clip_per_frame_feats(model, frames8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames8); x = _resize_norm(x, 224, CLIP_MEAN, CLIP_STD)
    tk = model(pixel_values=x).last_hidden_state[:, 1:, :]
    B, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(B, g, g, D).detach().float().cpu().numpy()


@torch.no_grad()
def dinov2_per_frame_feats(model, frames8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames8); x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    tk = model(pixel_values=x).last_hidden_state[:, 1:, :]
    B, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(B, g, g, D).detach().float().cpu().numpy()


@torch.no_grad()
def dinov3_per_frame_feats(model, frames8: np.ndarray, n_prefix: int) -> np.ndarray:
    x = _to_bchw01(frames8); x = _resize_norm(x, 256, IMAGENET_MEAN, IMAGENET_STD)
    tk = model.forward_features(x)[:, n_prefix:, :]
    B, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(B, g, g, D).detach().float().cpu().numpy()


@torch.no_grad()
def siglip_per_frame_feats(model, frames8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames8); x = _resize_norm(x, 256, SIGLIP_MEAN, SIGLIP_STD)
    tk = model(pixel_values=x).last_hidden_state
    B, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(B, g, g, D).detach().float().cpu().numpy()


@torch.no_grad()
def videomae_per_frame_feats(model, clip16: np.ndarray) -> np.ndarray:
    x = _to_bchw01(clip16)
    x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    x = x.unsqueeze(0)
    tk = model(pixel_values=x).last_hidden_state
    B, N, D = tk.shape
    Tp = CLIP_LEN // TUBELET
    N_spatial = N // Tp
    g = int(round(N_spatial ** 0.5))
    assert g * g * Tp == N, f"unexpected VideoMAE token count {N}"
    f = tk.view(B, Tp, g, g, D).squeeze(0)
    return f.detach().float().cpu().numpy()


def feats_to_norm(feats: np.ndarray) -> np.ndarray:
    """(N,h,w,D) float32 -> (N,h,w) float32 L2 norm map."""
    return np.linalg.norm(feats, axis=-1)


# ============================================================= Step 1: outlier diagnostics
def outlier_diagnostics_for_seq(norm_maps: np.ndarray) -> dict:
    """norm_maps: (N, h, w). Returns per-frame stats + cross-frame position
    stability of top-3 highest-norm patches."""
    N, h, w = norm_maps.shape
    flat = norm_maps.reshape(N, -1)   # (N, h*w)

    per_frame = []
    top3_sets = []
    for t in range(N):
        v = flat[t]
        mean_v = float(v.mean())
        max_v = float(v.max())
        median_v = float(np.median(v))
        ratio = max_v / (mean_v + 1e-9)
        # excess kurtosis (Fisher), matches scipy.stats.kurtosis(fisher=True)
        from scipy.stats import kurtosis
        kurt = float(kurtosis(v, fisher=True, bias=True))
        outlier_frac = float(np.mean(v > 3.0 * median_v))
        top3_idx = np.argsort(v)[-3:][::-1].tolist()   # descending, flat indices
        top3_sets.append(set(top3_idx))
        per_frame.append({
            "frame": t,
            "max_mean_ratio": ratio,
            "excess_kurtosis": kurt,
            "outlier_frac_gt3x_median": outlier_frac,
            "top3_flat_idx": top3_idx,
        })

    # position stability: overlap of each frame's top3 set with frame 0's top3 set
    ref = top3_sets[0]
    overlaps = [len(ref & s) for s in top3_sets]           # 0..3, frame0 vs itself = 3
    jaccard = [len(ref & s) / len(ref | s) if len(ref | s) > 0 else 0.0
               for s in top3_sets]
    frac_frames_ge1_overlap = float(np.mean([o >= 1 for o in overlaps]))
    frac_frames_full_overlap = float(np.mean([o == 3 for o in overlaps]))

    # Also: global position stability -- how many *unique* positions ever
    # appear across all frames' top-3 sets (3 = perfectly fixed single position
    # doesn't reduce it since it's per-frame top3; use union size out of N*3 max)
    union_positions = set().union(*top3_sets)

    return {
        "n_frames": N, "grid_h": h, "grid_w": w,
        "per_frame": per_frame,
        "mean_max_mean_ratio": float(np.mean([pf["max_mean_ratio"] for pf in per_frame])),
        "mean_excess_kurtosis": float(np.mean([pf["excess_kurtosis"] for pf in per_frame])),
        "mean_outlier_frac": float(np.mean([pf["outlier_frac_gt3x_median"] for pf in per_frame])),
        "top3_overlap_with_frame0": overlaps,
        "top3_jaccard_with_frame0": jaccard,
        "frac_frames_ge1_overlap_with_frame0": frac_frames_ge1_overlap,
        "frac_frames_full_overlap_with_frame0": frac_frames_full_overlap,
        "n_unique_top3_positions_across_all_frames": len(union_positions),
        "max_possible_unique_positions": 3 * N,
    }


# ============================================================= Step 2: outlier-excluded Pearson
def excluded_pearson(norm_maps: np.ndarray, k_percent: float) -> float:
    """norm_maps: (N,h,w). For each adjacent pair (i,i+1): mask out patches
    that are in the top-k% by norm value in EITHER frame i or frame i+1
    (union rule), then compute mean-centered cosine (Pearson) over the
    surviving positions only. Returns mean over adjacent pairs."""
    N = norm_maps.shape[0]
    v = norm_maps.reshape(N, -1)
    P = v.shape[1]
    n_excl = max(1, int(round(P * k_percent / 100.0)))

    def top_mask(row):
        idx = np.argsort(row)[-n_excl:]
        m = np.zeros(P, dtype=bool)
        m[idx] = True
        return m

    rs = []
    for i in range(N - 1):
        excl = top_mask(v[i]) | top_mask(v[i + 1])
        keep = ~excl
        if keep.sum() < 3:
            # degenerate; fall back to no exclusion for this pair
            keep = np.ones(P, dtype=bool)
        a, b = v[i][keep], v[i + 1][keep]
        rs.append(_cent_cos(a, b))
    return float(np.mean(rs))


# ============================================================= Step 3: content cosine
def content_cosine(feats: np.ndarray, excl_mask_per_frame=None) -> float:
    """feats: (N,h,w,D) float32. For each adjacent pair, per-patch cosine sim
    between D-dim vectors at same spatial position, averaged over patches
    (optionally excluding masked patch positions), then averaged over pairs.
    excl_mask_per_frame: optional (N, h*w) bool array, True = excluded."""
    N, h, w, D = feats.shape
    flat = feats.reshape(N, h * w, D)
    norms = np.linalg.norm(flat, axis=-1, keepdims=True)
    unit = flat / (norms + 1e-9)

    rs = []
    for i in range(N - 1):
        cos = np.sum(unit[i] * unit[i + 1], axis=-1)   # (h*w,)
        if excl_mask_per_frame is not None:
            excl = excl_mask_per_frame[i] | excl_mask_per_frame[i + 1]
            keep = ~excl
            if keep.sum() < 3:
                keep = np.ones(h * w, dtype=bool)
            cos = cos[keep]
        rs.append(float(cos.mean()))
    return float(np.mean(rs))


def topk_mask_per_frame(norm_maps: np.ndarray, k_percent: float) -> np.ndarray:
    """norm_maps: (N,h,w) -> (N, h*w) bool mask, True where patch is in the
    top-k% highest-norm patches of that frame."""
    N = norm_maps.shape[0]
    v = norm_maps.reshape(N, -1)
    P = v.shape[1]
    n_excl = max(1, int(round(P * k_percent / 100.0)))
    mask = np.zeros_like(v, dtype=bool)
    for t in range(N):
        idx = np.argsort(v[t])[-n_excl:]
        mask[t, idx] = True
    return mask


# ============================================================= stats aggregation
def aggregate_metric(values_by_seq: dict, vjepa_key="vjepa"):
    """values_by_seq: {seq_name: {encoder_key: value}}. Returns summary dict
    per encoder with mean/std/CI/wilcoxon vs vjepa, matching temporal_scan_v3
    style."""
    from scipy.stats import wilcoxon
    seq_names = list(values_by_seq.keys())
    encoders_present = ENCODERS

    v_vec = np.array([values_by_seq[s].get(vjepa_key) for s in seq_names], dtype=np.float64)

    summary = {}
    for k, label in encoders_present:
        vals_list = [values_by_seq[s].get(k) for s in seq_names]
        if any(v is None for v in vals_list):
            summary[k] = {"label": label, "status": "missing"}
            continue
        vals = np.array(vals_list, dtype=np.float64)
        m = float(vals.mean()); s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        lo, hi = bootstrap_ci_mean(vals)
        entry = {
            "label": label, "mean": m, "std": s,
            "ci95_lo": lo, "ci95_hi": hi, "n": int(len(vals)),
            "per_seq_values": dict(zip(seq_names, vals.tolist())),
        }
        if k != vjepa_key:
            try:
                wres = wilcoxon(v_vec, vals, alternative="greater", zero_method="pratt")
                entry["wilcoxon_p_vs_vjepa"] = float(wres.pvalue)
                entry["wilcoxon_stat"] = float(wres.statistic)
            except ValueError as e:
                entry["wilcoxon_p_vs_vjepa"] = None
                entry["wilcoxon_error"] = str(e)
            entry["cohens_d_paired_vs_vjepa"] = paired_cohens_d(v_vec, vals)
            entry["delta_mean_vjepa_minus"] = float((v_vec - vals).mean())
            dlo, dhi = bootstrap_ci_delta(v_vec.tolist(), vals.tolist())
            entry["delta_ci95_lo"] = dlo
            entry["delta_ci95_hi"] = dhi
        summary[k] = entry
    return summary


# ============================================================= main
def main():
    t_all = time.time()
    prepared = prepare_sequences(CANDIDATES)
    n_seq = len(prepared)
    print(f"\n[*] {n_seq} sequences prepared\n", flush=True)

    n_prefix_dinov3 = None
    videomae_failed = None

    # cache: seq_name -> encoder_key -> np.ndarray raw feats (Tp/8, Hp, Wp, D) float32
    feats_cache = {p["name"]: {} for p in prepared}

    # ---------------------------------------------------------- V-JEPA
    print("[*] V-JEPA 2.1 ViT-L ...", flush=True); t0 = time.time()
    vj = build_vjepa()
    print(f"    loaded +{time.time()-t0:.1f}s", flush=True)
    for p in prepared:
        f = vjepa_per_frame_feats(vj, p["clip16"])
        feats_cache[p["name"]]["vjepa"] = f
        print(f"    {p['name']:22s} feat shape={f.shape}", flush=True)
    del vj; torch.cuda.empty_cache()

    # ---------------------------------------------------------- DINOv3
    import timm
    print("\n[*] DINOv3 ViT-L/16 (timm) ...", flush=True)
    d3 = timm.create_model(DINO3_ID, pretrained=True, num_classes=0).to(DEVICE).eval()
    n_prefix_dinov3 = int(getattr(d3, "num_prefix_tokens", 5))
    for p in prepared:
        f = dinov3_per_frame_feats(d3, p["thumbs"], n_prefix_dinov3)
        feats_cache[p["name"]]["dinov3"] = f
        print(f"    {p['name']:22s} feat shape={f.shape}", flush=True)
    del d3; torch.cuda.empty_cache()

    # ---------------------------------------------------------- DINOv2
    from transformers import Dinov2Model, CLIPVisionModel, SiglipVisionModel
    print("\n[*] DINOv2 ViT-L/14 ...", flush=True)
    d2 = Dinov2Model.from_pretrained(DINO_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    for p in prepared:
        f = dinov2_per_frame_feats(d2, p["thumbs"])
        feats_cache[p["name"]]["dinov2"] = f
        print(f"    {p['name']:22s} feat shape={f.shape}", flush=True)
    del d2; torch.cuda.empty_cache()

    # ---------------------------------------------------------- CLIP
    print("\n[*] CLIP ViT-L/14 ...", flush=True)
    cl = CLIPVisionModel.from_pretrained(CLIP_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    for p in prepared:
        f = clip_per_frame_feats(cl, p["thumbs"])
        feats_cache[p["name"]]["clip"] = f
        print(f"    {p['name']:22s} feat shape={f.shape}", flush=True)
    del cl; torch.cuda.empty_cache()

    # ---------------------------------------------------------- SigLIP2
    print("\n[*] SigLIP 2 ViT-L/16 ...", flush=True)
    sg = SiglipVisionModel.from_pretrained(SIGLIP2_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    for p in prepared:
        f = siglip_per_frame_feats(sg, p["thumbs"])
        feats_cache[p["name"]]["siglip"] = f
        print(f"    {p['name']:22s} feat shape={f.shape}", flush=True)
    del sg; torch.cuda.empty_cache()

    # ---------------------------------------------------------- VideoMAE
    try:
        from transformers import VideoMAEModel
        print("\n[*] VideoMAE-large ...", flush=True)
        vm = VideoMAEModel.from_pretrained(VMAE_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
        for p in prepared:
            f = videomae_per_frame_feats(vm, p["clip16"])
            feats_cache[p["name"]]["videomae"] = f
            print(f"    {p['name']:22s} feat shape={f.shape}", flush=True)
        del vm; torch.cuda.empty_cache()
    except Exception as e:
        videomae_failed = f"{type(e).__name__}: {e}"
        print(f"[WARN] VideoMAE failed: {videomae_failed}", flush=True)

    print(f"\n[*] All models done, feature extraction complete. "
          f"+{time.time()-t_all:.1f}s\n", flush=True)

    # ======================================================= Step 1: outlier diagnostics
    print("[*] Step 1: outlier-token diagnostics ...", flush=True)
    outlier_diag = {}
    for p in prepared:
        if p["name"] not in STEP1_SEQS:
            continue
        outlier_diag[p["name"]] = {"dataset": p["dataset"]}
        for enc_key in ("vjepa", "dinov3"):
            f = feats_cache[p["name"]].get(enc_key)
            if f is None:
                continue
            nm = feats_to_norm(f)
            outlier_diag[p["name"]][enc_key] = outlier_diagnostics_for_seq(nm)
        print(f"    done {p['name']}", flush=True)

    # ======================================================= Step 2: outlier-excluded Pearson
    print("\n[*] Step 2: outlier-excluded norm-Pearson metric ...", flush=True)
    # original (unexcluded) recomputed here too, for internal consistency check
    orig_by_seq = {p["name"]: {} for p in prepared}
    excl_by_seq = {k: {p["name"]: {} for p in prepared} for k in K_LIST}

    for p in prepared:
        for enc_key, _ in ENCODERS:
            f = feats_cache[p["name"]].get(enc_key)
            if f is None:
                continue
            nm = feats_to_norm(f)
            orig_by_seq[p["name"]][enc_key] = adjacent_pearson(nm)
            for k in K_LIST:
                excl_by_seq[k][p["name"]][enc_key] = excluded_pearson(nm, k)
    print("    computed for all seqs/encoders", flush=True)

    summary_orig_recomputed = aggregate_metric(orig_by_seq)
    summary_excl = {k: aggregate_metric(excl_by_seq[k]) for k in K_LIST}

    # ======================================================= Step 3: content cosine
    print("\n[*] Step 3: content-cosine metric ...", flush=True)
    cos_by_seq = {p["name"]: {} for p in prepared}
    cos_excl_by_seq = {k: {p["name"]: {} for p in prepared} for k in K_LIST}

    for p in prepared:
        for enc_key, _ in ENCODERS:
            f = feats_cache[p["name"]].get(enc_key)
            if f is None:
                continue
            cos_by_seq[p["name"]][enc_key] = content_cosine(f)
            nm = feats_to_norm(f)
            for k in K_LIST:
                mask = topk_mask_per_frame(nm, k)
                cos_excl_by_seq[k][p["name"]][enc_key] = content_cosine(f, excl_mask_per_frame=mask)
    print("    computed for all seqs/encoders", flush=True)

    summary_cos = aggregate_metric(cos_by_seq)
    summary_cos_excl = {k: aggregate_metric(cos_excl_by_seq[k]) for k in K_LIST}

    # ======================================================= Step 4: predictive coherence (skip)
    predictive_coherence = {
        "attempted": False,
        "reason": ("Skipped per task instructions: reconstructing V-JEPA 2.1's "
                   "predictor context/target mask sampling correctly outside the "
                   "training pipeline is nontrivial (app.vjepa_2_1 predictor expects "
                   "specific context/target block mask tensors generated by the "
                   "training-time mask collator, not just raw encoder tokens), and "
                   "hacking together an ad-hoc masking scheme risks producing a "
                   "meaningless or misleading number. Steps 1-3 were prioritized and "
                   "completed; Step 4 was explicitly marked optional/skippable in the "
                   "task spec if uncertain."),
    }

    # ======================================================= assemble output
    original_hardcoded = {
        "source": "/home/ma-user/workspace/zihaohe/quick_ablations/temporal_scan_v3.json (summary.*.mean)",
        "vjepa": 0.49939752320912123,
        "dinov3": 0.6990928214125801,
        "dinov2": 0.6085531257707657,
        "siglip": 0.5832805034033044,
        "clip": 0.3837105055472213,
        "videomae": 0.621425580243667,
    }

    meta = {
        "n_seqs": n_seq,
        "sequences": [p["name"] for p in prepared],
        "step1_sequences": sorted(STEP1_SEQS),
        "step1_encoders": ["vjepa", "dinov3"],
        "k_values_percent": K_LIST,
        "exclusion_rule": (
            "For the outlier-excluded norm-Pearson metric (Step 2) and the "
            "outlier-excluded content-cosine metric (Step 3 variant): for each "
            "adjacent frame pair (i, i+1), a patch spatial position is excluded "
            "from the correlation/cosine average if it is among the top-k% "
            "highest-norm patches of EITHER frame i OR frame i+1 (union rule, "
            "not intersection). This is deliberately conservative -- it removes "
            "more patches than a per-frame-only or intersection-only rule -- "
            "specifically to avoid the residual bias where 'boring surviving "
            "votes agree because they are all background' could still occur if "
            "only one frame's outliers were masked. Top-k% is computed by count "
            "per frame (e.g. 256 patches at k=5% -> ceil(256*0.05)=13 patches "
            "masked), independently for each frame."
        ),
        "kurtosis_definition": "scipy.stats.kurtosis(fisher=True, bias=True) i.e. excess kurtosis (0 = normal dist).",
        "outlier_frac_definition": "fraction of patches in a frame with norm > 3x that frame's median patch norm.",
        "position_stability_definition": (
            "For each sequence x encoder, take the flat-index set of the top-3 "
            "highest-norm patches in each of the 8 frames. Compare every frame's "
            "top-3 set against frame 0's top-3 set via (a) raw overlap count "
            "(0-3), (b) Jaccard index, (c) fraction of frames with >=1 shared "
            "position with frame 0, (d) fraction of frames with all 3 positions "
            "identical to frame 0, and (e) total number of unique positions that "
            "ever appear in any frame's top-3 (out of a max of 3*8=24 if no reuse "
            "at all; a low number close to 3 means the same few positions are "
            "chosen every frame -- i.e. spatially-fixed outlier tokens)."
        ),
        "content_cosine_definition": (
            "Raw per-patch D-dim feature vectors (post prefix/CLS-token removal, "
            "same tokens used for the norm-map metric) are L2-normalized and "
            "cosine similarity is computed independently at each spatial grid "
            "position between adjacent frames, then averaged over all patches, "
            "then averaged over the 7 adjacent pairs. This directly measures "
            "whether the CONTENT (direction) of a patch's feature vector is "
            "stable across time, independent of its norm -- so it should NOT be "
            "inflated by fixed-position high-norm outlier tokens whose direction "
            "may still vary, and conversely would reveal true temporal coherence "
            "the norm-only metric might miss."
        ),
        "videomae_failed": videomae_failed,
        "encoders_recomputed": [k for k, _ in ENCODERS],
        "original_norm_pearson_hardcoded_from_v3": original_hardcoded,
        "consistency_check_note": (
            "summary.orig_recomputed below recomputes the original (unexcluded) "
            "norm-Pearson metric from the SAME cached raw features used for all "
            "other metrics in this script, as an internal consistency check "
            "against temporal_scan_v3.json's original values (should match to "
            "float precision modulo any burst-index/frame-loading nondeterminism)."
        ),
        "wall_clock_seconds": None,  # filled below
    }

    out = {
        "meta": meta,
        "outlier_diagnostics": outlier_diag,
        "outlier_excluded_metric": {
            "orig_recomputed": summary_orig_recomputed,
            "k5": summary_excl[5],
            "k10": summary_excl[10],
        },
        "content_cosine_metric": {
            "full": summary_cos,
            "k5_excluded": summary_cos_excl[5],
            "k10_excluded": summary_cos_excl[10],
        },
        "predictive_coherence": predictive_coherence,
    }

    meta["wall_clock_seconds"] = time.time() - t_all
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=2)

    # ======================================================= print comparison table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE (aggregate means, 11 seqs unless noted)")
    print("=" * 100)
    header = f"{'encoder':<14s} {'orig(v3)':>10s} {'orig(recomp)':>13s} {'k5-excl':>10s} {'k10-excl':>10s} {'cos-full':>10s} {'cos-k5':>10s} {'cos-k10':>10s}"
    print(header)
    print("-" * len(header))
    for enc_key, label in ENCODERS:
        orig_v3 = original_hardcoded.get(enc_key)
        orig_rc = summary_orig_recomputed.get(enc_key, {}).get("mean")
        k5 = summary_excl[5].get(enc_key, {}).get("mean")
        k10 = summary_excl[10].get(enc_key, {}).get("mean")
        cf = summary_cos.get(enc_key, {}).get("mean")
        c5 = summary_cos_excl[5].get(enc_key, {}).get("mean")
        c10 = summary_cos_excl[10].get(enc_key, {}).get("mean")

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "N/A"

        print(f"{label:<14s} {fmt(orig_v3):>10s} {fmt(orig_rc):>13s} {fmt(k5):>10s} "
              f"{fmt(k10):>10s} {fmt(cf):>10s} {fmt(c5):>10s} {fmt(c10):>10s}")
    print("=" * 100)

    if "dinov3" in summary_excl[5] and "vjepa" in summary_excl[5]:
        gap_orig = original_hardcoded["dinov3"] - original_hardcoded["vjepa"]
        gap_k5 = summary_excl[5]["dinov3"]["mean"] - summary_excl[5]["vjepa"]["mean"]
        gap_k10 = summary_excl[10]["dinov3"]["mean"] - summary_excl[10]["vjepa"]["mean"]
        gap_cos = summary_cos["dinov3"]["mean"] - summary_cos["vjepa"]["mean"]
        print(f"\nDINOv3 - V-JEPA gap: orig={gap_orig:+.4f}  k5-excl={gap_k5:+.4f}  "
              f"k10-excl={gap_k10:+.4f}  content-cosine={gap_cos:+.4f}")
        print(f"Wilcoxon p (dinov3 vs vjepa): orig(v3)=0.999  "
              f"k5={summary_excl[5]['dinov3'].get('wilcoxon_p_vs_vjepa')}  "
              f"k10={summary_excl[10]['dinov3'].get('wilcoxon_p_vs_vjepa')}  "
              f"cos={summary_cos['dinov3'].get('wilcoxon_p_vs_vjepa')}")

    print(f"\n[SAVE] {OUT_JSON}")
    print(f"[DONE] {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
