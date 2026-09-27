"""videomae_vjepa_comparison.py

Add VideoMAE-large (MCG-NJU/videomae-large) as a genuine video-native
encoder baseline to the existing 5-encoder single-frame invariance
comparison (V-JEPA 2, DINOv3, DINOv2, CLIP, SigLIP 2), addressing the
AAAI-27 reviewer concern that the invariance test only compares V-JEPA
against IMAGE encoders.

This script mirrors dump_encoder_5x.py's EXACT protocol:
  - Same 15 SDSD-indoor paired sequences (SEQ_NAMES), same loader
    (load_pair_frames: center 16-frame clip, 512x512 center crop).
  - Same metric: per-patch relative L2 shift
        rho[i,j] = ||feat_LQ[i,j] - feat_HQ[i,j]||_2 / (||feat_HQ[i,j]||_2 + 1e-6)
    averaged over the (h,w) grid -> one scalar (rho_patch) per sequence.
  - VideoMAE is fed the actual 16-frame clip NATIVELY (unlike the other
    5 encoders, which are image encoders run per-frame and averaged over
    time): VideoMAE-large is pretrained on 16-frame clips @ 224x224 with
    tubelet_size=2, so a single forward pass already fuses time into the
    8 temporal tokens. We take the last_hidden_state (last transformer
    block, pre-decoder; this is the standard "encoder features" used by
    VideoMAEModel, analogous to the other encoders' last_hidden_state),
    reshape to (Tp=8, Hp=14, Wp=14, D=1024), and average over the
    temporal token dim exactly like V-JEPA's own
    `.view(1, Tp, Hp, Wp, D).mean(dim=1)` -- so the final aggregation
    style matches V-JEPA's, not the "per-frame-then-average-pixels"
    style used for the image encoders.

We RE-RUN V-JEPA on the same 15 pairs in this script (rather than only
trusting the previously-saved dump_encoder_5x.py numbers) so that the
V-JEPA vs VideoMAE paired comparison is computed from features produced
in the exact same process/run, removing any risk of environment/version
drift between the two encoders' numbers.

Output (all under quick_ablations/videomae_comparison/):
  videomae_vjepa_raw.json       -- per-sequence rho_patch for both encoders
  videomae_vjepa_pairwise.json  -- Wilcoxon / bootstrap CI / Cohen's d / win count
  videomae_vjepa_comparison.log -- full run log (written by caller via tee)
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/cache/zihaohe/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/zihaohe/hf_cache")

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/vjepa_llve")
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa

DEVICE = torch.device("cuda")

SDSD_ROOT  = "/cache/zihaohe/datasets/SDSD_indoor/indoor_static_np"
VJEPA_REPO = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT = "/cache/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"
VMAE_ID    = "MCG-NJU/videomae-large"
CACHE_DIR  = "/cache/zihaohe/hf_cache"

OUT_DIR      = "/home/ma-user/workspace/zihaohe/quick_ablations/videomae_comparison"
RAW_JSON     = os.path.join(OUT_DIR, "videomae_vjepa_raw.json")
PAIRWISE_JSON = os.path.join(OUT_DIR, "videomae_vjepa_pairwise.json")

SEQ_NAMES = [
    "pair50","pair52","pair58","pair5_2","pair6","pair60","pair68","pair6_2",
    "pair7","pair70","pair7_2","pair8","pair8_2","pair9","pair9_2",
]

CLIP_LEN = 16
CROP     = 512
TUBELET  = 2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# Existing 5-encoder means, from dump_encoder_5x.py / Table tab:encoder_stats
# (reported verbatim for context in the printed comparison table; NOT
# recomputed here -- provenance: /tmp/encoder_features_5x_summary.json as
# summarized in sections/07_appendix.tex).
EXISTING_MEANS = {
    "V-JEPA 2": 0.431,
    "DINOv3":   0.434,
    "DINOv2":   0.462,
    "CLIP":     0.554,
    "SigLIP2":  0.603,
}


# ================================================================= data
def load_pair_frames(seq: str, side: str) -> np.ndarray:
    d = os.path.join(SDSD_ROOT, side, seq)
    fns = sorted(os.listdir(d))
    T = len(fns)
    assert T >= CLIP_LEN, f"{seq}/{side}: only {T} frames"
    start = (T - CLIP_LEN) // 2
    fns = fns[start:start + CLIP_LEN]
    arrs = []
    for fn in fns:
        a = np.load(os.path.join(d, fn))
        H, W, _ = a.shape
        y0 = (H - CROP) // 2
        x0 = (W - CROP) // 2
        arrs.append(a[y0:y0+CROP, x0:x0+CROP, :])
    return np.stack(arrs, axis=0)   # (T, 512, 512, 3) uint8


# ================================================================= preproc
def _to_bchw01(frames_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


def _resize_norm(x_bchw: torch.Tensor, size: int, mean: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x_bchw, size=(size, size), mode="bilinear",
                      align_corners=False)
    m = mean.to(x.device).view(1, 3, 1, 1)
    s = std.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


# ================================================================= V-JEPA
def build_vjepa():
    return VJepaBackboneLocal21(
        local_repo=VJEPA_REPO,
        local_ckpt=VJEPA_CKPT,
        feature_layers=(23,),
        image_size=CROP,
        frames_per_clip=CLIP_LEN,
        patch_size=16,
        tubelet_size=TUBELET,
        freeze=True,
    ).to(DEVICE).eval()


@torch.no_grad()
def vjepa_patch_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()   # (1,3,T,H,W)
    out = model(x)
    feats = out["feats"][0]                               # (1, N, 1024)
    Tp, Hp, Wp = out["token_grid"]
    D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D).mean(dim=1).squeeze(0)   # (Hp, Wp, D)
    return f.detach().float().cpu().numpy()


# ================================================================= VideoMAE
@torch.no_grad()
def videomae_patch_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    """VideoMAE-large is a proper spatio-temporal ViT: pretrained natively
    on 16-frame clips @ 224x224, tubelet_size=2, patch_size=16.
    Feed the FULL 16-frame clip in a single forward pass (native temporal
    modeling, unlike the per-frame-then-average image encoders).

    Token layout of VideoMAEModel.last_hidden_state: (B, Tp*Hp*Wp, D) with
    Tp=16/2=8, Hp=Wp=224/16=14, D=1024. No CLS token (use_mean_pooling in
    config only affects the classification head, not encoder outputs).

    Reshape to (1, Tp, Hp, Wp, D) and average over the temporal token dim,
    matching V-JEPA's own aggregation exactly:
        `.view(1, Tp, Hp, Wp, D).mean(dim=1)`
    """
    x = _to_bchw01(frames_uint8)                          # (16, 3, H, W)
    x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    x = x.unsqueeze(0)                                    # (1, 16, 3, 224, 224)
    out = model(pixel_values=x)
    tokens = out.last_hidden_state                        # (1, Tp*Hp*Wp, D)
    B, N, D = tokens.shape
    Tp = CLIP_LEN // TUBELET                              # 8
    N_spatial = N // Tp
    Hp = Wp = int(round(N_spatial ** 0.5))
    assert Hp * Wp * Tp == N, f"unexpected VideoMAE token count {N} (Tp={Tp})"
    f = tokens.view(1, Tp, Hp, Wp, D).mean(dim=1).squeeze(0)   # (Hp, Wp, D)
    return f.detach().float().cpu().numpy(), (Tp, Hp, Wp, D)


# ================================================================= maps
def compute_maps(lq_feat: np.ndarray, hq_feat: np.ndarray):
    lq = lq_feat.astype(np.float64)
    hq = hq_feat.astype(np.float64)
    diff = lq - hq
    diff_norm = np.linalg.norm(diff, axis=-1)
    hq_norm   = np.linalg.norm(hq,   axis=-1)
    lq_norm   = np.linalg.norm(lq,   axis=-1)
    shift = diff_norm / (hq_norm + 1e-6)
    return (
        shift.astype(np.float32),
        lq_norm.astype(np.float32),
        hq_norm.astype(np.float32),
    )


# ================================================================= stats
def bootstrap_ci_mean(vals, n_boot=10000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=np.float64); N = len(v)
    boots = rng.choice(v, size=(n_boot, N), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return lo, hi


def bootstrap_ci_delta(a, b, n_boot=10000, ci=95, seed=0):
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


# ================================================================= main
def main():
    from transformers import VideoMAEModel
    from scipy.stats import wilcoxon

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[*] loading frames for all 15 pairs ...", flush=True)
    lq_frames, hq_frames = {}, {}
    for seq in SEQ_NAMES:
        lq_frames[seq] = load_pair_frames(seq, "input")
        hq_frames[seq] = load_pair_frames(seq, "GT")
    print(f"    loaded 15 pairs in {time.time()-t0:.1f}s", flush=True)

    # ---- V-JEPA (re-run fresh in this same process for a same-run comparison) ----
    print("[*] loading V-JEPA 2.1 ViT-L ...", flush=True)
    vjepa = build_vjepa()
    vjepa_rho = {}
    for seq in SEQ_NAMES:
        lq_feat = vjepa_patch_feats(vjepa, lq_frames[seq])
        hq_feat = vjepa_patch_feats(vjepa, hq_frames[seq])
        s, ln, hn = compute_maps(lq_feat, hq_feat)
        vjepa_rho[seq] = float(s.mean())
        print(f"    vjepa  {seq:>9s}  rho_patch={vjepa_rho[seq]:.4f}", flush=True)
    del vjepa
    torch.cuda.empty_cache()

    # ---- VideoMAE-large ----
    print("[*] loading VideoMAE-large (MCG-NJU/videomae-large) ...", flush=True)
    vmae = VideoMAEModel.from_pretrained(VMAE_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    print(f"    config: num_frames={vmae.config.num_frames} "
          f"tubelet_size={vmae.config.tubelet_size} "
          f"patch_size={vmae.config.patch_size} "
          f"image_size={vmae.config.image_size} "
          f"hidden_size={vmae.config.hidden_size}", flush=True)

    vmae_rho = {}
    grid_info = None
    for seq in SEQ_NAMES:
        lq_feat, grid_info = videomae_patch_feats(vmae, lq_frames[seq])
        hq_feat, _ = videomae_patch_feats(vmae, hq_frames[seq])
        s, ln, hn = compute_maps(lq_feat, hq_feat)
        vmae_rho[seq] = float(s.mean())
        print(f"    vmae   {seq:>9s}  rho_patch={vmae_rho[seq]:.4f}", flush=True)
    del vmae
    torch.cuda.empty_cache()

    # ================================================================= raw dump
    raw = {
        "meta": {
            "task": "single-frame invariance (rho_L2 relative per-patch shift), "
                     "VideoMAE-large added as genuine video-native encoder baseline",
            "metric": "rho_patch = mean_(h,w) ||feat_LQ - feat_HQ||_2 / (||feat_HQ||_2 + 1e-6)",
            "n_pairs": len(SEQ_NAMES),
            "pairs": SEQ_NAMES,
            "crop": CROP,
            "clip_len": CLIP_LEN,
            "vjepa_grid": "Tp=8, Hp=Wp=32 (512px, patch16, tubelet2), T-mean of block 23",
            "videomae_grid": f"Tp={grid_info[0]}, Hp={grid_info[1]}, Wp={grid_info[2]}, D={grid_info[3]} "
                              f"(224px native, patch16, tubelet2, T-mean of last_hidden_state)",
            "videomae_native_temporal": True,
            "note": "VideoMAE consumes the full 16-frame clip in ONE forward pass "
                     "(native spatio-temporal ViT), unlike the 5 image encoders "
                     "(CLIP/DINOv2/DINOv3/SigLIP2) which run per-frame and are "
                     "averaged over time post-hoc. V-JEPA re-run fresh in this "
                     "script (same process) for an apples-to-apples same-run pair "
                     "with VideoMAE; V-JEPA numbers here are expected to match "
                     "dump_encoder_5x.py's saved values up to float precision.",
        },
        "vjepa_rho_patch": vjepa_rho,
        "videomae_rho_patch": vmae_rho,
    }
    with open(RAW_JSON, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"[*] wrote {RAW_JSON}", flush=True)

    # ================================================================= pairwise stats
    v_vec = np.array([vjepa_rho[s] for s in SEQ_NAMES], dtype=np.float64)
    m_vec = np.array([vmae_rho[s]  for s in SEQ_NAMES], dtype=np.float64)

    v_mean, v_std = float(v_vec.mean()), float(v_vec.std(ddof=1))
    m_mean, m_std = float(m_vec.mean()), float(m_vec.std(ddof=1))
    v_lo, v_hi = bootstrap_ci_mean(v_vec)
    m_lo, m_hi = bootstrap_ci_mean(m_vec)

    # Same convention as dump_encoder_spatial_maps_v2.py: lower shift = more
    # invariant. H1: V-JEPA's shift < VideoMAE's shift (V-JEPA "wins").
    try:
        wres = wilcoxon(v_vec, m_vec, alternative="less", zero_method="pratt")
        wilcoxon_p = float(wres.pvalue)
        wilcoxon_stat = float(wres.statistic)
        wilcoxon_error = None
    except ValueError as e:
        wilcoxon_p = None
        wilcoxon_stat = None
        wilcoxon_error = str(e)

    cohens_d = paired_cohens_d(v_vec, m_vec)
    delta_mean = float((v_vec - m_vec).mean())   # V-JEPA minus VideoMAE; negative = V-JEPA better
    delta_lo, delta_hi = bootstrap_ci_delta(v_vec.tolist(), m_vec.tolist())
    n_vjepa_wins = int(np.sum(v_vec < m_vec))   # sequences where V-JEPA has LOWER (better) shift

    pairwise = {
        "meta": {
            "comparison": "V-JEPA 2.1 vs VideoMAE-large, paired per-sequence rho_patch, N=15 SDSD-indoor pairs",
            "methodology": "identical to dump_encoder_spatial_maps_v2.py / Table tab:encoder_pairwise: "
                            "paired one-sided Wilcoxon signed-rank test (H1: V-JEPA shift < competitor shift, "
                            "zero_method='pratt'), bootstrap 95% CI on mean and on difference-of-means "
                            "(10,000 resamples, seed=0), paired Cohen's d, count of sequences where V-JEPA wins "
                            "(strictly lower shift).",
            "n": 15,
        },
        "V-JEPA 2.1": {
            "mean": v_mean, "std": v_std, "ci95_lo": v_lo, "ci95_hi": v_hi,
            "per_seq": dict(zip(SEQ_NAMES, v_vec.tolist())),
        },
        "VideoMAE-L": {
            "mean": m_mean, "std": m_std, "ci95_lo": m_lo, "ci95_hi": m_hi,
            "per_seq": dict(zip(SEQ_NAMES, m_vec.tolist())),
        },
        "pairwise_stats": {
            "wilcoxon_p_vjepa_less": wilcoxon_p,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_error": wilcoxon_error,
            "cohens_d_paired_vjepa_minus_videomae": cohens_d,
            "delta_mean_vjepa_minus_videomae": delta_mean,
            "delta_ci95_lo": delta_lo,
            "delta_ci95_hi": delta_hi,
            "n_sequences_vjepa_wins": n_vjepa_wins,
            "n_sequences_total": 15,
        },
        "existing_5encoder_means_for_context": EXISTING_MEANS,
    }
    with open(PAIRWISE_JSON, "w") as f:
        json.dump(pairwise, f, indent=2)
    print(f"[*] wrote {PAIRWISE_JSON}", flush=True)

    # ================================================================= print table
    print("\n" + "=" * 90)
    print("6-encoder ranking (mean rho_patch, lower = more LQ/HQ invariant)")
    print("-" * 90)
    all_means = dict(EXISTING_MEANS)
    all_means["VideoMAE-L"] = m_mean
    all_means["V-JEPA 2 (this run)"] = v_mean
    for k, v in sorted(all_means.items(), key=lambda kv: kv[1]):
        print(f"  {k:<24s} {v:.4f}")
    print("-" * 90)
    print(f"V-JEPA vs VideoMAE: delta(V-JEPA - VideoMAE) = {delta_mean:+.5f} "
          f"[{delta_lo:.5f},{delta_hi:.5f}]  wilcoxon p(V<VMAE)={wilcoxon_p}  "
          f"cohens_d={cohens_d:+.4f}  V-JEPA wins {n_vjepa_wins}/15")
    print("=" * 90)
    print(f"[DONE] total wall time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
