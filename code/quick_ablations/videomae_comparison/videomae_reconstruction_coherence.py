"""videomae_reconstruction_coherence.py

Reviewer's second concern: the existing "predictive-coherence" test
(App.~\\ref{app:predictive_coherence}, vjepa2_base_predictive_coherence.py)
exercises V-JEPA's own future-token FEATURE prediction objective, which
image encoders cannot do by construction, so it inherently favors V-JEPA.
VideoMAE is a genuine video-native self-supervised model, but its
pretraining objective is masked-PATCH-PIXEL reconstruction (via a decoder,
MSE loss on normalized pixel patches), not future-frame feature
prediction. It is therefore NOT a like-for-like predictor comparison
with V-JEPA's predictor. This script designs the closest architecturally
faithful analog available for VideoMAE and reports it as a separate,
clearly-labeled test -- NOT fused into the V-JEPA predictor numbers.

Design (mirrors the SPIRIT of the V-JEPA test as closely as VideoMAE's own
pretraining head allows):
  - Same 15 SDSD-indoor LQ/HQ pairs used in steps 1-3 (real low-light vs
    real ground-truth clips, 16 frames, 512 crop, resized to native 224).
  - VideoMAE's own token grid is (Tp=8, Hp=14, Wp=14). We mask ALL patches
    of the last 4 temporal tubelets (t=4,5,6,7 -- the "future" half of the
    clip, exactly the same 4 target tubelets used for V-JEPA in
    vjepa2_base_predictive_coherence.py) and leave tubelets t=0..3 fully
    visible as context. This uses VideoMAEForPreTraining's own
    bool_masked_pos mechanism and its own trained decoder + mask token --
    i.e. genuinely exercises the model's own reconstruction objective,
    just with a "future-tubelet" masking pattern instead of the random/tube
    masking used at pretraining time.
  - We take the decoder logits for the masked (future) tokens: these are
    predictions of normalized pixel-patch vectors (config.norm_pix_loss=True,
    the model's own native target space -- NOT V-JEPA's feature space, so
    scores are NOT numerically comparable across the two tests).
  - predictor_score = mean per-patch cosine(predicted_patch, true_normalized_patch)
    over the masked future tokens.
  - naive_score = mean per-patch cosine(true_patch[tubelet t-1], true_patch[tubelet t])
    i.e. "copy the last visible tubelet" -- the same style of naive
    baseline used in the V-JEPA test, computed in the SAME normalized-pixel
    space as the reconstruction target so the two scores are on a common
    scale for this test.
  - We run this separately on LQ and HQ clips and report both, plus the
    predictor-vs-naive gap on each, so the reader can see whether
    VideoMAE's own reconstruction objective is coherent at all on this
    data, and whether it degrades under real low-light corruption
    (LQ) vs ground-truth (HQ) -- the natural video-native analog of "does
    structure still cohere," even though the underlying operation
    (pixel reconstruction) differs fundamentally from V-JEPA's
    (feature prediction).

Caveat (reported honestly, not smoothed over): because the *targets* are
raw normalized pixel intensities rather than learned features, the score
scale is not directly comparable to the V-JEPA predictor-vs-naive numbers,
and LQ pixel noise directly inflates the raw reconstruction task itself
(different noise statistics between LQ/HQ can move the score independent
of "representation coherence"). We report this test as a supplementary,
clearly-scoped result, not as a merged "VideoMAE beats/loses to V-JEPA on
predictive coherence" claim.

Output: videomae_comparison/videomae_reconstruction_coherence.json
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/cache/zihaohe/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/zihaohe/hf_cache")

DEVICE = torch.device("cuda")

SDSD_ROOT  = "/cache/zihaohe/datasets/SDSD_indoor/indoor_static_np"
VMAE_ID    = "MCG-NJU/videomae-large"
CACHE_DIR  = "/cache/zihaohe/hf_cache"

OUT_DIR  = "/home/ma-user/workspace/zihaohe/quick_ablations/videomae_comparison"
OUT_JSON = os.path.join(OUT_DIR, "videomae_reconstruction_coherence.json")

SEQ_NAMES = [
    "pair50","pair52","pair58","pair5_2","pair6","pair60","pair68","pair6_2",
    "pair7","pair70","pair7_2","pair8","pair8_2","pair9","pair9_2",
]

CLIP_LEN = 16
CROP     = 512
TUBELET  = 2
IMG_SIZE = 224
PATCH    = 16

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


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


def _to_bt_chw01(frames_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()   # (T,3,H,W)


def prep_clip(frames_uint8: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (1,T,3,224,224) normalized, matching VideoMAEModel's
    expected pixel_values layout (B, T, C, H, W)."""
    x = _to_bt_chw01(frames_uint8)                       # (T,3,H,W)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                      align_corners=False)
    m = IMAGENET_MEAN.to(DEVICE).view(1, 3, 1, 1)
    s = IMAGENET_STD.to(DEVICE).view(1, 3, 1, 1)
    x = (x - m) / s
    return x.unsqueeze(0)                                 # (1,T,3,224,224)


def tubelet_flat_indices(t, Hp, Wp):
    base = t * Hp * Wp
    return list(range(base, base + Hp * Wp))


def normalized_pixel_patches(pixel_values_bt_chw: torch.Tensor) -> torch.Tensor:
    """Recompute VideoMAEForPreTraining's own label construction (per its
    forward()): unnormalize -> split into (T//ts, ts, C, H//ps, ps, W//ps, ps)
    -> permute -> flatten to (B, N, ts*ps*ps*C) -> per-patch mean/var norm
    (norm_pix_loss=True). Returns (B, N, ts*ps*ps*C) float32, same space as
    decoder logits."""
    device = pixel_values_bt_chw.device
    dtype = pixel_values_bt_chw.dtype
    mean = IMAGENET_MEAN.to(device=device, dtype=dtype)[None, None, :, None, None]
    std  = IMAGENET_STD.to(device=device, dtype=dtype)[None, None, :, None, None]
    frames = pixel_values_bt_chw * std + mean  # back to [0,1], (B,T,C,H,W)

    B, T, C, H, W = frames.shape
    ts, ps = TUBELET, PATCH
    frames = frames.view(B, T // ts, ts, C, H // ps, ps, W // ps, ps)
    frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    frames = frames.view(B, (T // ts) * (H // ps) * (W // ps), ts * ps * ps, C)
    frames_norm = (frames - frames.mean(dim=-2, keepdim=True)) / (
        frames.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    patches = frames_norm.view(B, (T // ts) * (H // ps) * (W // ps), ts * ps * ps * C)
    return patches


@torch.no_grad()
def run_future_tubelet_reconstruction(model, frames_uint8: np.ndarray,
                                       target_tubelets):
    """Mask the given future tubelets entirely; reconstruct via the model's
    own decoder; return per-target-tubelet predictor cosine score and a
    naive 'copy last visible tubelet' cosine score, both in the model's
    native normalized-pixel-patch space."""
    x = prep_clip(frames_uint8)                     # (1,16,3,224,224)
    Tp = CLIP_LEN // TUBELET                          # 8
    Hp = Wp = IMG_SIZE // PATCH                       # 14
    N = Tp * Hp * Wp

    bool_masked = torch.zeros((1, N), dtype=torch.bool, device=DEVICE)
    for t in target_tubelets:
        idx = tubelet_flat_indices(t, Hp, Wp)
        bool_masked[0, idx] = True

    out = model(pixel_values=x, bool_masked_pos=bool_masked)
    logits = out.logits    # (1, num_masked_total, ts*ps*ps*C) -- order matches
                            # bool_masked_pos flattened True-positions, in
                            # the SAME row-major order as tubelet_flat_indices
                            # since our True block per tubelet is contiguous
                            # and tubelets are appended in increasing t order.

    true_patches = normalized_pixel_patches(x)        # (1, N, D_patch)

    per_t_pred = {}
    per_t_naive = {}
    # logits are packed in the order of bool_masked_pos==True positions,
    # which for our construction is exactly concat of tubelets in
    # increasing t (each tubelet contributes Hp*Wp contiguous rows).
    n_per_tubelet = Hp * Wp
    for k, t in enumerate(target_tubelets):
        tgt_idx = tubelet_flat_indices(t, Hp, Wp)
        prev_idx = tubelet_flat_indices(t - 1, Hp, Wp)
        pred_slice = logits[:, k * n_per_tubelet:(k + 1) * n_per_tubelet, :]
        true_tgt = true_patches[:, tgt_idx, :]
        true_prev = true_patches[:, prev_idx, :]

        cos_pred = F.cosine_similarity(pred_slice, true_tgt, dim=-1)
        cos_naive = F.cosine_similarity(true_prev, true_tgt, dim=-1)
        per_t_pred[t] = float(cos_pred.mean().item())
        per_t_naive[t] = float(cos_naive.mean().item())

    return per_t_pred, per_t_naive


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
    from transformers import VideoMAEForPreTraining
    from scipy.stats import wilcoxon

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[*] loading VideoMAEForPreTraining (MCG-NJU/videomae-large) ...", flush=True)
    model = VideoMAEForPreTraining.from_pretrained(VMAE_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    Tp = CLIP_LEN // TUBELET
    target_tubelets = list(range(Tp - 4, Tp))   # [4,5,6,7], mirrors V-JEPA test
    print(f"    Tp={Tp}, target future tubelets (0-indexed): {target_tubelets}", flush=True)

    print("[*] loading frames for all 15 pairs ...", flush=True)
    lq_frames, hq_frames = {}, {}
    for seq in SEQ_NAMES:
        lq_frames[seq] = load_pair_frames(seq, "input")
        hq_frames[seq] = load_pair_frames(seq, "GT")
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)

    per_seq = {}
    lq_pred_scores, lq_naive_scores = [], []
    hq_pred_scores, hq_naive_scores = [], []

    for seq in SEQ_NAMES:
        lq_pt_pred, lq_pt_naive = run_future_tubelet_reconstruction(model, lq_frames[seq], target_tubelets)
        hq_pt_pred, hq_pt_naive = run_future_tubelet_reconstruction(model, hq_frames[seq], target_tubelets)

        lq_pred = float(np.mean(list(lq_pt_pred.values())))
        lq_naive = float(np.mean(list(lq_pt_naive.values())))
        hq_pred = float(np.mean(list(hq_pt_pred.values())))
        hq_naive = float(np.mean(list(hq_pt_naive.values())))

        lq_pred_scores.append(lq_pred); lq_naive_scores.append(lq_naive)
        hq_pred_scores.append(hq_pred); hq_naive_scores.append(hq_naive)

        per_seq[seq] = {
            "LQ_predictor_score": lq_pred, "LQ_naive_score": lq_naive,
            "HQ_predictor_score": hq_pred, "HQ_naive_score": hq_naive,
            "LQ_per_target_tubelet_predictor": lq_pt_pred,
            "LQ_per_target_tubelet_naive": lq_pt_naive,
            "HQ_per_target_tubelet_predictor": hq_pt_pred,
            "HQ_per_target_tubelet_naive": hq_pt_naive,
        }
        print(f"    {seq:>9s}  LQ: pred={lq_pred:.4f} naive={lq_naive:.4f} (gap={lq_pred-lq_naive:+.4f})"
              f"   HQ: pred={hq_pred:.4f} naive={hq_naive:.4f} (gap={hq_pred-hq_naive:+.4f})",
              flush=True)

    del model
    torch.cuda.empty_cache()

    lq_pred_arr = np.array(lq_pred_scores); lq_naive_arr = np.array(lq_naive_scores)
    hq_pred_arr = np.array(hq_pred_scores); hq_naive_arr = np.array(hq_naive_scores)

    def summarize(vals, label):
        m = float(np.mean(vals)); s = float(np.std(vals, ddof=1))
        lo, hi = bootstrap_ci_mean(vals)
        return {"label": label, "mean": m, "std": s, "ci95_lo": lo, "ci95_hi": hi, "n": len(vals)}

    summary = {
        "LQ_predictor": summarize(lq_pred_arr, "VideoMAE reconstruction cosine, future tubelets, LQ input"),
        "LQ_naive":     summarize(lq_naive_arr, "Naive copy-last-tubelet baseline, LQ input"),
        "HQ_predictor": summarize(hq_pred_arr, "VideoMAE reconstruction cosine, future tubelets, HQ (GT) input"),
        "HQ_naive":     summarize(hq_naive_arr, "Naive copy-last-tubelet baseline, HQ (GT) input"),
    }

    # predictor vs naive, within each of LQ / HQ (does the model's own
    # objective actually beat the trivial baseline on this data?)
    for tag, pred_arr, naive_arr in [("LQ", lq_pred_arr, lq_naive_arr), ("HQ", hq_pred_arr, hq_naive_arr)]:
        try:
            wres = wilcoxon(pred_arr, naive_arr, alternative="greater", zero_method="pratt")
            wp, wstat = float(wres.pvalue), float(wres.statistic)
            werr = None
        except ValueError as e:
            wp, wstat, werr = None, None, str(e)
        d = paired_cohens_d(pred_arr, naive_arr)
        delta = float((pred_arr - naive_arr).mean())
        dlo, dhi = bootstrap_ci_delta(pred_arr.tolist(), naive_arr.tolist())
        summary[f"{tag}_predictor_vs_naive"] = {
            "wilcoxon_p_greater": wp, "wilcoxon_stat": wstat, "wilcoxon_error": werr,
            "cohens_d_paired": d, "delta_mean_predictor_minus_naive": delta,
            "delta_ci95_lo": dlo, "delta_ci95_hi": dhi,
        }

    # LQ vs HQ predictor-score comparison (the "coherence under corruption" axis)
    try:
        wres_lqhq = wilcoxon(hq_pred_arr, lq_pred_arr, alternative="greater", zero_method="pratt")
        wp_lqhq, wstat_lqhq = float(wres_lqhq.pvalue), float(wres_lqhq.statistic)
        werr_lqhq = None
    except ValueError as e:
        wp_lqhq, wstat_lqhq, werr_lqhq = None, None, str(e)
    d_lqhq = paired_cohens_d(hq_pred_arr, lq_pred_arr)
    delta_lqhq = float((hq_pred_arr - lq_pred_arr).mean())
    dlo_lqhq, dhi_lqhq = bootstrap_ci_delta(hq_pred_arr.tolist(), lq_pred_arr.tolist())
    summary["HQ_predictor_vs_LQ_predictor"] = {
        "note": "Tests whether VideoMAE's own future-tubelet reconstruction is more "
                "coherent (higher cosine to true normalized pixel patches) on clean "
                "GT clips than on real low-light LQ clips. H1: HQ > LQ.",
        "wilcoxon_p_hq_greater": wp_lqhq, "wilcoxon_stat": wstat_lqhq, "wilcoxon_error": werr_lqhq,
        "cohens_d_paired_hq_minus_lq": d_lqhq,
        "delta_mean_hq_minus_lq": delta_lqhq,
        "delta_ci95_lo": dlo_lqhq, "delta_ci95_hi": dhi_lqhq,
        "n_sequences_hq_greater": int(np.sum(hq_pred_arr > lq_pred_arr)),
        "n_sequences_total": len(SEQ_NAMES),
    }

    out = {
        "meta": {
            "test": "VideoMAE future-tubelet masked reconstruction coherence "
                     "(analog to V-JEPA's future-tubelet feature-prediction coherence test)",
            "IMPORTANT_CAVEAT": (
                "This is NOT numerically comparable to V-JEPA's predictor-vs-naive "
                "cosine scores in vjepa2_base_predictive_coherence.json. V-JEPA's "
                "predictor operates in LEARNED FEATURE space (predicting the "
                "encoder's own 1024-d representation of the target tubelet). "
                "VideoMAE's decoder operates in NORMALIZED PIXEL-PATCH space "
                "(predicting per-patch mean/var-normalized RGB values, its native "
                "MSE pretraining target). The two 'predictor_score' numbers below "
                "are on different scales measuring different things and must not "
                "be reported as a single unified predictive-coherence ranking. "
                "This test is reported as a separate, video-native reconstruction- "
                "coherence result, addressing the reviewer's point that a genuine "
                "video-native self-supervised objective exists for VideoMAE even "
                "though it is not the same objective as V-JEPA's."
            ),
            "sequences": SEQ_NAMES,
            "n_pairs": len(SEQ_NAMES),
            "crop": CROP, "clip_len": CLIP_LEN, "img_size": IMG_SIZE,
            "tubelet_size": TUBELET, "patch_size": PATCH,
            "token_grid_Tp_Hp_Wp": [Tp, IMG_SIZE // PATCH, IMG_SIZE // PATCH],
            "target_tubelets_0indexed": target_tubelets,
            "context_tubelets_0indexed": list(range(0, Tp - 4)),
            "masking_scheme": "entire future tubelets (all spatial patches at "
                              "t in target_tubelets) masked; all earlier tubelets "
                              "fully visible as context. Uses VideoMAEForPreTraining's "
                              "own decoder + mask_token + position_embeddings.",
            "predictor_metric": "mean per-patch cosine(decoder_logit, true_normalized_pixel_patch) "
                                 "over masked future-tubelet tokens",
            "naive_metric": "mean per-patch cosine(true_normalized_patch[t-1], true_normalized_patch[t]) "
                             "-- copy-last-visible-tubelet baseline, same normalized-pixel space",
        },
        "per_seq": per_seq,
        "summary": summary,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'metric':<45s} {'mean':>10s} {'95% CI':>22s}")
    print("-" * 100)
    for k in ["LQ_predictor", "LQ_naive", "HQ_predictor", "HQ_naive"]:
        e = summary[k]
        print(f"{e['label']:<45s} {e['mean']:>10.5f} [{e['ci95_lo']:.4f},{e['ci95_hi']:.4f}]")
    print("-" * 100)
    for tag in ["LQ", "HQ"]:
        e = summary[f"{tag}_predictor_vs_naive"]
        print(f"{tag} predictor>naive: wilcoxon p={e['wilcoxon_p_greater']}  "
              f"cohens_d={e['cohens_d_paired']:+.4f}  delta={e['delta_mean_predictor_minus_naive']:+.5f}")
    e = summary["HQ_predictor_vs_LQ_predictor"]
    print(f"HQ predictor > LQ predictor: wilcoxon p={e['wilcoxon_p_hq_greater']}  "
          f"cohens_d={e['cohens_d_paired_hq_minus_lq']:+.4f}  delta={e['delta_mean_hq_minus_lq']:+.5f}  "
          f"HQ wins {e['n_sequences_hq_greater']}/{e['n_sequences_total']}")
    print("=" * 100)
    print(f"[SAVE] {OUT_JSON}")
    print(f"[DONE] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
