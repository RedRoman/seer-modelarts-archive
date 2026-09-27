"""vjepa2_base_predictive_coherence.py

Genuine self-supervised predictive-coherence test using the ORIGINAL (non-2.1,
non-distillation) V-JEPA2 ViT-L checkpoint. Unlike the "2.1" checkpoint used
elsewhere in this codebase (vjepa2_1_vitl_dist_vitG_384.pt, whose predictor
distills into a separate 1664-dim ViT-Gigantic teacher space), the base
V-JEPA2 predictor's out_embed_dim defaults to its OWN encoder's embed_dim
(1024 for ViT-L) -- see src/hub/backbones.py::_make_vjepa2_model, which passes
no `teacher_embed_dim` kwarg to the predictor builder. This makes genuine
self-prediction (predict tubelet t's own encoder features from earlier
tubelets) possible and faithful to V-JEPA's actual pretraining objective.

Checkpoint: facebookresearch/vjepa2 vit_large, downloaded directly from
  https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
via src.hub.backbones.vjepa2_vit_large (encoder + predictor, both loaded from
the same state dict, keys "target_encoder" and "predictor").

Data pipeline (SAME 11 sequences, same crop/loading conventions) reused from
temporal_scan_v3.py: list_bvi, list_sdsd, burst_indices, build_pair_clip,
load_frames, _load_bvi, _load_sdsd, center_crop.

Protocol per sequence:
  1. Build a 16-frame clip (tubelet_size=2 -> Tp=8 temporal tokens), run the
     frozen V-JEPA2-base encoder -> full token grid (Tp=8, Hp, Wp, D=1024).
  2. For each target tubelet t in {4,5,6,7} (needs >=4 tubelets of context):
       masks_x = flat indices of tubelets [0..t-1]           (context)
       masks_y = flat indices of tubelet t                    (target)
       pred = predictor(context_tokens, masks_x, masks_y)     # (1, Hp*Wp, 1024)
       predictor_score(t) = mean_i cos(pred_i, true_i)  where true = encoder
                              features actually at tubelet t (same space).
       naive_score(t)     = mean_i cos(true_{t-1,i}, true_{t,i})
  3. Average across the 4 target tubelets -> one predictor-score and one
     naive-score per sequence.
  4. DINOv3 naive-identity content-cosine is NOT recomputed -- reused
     verbatim from temporal_artifact_diagnosis.json's content_cosine_metric
     ("full", per-sequence values), since frame/crop/resolution protocol for
     that number is unaffected by which V-JEPA checkpoint we test here. It is
     reported purely as a same-axis reference (DINOv3 has no predictor).

Output: /home/ma-user/workspace/zihaohe/quick_ablations/vjepa2_base_predictive_coherence.json
"""
from __future__ import annotations
import os, sys, json, time, traceback
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/cache/zihaohe/hf_cache")

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/quick_ablations")
from temporal_scan_v3 import (  # noqa
    BVI_ROOT, SDSD_ROOT, CROP, N_SAMPLE, CLIP_LEN, TUBELET,
    center_crop, _load_bvi, _load_sdsd, list_bvi, list_sdsd, burst_indices,
    build_pair_clip, load_frames, bootstrap_ci_mean, bootstrap_ci_delta,
    paired_cohens_d, DEVICE,
)

VJEPA2_REPO = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VITL_CKPT_LOCAL = "/cache/zihaohe/checkpoint/vitl.pt"
ARTIFACT_DIAG_JSON = "/home/ma-user/workspace/zihaohe/quick_ablations/temporal_artifact_diagnosis.json"
OUT_JSON = "/home/ma-user/workspace/zihaohe/quick_ablations/vjepa2_base_predictive_coherence.json"

IMG_SIZE = 256   # native config for vjepa2 vit_large (pretrain-256px-16f.yaml)

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


# ============================================================= model
def build_vjepa2_base():
    """Load base (non-distillation) V-JEPA2 ViT-L encoder + predictor.
    Downloads/loads from the local vitl.pt checkpoint (same file layout as
    dl.fbaipublicfiles.com/vjepa2/vitl.pt: {"target_encoder":..., "predictor":...})."""
    if VJEPA2_REPO not in sys.path:
        sys.path.insert(0, VJEPA2_REPO)
    from src.hub.backbones import _make_vjepa2_model, _clean_backbone_key

    encoder, predictor = _make_vjepa2_model(
        model_name="vit_large",
        checkpoint_key="target_encoder",
        img_size=IMG_SIZE,
        patch_size=16,
        tubelet_size=TUBELET,
        num_frames=CLIP_LEN,
        predictor_embed_dim=384,
        predictor_out_embed_dim=None,   # -> defaults to embed_dim (1024), NOT a teacher dim
        pretrained=False,               # we load weights ourselves from local file
    )

    sd = torch.load(VITL_CKPT_LOCAL, map_location="cpu")
    enc_sd = _clean_backbone_key(dict(sd["target_encoder"]))
    pred_sd = _clean_backbone_key(dict(sd["predictor"]))
    enc_missing = encoder.load_state_dict(enc_sd, strict=False)
    pred_missing = predictor.load_state_dict(pred_sd, strict=False)

    encoder = encoder.to(DEVICE).eval()
    predictor = predictor.to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)

    return encoder, predictor, enc_missing, pred_missing, list(sd.keys())


@torch.no_grad()
def encode_clip(encoder, clip16: np.ndarray):
    """clip16: (16,H,W,3) uint8 -> features (Tp,Hp,Wp,D) float32 + grid dims."""
    x = torch.from_numpy(clip16).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1,3,T,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1, 1)
    if x.shape[-1] != IMG_SIZE or x.shape[-2] != IMG_SIZE:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        x = x.reshape(B, T, C, IMG_SIZE, IMG_SIZE).permute(0, 2, 1, 3, 4).contiguous()
    x = (x - mean) / std

    out = encoder(x)   # (1, N, D) -- no out_layers set, so single tensor after final norm
    B, N, D = out.shape
    Tp = CLIP_LEN // TUBELET
    Hp = IMG_SIZE // 16
    Wp = IMG_SIZE // 16
    assert Tp * Hp * Wp == N, f"grid mismatch: Tp*Hp*Wp={Tp*Hp*Wp} != N={N}"
    feats = out.view(1, Tp, Hp, Wp, D)
    return feats, (Tp, Hp, Wp, D)


def tubelet_flat_indices(t, Hp, Wp):
    """Flat token indices belonging to temporal tubelet index t, matching
    PatchEmbed3D's Conv3d(...).flatten(2) row-major (T,H,W) order."""
    base = t * Hp * Wp
    return list(range(base, base + Hp * Wp))


@torch.no_grad()
def predictor_score_for_clip(encoder, predictor, clip16: np.ndarray, target_tubelets):
    feats, (Tp, Hp, Wp, D) = encode_clip(encoder, clip16)   # (1,Tp,Hp,Wp,D)
    flat = feats.view(1, Tp * Hp * Wp, D)                    # (1, N, D)

    per_t_pred = {}
    per_t_naive = {}
    for t in target_tubelets:
        assert t >= 1, "need at least one prior tubelet for naive baseline"
        ctx_idx = []
        for tt in range(t):
            ctx_idx.extend(tubelet_flat_indices(tt, Hp, Wp))
        tgt_idx = tubelet_flat_indices(t, Hp, Wp)
        prev_idx = tubelet_flat_indices(t - 1, Hp, Wp)

        masks_x = [torch.tensor([ctx_idx], dtype=torch.int64, device=DEVICE)]
        masks_y = [torch.tensor([tgt_idx], dtype=torch.int64, device=DEVICE)]

        ctx_tokens = torch.gather(
            flat, 1, torch.tensor(ctx_idx, device=DEVICE).view(1, -1, 1).expand(-1, -1, D)
        )  # (1, N_ctx, D)

        pred = predictor(ctx_tokens, masks_x, masks_y)   # (1, len(tgt_idx), D)

        true_tgt = flat[:, tgt_idx, :]     # (1, Hp*Wp, D)
        true_prev = flat[:, prev_idx, :]   # (1, Hp*Wp, D)

        cos_pred = F.cosine_similarity(pred, true_tgt, dim=-1)     # (1, Hp*Wp)
        cos_naive = F.cosine_similarity(true_prev, true_tgt, dim=-1)

        per_t_pred[t] = float(cos_pred.mean().item())
        per_t_naive[t] = float(cos_naive.mean().item())

    return per_t_pred, per_t_naive, (Tp, Hp, Wp, D)


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
        clip16 = load_frames(files, pair, loader)   # (16, CROP, CROP, 3)
        prepared.append(dict(dataset=dset, name=name, T=T, idx=idx,
                              delta=delta, clip16=clip16))
        print(f"[LOAD] {name:22s} T={T} idx={idx} delta={delta}", flush=True)
    return prepared


# ============================================================= main
def main():
    from scipy.stats import wilcoxon
    t_all = time.time()

    print("[*] Building V-JEPA2 (base) ViT-L encoder+predictor ...", flush=True)
    t0 = time.time()
    encoder, predictor, enc_missing, pred_missing, ckpt_keys = build_vjepa2_base()
    print(f"    loaded +{time.time()-t0:.1f}s", flush=True)
    print(f"    checkpoint top-level keys: {ckpt_keys}", flush=True)
    print(f"    encoder load_state_dict missing_keys={len(enc_missing.missing_keys)} "
          f"unexpected_keys={len(enc_missing.unexpected_keys)}", flush=True)
    print(f"    predictor load_state_dict missing_keys={len(pred_missing.missing_keys)} "
          f"unexpected_keys={len(pred_missing.unexpected_keys)}", flush=True)

    # ---- GATE: confirm predictor output dim genuinely matches encoder embed_dim ----
    proj = predictor.predictor_proj
    out_dim = proj.weight.shape[0]
    in_dim_predictor_embed = predictor.predictor_embed.in_features
    encoder_embed_dim = encoder.embed_dim
    print(f"[GATE] predictor.predictor_proj.weight.shape = {tuple(proj.weight.shape)}", flush=True)
    print(f"[GATE] predictor.predictor_embed.in_features (encoder->predictor in) = {in_dim_predictor_embed}", flush=True)
    print(f"[GATE] encoder.embed_dim = {encoder_embed_dim}", flush=True)

    gate_pass = (out_dim == encoder_embed_dim == 1024)
    if not gate_pass:
        print(f"[GATE FAIL] predictor out_dim={out_dim} != encoder_embed_dim={encoder_embed_dim} "
              f"(or not 1024). STOPPING per task instructions.", flush=True)
        out = {
            "meta": {
                "gate_passed": False,
                "predictor_proj_out_dim": int(out_dim),
                "encoder_embed_dim": int(encoder_embed_dim),
                "checkpoint": VITL_CKPT_LOCAL,
            }
        }
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as fp:
            json.dump(out, fp, indent=2)
        return 1

    print("[GATE PASS] predictor output dim matches own encoder embed_dim (1024). Proceeding.", flush=True)

    # ---- data ----
    prepared = prepare_sequences(CANDIDATES)
    n_seq = len(prepared)
    print(f"\n[*] {n_seq} sequences prepared\n", flush=True)

    Tp_expected = CLIP_LEN // TUBELET  # 8
    target_tubelets = list(range(Tp_expected - 4, Tp_expected))  # e.g. [4,5,6,7]
    print(f"[*] target tubelets (0-indexed): {target_tubelets} (Tp={Tp_expected})", flush=True)

    # ---- run per sequence ----
    per_seq_out = {}
    pred_scores = []
    naive_scores = []
    grid_info = None
    for p in prepared:
        per_t_pred, per_t_naive, grid = predictor_score_for_clip(
            encoder, predictor, p["clip16"], target_tubelets
        )
        grid_info = grid
        seq_pred = float(np.mean(list(per_t_pred.values())))
        seq_naive = float(np.mean(list(per_t_naive.values())))
        pred_scores.append(seq_pred)
        naive_scores.append(seq_naive)
        per_seq_out[p["name"]] = {
            "dataset": p["dataset"], "T": int(p["T"]), "idx": p["idx"],
            "delta": int(p["delta"]),
            "predictor_score": seq_pred,
            "naive_baseline_score": seq_naive,
            "per_target_tubelet_predictor": per_t_pred,
            "per_target_tubelet_naive": per_t_naive,
        }
        print(f"    {p['name']:22s} predictor={seq_pred:.5f}  naive={seq_naive:.5f}  "
              f"delta={seq_pred-seq_naive:+.5f}", flush=True)

    del encoder, predictor
    torch.cuda.empty_cache()

    # ---- DINOv3 naive-identity reference (reused from prior diagnosis) ----
    dinov3_naive = {}
    dinov3_note = None
    try:
        with open(ARTIFACT_DIAG_JSON) as fp:
            diag = json.load(fp)
        dinov3_naive = diag["content_cosine_metric"]["full"]["dinov3"]["per_seq_values"]
        dinov3_note = ("Reused verbatim from temporal_artifact_diagnosis.json "
                        "content_cosine_metric.full.dinov3 (adjacent-frame content "
                        "cosine, same 11 sequences, same crop/loading protocol). "
                        "DINOv3 has no predictor; this is a same-axis reference only, "
                        "not a predictor-vs-predictor comparison.")
    except Exception as e:
        dinov3_note = f"Failed to load reference: {type(e).__name__}: {e}"

    dinov3_vals = [dinov3_naive.get(p["name"]) for p in prepared]
    dinov3_available = all(v is not None for v in dinov3_vals)

    # ============================================================= stats
    pred_arr = np.array(pred_scores, dtype=np.float64)
    naive_arr = np.array(naive_scores, dtype=np.float64)

    pred_mean, pred_std = float(pred_arr.mean()), float(pred_arr.std(ddof=1))
    naive_mean, naive_std = float(naive_arr.mean()), float(naive_arr.std(ddof=1))
    pred_lo, pred_hi = bootstrap_ci_mean(pred_arr)
    naive_lo, naive_hi = bootstrap_ci_mean(naive_arr)

    try:
        wres = wilcoxon(pred_arr, naive_arr, alternative="greater", zero_method="pratt")
        wilcoxon_p = float(wres.pvalue)
        wilcoxon_stat = float(wres.statistic)
    except ValueError as e:
        wilcoxon_p = None
        wilcoxon_stat = None
        wilcoxon_error = str(e)
    else:
        wilcoxon_error = None

    cohens_d = paired_cohens_d(pred_arr, naive_arr)
    delta_mean = float((pred_arr - naive_arr).mean())
    delta_lo, delta_hi = bootstrap_ci_delta(pred_arr.tolist(), naive_arr.tolist())

    summary = {
        "vjepa2_base_predictor": {
            "label": "V-JEPA2 (base, genuine predictor)",
            "mean": pred_mean, "std": pred_std,
            "ci95_lo": pred_lo, "ci95_hi": pred_hi,
            "n": n_seq, "per_seq_values": dict(zip([p["name"] for p in prepared], pred_scores)),
        },
        "vjepa2_base_naive": {
            "label": "V-JEPA2 (base) naive-identity baseline",
            "mean": naive_mean, "std": naive_std,
            "ci95_lo": naive_lo, "ci95_hi": naive_hi,
            "n": n_seq, "per_seq_values": dict(zip([p["name"] for p in prepared], naive_scores)),
        },
        "predictor_vs_naive": {
            "wilcoxon_p_greater": wilcoxon_p,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_error": wilcoxon_error,
            "cohens_d_paired": cohens_d,
            "delta_mean_predictor_minus_naive": delta_mean,
            "delta_ci95_lo": delta_lo,
            "delta_ci95_hi": delta_hi,
        },
        "dinov3_naive_reference": {
            "label": "DINOv3 adjacent-frame content-cosine (reference only, no predictor)",
            "available": dinov3_available,
            "note": dinov3_note,
            "mean": float(np.mean(dinov3_vals)) if dinov3_available else None,
            "std": float(np.std(dinov3_vals, ddof=1)) if dinov3_available else None,
            "ci95": list(bootstrap_ci_mean(np.array(dinov3_vals, dtype=np.float64))) if dinov3_available else None,
            "n": n_seq if dinov3_available else 0,
            "per_seq_values": dinov3_naive if dinov3_available else {},
        },
    }

    out = {
        "meta": {
            "gate_passed": True,
            "checkpoint": VITL_CKPT_LOCAL,
            "checkpoint_source_url": "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
            "checkpoint_top_level_keys": ckpt_keys,
            "predictor_proj_out_dim": int(out_dim),
            "encoder_embed_dim": int(encoder_embed_dim),
            "encoder_missing_keys_count": len(enc_missing.missing_keys),
            "encoder_unexpected_keys_count": len(enc_missing.unexpected_keys),
            "predictor_missing_keys_count": len(pred_missing.missing_keys),
            "predictor_unexpected_keys_count": len(pred_missing.unexpected_keys),
            "vjepa2_repo": VJEPA2_REPO,
            "model_name": "vit_large",
            "img_size": IMG_SIZE,
            "clip_len_frames": CLIP_LEN,
            "tubelet_size": TUBELET,
            "patch_size": 16,
            "token_grid_Tp_Hp_Wp_D": list(grid_info) if grid_info else None,
            "target_tubelets_0indexed": target_tubelets,
            "n_seqs": n_seq,
            "sequences": [p["name"] for p in prepared],
            "metric_predictor": "mean per-patch cosine(predicted, true) at target tubelet, "
                                 "predictor sees ALL earlier tubelets as context "
                                 "(masks_x=flat idx of tubelets [0..t-1], masks_y=flat idx of tubelet t)",
            "metric_naive": "mean per-patch cosine(true[t-1], true[t]) -- no predictor, "
                             "adjacent-tubelet identity baseline",
            "dinov3_reference_source": ARTIFACT_DIAG_JSON,
        },
        "per_seq": per_seq_out,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=2)

    print("\n" + "=" * 90)
    print(f"{'metric':<40s} {'mean':>10s} {'95% CI':>22s}")
    print("-" * 90)
    print(f"{'V-JEPA2(base) predictor score':<40s} {pred_mean:>10.5f} "
          f"[{pred_lo:.4f},{pred_hi:.4f}]")
    print(f"{'V-JEPA2(base) naive baseline':<40s} {naive_mean:>10.5f} "
          f"[{naive_lo:.4f},{naive_hi:.4f}]")
    if dinov3_available:
        d3m = float(np.mean(dinov3_vals))
        d3lo, d3hi = bootstrap_ci_mean(np.array(dinov3_vals, dtype=np.float64))
        print(f"{'DINOv3 naive baseline (reference)':<40s} {d3m:>10.5f} "
              f"[{d3lo:.4f},{d3hi:.4f}]")
    print("-" * 90)
    print(f"Wilcoxon predictor>naive: p={wilcoxon_p}  stat={wilcoxon_stat}  "
          f"cohens_d={cohens_d:+.4f}  delta_mean={delta_mean:+.5f}")
    print("=" * 90)
    print(f"[SAVE] {OUT_JSON}")
    print(f"[DONE] {time.time()-t_all:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
