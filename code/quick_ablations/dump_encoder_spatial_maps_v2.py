"""dump_encoder_spatial_maps_v2.py

Task 2 — extend dump_encoder_5x.py by adding VideoMAE-large and a
DINOv3-temporal-agg baseline on the same 15 SDSD-indoor pairs.

Metric (per patch, per pair):
    rho_patch = ||phi_LQ - phi_HQ||_2 / (||phi_HQ||_2 + 1e-6)

Two accounting styles for per-clip aggregation (matter only for
encoders that give one feature per (t, patch) — i.e. all of them):

  (A) T-mean features first, then per-patch shift
      shift = ||mean_t phi(x_t^LQ) - mean_t phi(x_t^HQ)|| / ||mean_t phi(x_t^HQ)||
      — this is the "temporal-agg" style, matching dump_encoder_5x.py
        for image encoders and matching V-JEPA's tubelet-averaged output.

  (B) per-frame shift, then mean over T
      shift_t = ||phi(x_t^LQ) - phi(x_t^HQ)|| / ||phi(x_t^HQ)||
      shift = mean_t shift_t
      — this is the per-frame style; genuinely uses per-frame content.

For image encoders we report (A) as the primary metric (matching the
existing paper number), and add a `_perframe` variant (B) for DINOv3
labeled `dinov3_perframe` — kept just for internal reference.

The **DINOv3-temporal-agg** encoder key uses style (A) applied to
per-frame DINOv3 outputs — this is functionally identical to the
existing `dinov3` column but named explicitly so the reader sees the
axis being isolated (video-pretraining vs. temporal-token-fusion).

Per-pair scalar: mean of the per-patch rho over the (h, w) grid.

Outputs:
  /home/ma-user/workspace/zihaohe/quick_ablations/invariance_v2.json
  /tmp/invariance_v2.npz                              (per-patch shift maps)
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
CLIP_ID    = "openai/clip-vit-large-patch14"
DINO_ID    = "facebook/dinov2-large"
DINO3_ID   = "vit_large_patch16_dinov3.lvd1689m"
SIGLIP2_ID = "google/siglip2-large-patch16-256"
VMAE_ID    = "MCG-NJU/videomae-large"
CACHE_DIR  = "/cache/zihaohe/hf_cache"

OUT_JSON = "/home/ma-user/workspace/zihaohe/quick_ablations/invariance_v2.json"
OUT_NPZ  = "/tmp/invariance_v2.npz"

SEQ_NAMES = [
    "pair50","pair52","pair58","pair5_2","pair6","pair60","pair68","pair6_2",
    "pair7","pair70","pair7_2","pair8","pair8_2","pair9","pair9_2",
]

CLIP_LEN = 16
CROP     = 512
TUBELET  = 2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN     = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD      = torch.tensor([0.26862954, 0.26130258, 0.27577711])
SIGLIP_MEAN   = torch.tensor([0.5, 0.5, 0.5])
SIGLIP_STD    = torch.tensor([0.5, 0.5, 0.5])


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
        y0 = (H - CROP) // 2; x0 = (W - CROP) // 2
        arrs.append(a[y0:y0+CROP, x0:x0+CROP, :])
    return np.stack(arrs, axis=0)   # (T, 512, 512, 3) uint8


# ================================================================= preproc
def _to_bchw01(frames_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


def _resize_norm(x, size, mean, std):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = mean.to(x.device).view(1, 3, 1, 1); s = std.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


# ================================================================= V-JEPA
def build_vjepa():
    return VJepaBackboneLocal21(
        local_repo=VJEPA_REPO, local_ckpt=VJEPA_CKPT,
        feature_layers=(23,), image_size=CROP, frames_per_clip=CLIP_LEN,
        patch_size=16, tubelet_size=TUBELET, freeze=True,
    ).to(DEVICE).eval()


@torch.no_grad()
def vjepa_tavg_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) -> (Hp, Wp, D) T-averaged patch features."""
    x = torch.from_numpy(frames_uint8).to(DEVICE).float().div_(255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    out = model(x)
    feats = out["feats"][0]
    Tp, Hp, Wp = out["token_grid"]; D = feats.shape[-1]
    f = feats.view(1, Tp, Hp, Wp, D).mean(dim=1).squeeze(0)   # (Hp, Wp, D)
    return f.detach().float().cpu().numpy()


# ================================================================= image encoders
@torch.no_grad()
def clip_tavg_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames_uint8); x = _resize_norm(x, 224, CLIP_MEAN, CLIP_STD)
    tk = model(pixel_values=x).last_hidden_state[:, 1:, :]
    T, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(T, g, g, D).mean(dim=0).detach().float().cpu().numpy()


@torch.no_grad()
def dinov2_tavg_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames_uint8); x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    tk = model(pixel_values=x).last_hidden_state[:, 1:, :]
    T, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(T, g, g, D).mean(dim=0).detach().float().cpu().numpy()


@torch.no_grad()
def dinov3_tavg_feats(model, frames_uint8: np.ndarray, n_prefix: int) -> np.ndarray:
    x = _to_bchw01(frames_uint8); x = _resize_norm(x, 256, IMAGENET_MEAN, IMAGENET_STD)
    tk = model.forward_features(x)[:, n_prefix:, :]
    T, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(T, g, g, D).mean(dim=0).detach().float().cpu().numpy()


@torch.no_grad()
def dinov3_per_frame_feats(model, frames_uint8: np.ndarray, n_prefix: int) -> np.ndarray:
    """Return (T, g, g, D) — per-frame features, no time aggregation."""
    x = _to_bchw01(frames_uint8); x = _resize_norm(x, 256, IMAGENET_MEAN, IMAGENET_STD)
    tk = model.forward_features(x)[:, n_prefix:, :]
    T, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(T, g, g, D).detach().float().cpu().numpy()


@torch.no_grad()
def siglip_tavg_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    x = _to_bchw01(frames_uint8); x = _resize_norm(x, 256, SIGLIP_MEAN, SIGLIP_STD)
    tk = model(pixel_values=x).last_hidden_state
    T, N, D = tk.shape; g = int(round(N ** 0.5))
    return tk.view(T, g, g, D).mean(dim=0).detach().float().cpu().numpy()


# ================================================================= VideoMAE
@torch.no_grad()
def videomae_tavg_feats(model, frames_uint8: np.ndarray) -> np.ndarray:
    """VideoMAE-large @ 224, 16 frames -> (14, 14, 1024) after Tp-avg.
    Output: (Hp, Wp, D) with Hp=Wp=14."""
    x = _to_bchw01(frames_uint8)                          # (16, 3, H, W)
    x = _resize_norm(x, 224, IMAGENET_MEAN, IMAGENET_STD)
    x = x.unsqueeze(0)                                    # (1, 16, 3, 224, 224)
    tk = model(pixel_values=x).last_hidden_state          # (1, 1568, 1024)
    B, N, D = tk.shape
    Tp = CLIP_LEN // TUBELET                              # 8
    N_spatial = N // Tp; g = int(round(N_spatial ** 0.5))
    assert g * g * Tp == N, f"unexpected VideoMAE token count {N}"
    f = tk.view(B, Tp, g, g, D).mean(dim=1).squeeze(0)    # (g, g, D)
    return f.detach().float().cpu().numpy()


# ================================================================= metric
def per_patch_shift(lq_feat: np.ndarray, hq_feat: np.ndarray) -> np.ndarray:
    """(h,w,D) -> (h,w) relative L2 shift."""
    lq = lq_feat.astype(np.float64); hq = hq_feat.astype(np.float64)
    diff = lq - hq
    return (np.linalg.norm(diff, axis=-1) /
            (np.linalg.norm(hq, axis=-1) + 1e-6)).astype(np.float32)


def per_frame_shift_then_avg(lq_per_frame: np.ndarray,
                             hq_per_frame: np.ndarray) -> np.ndarray:
    """(T,h,w,D), (T,h,w,D) -> (h,w) : per-frame relative L2 shift, then avg T."""
    lq = lq_per_frame.astype(np.float64); hq = hq_per_frame.astype(np.float64)
    diff = lq - hq
    d = np.linalg.norm(diff, axis=-1)                     # (T,h,w)
    hn = np.linalg.norm(hq, axis=-1) + 1e-6                # (T,h,w)
    return (d / hn).mean(axis=0).astype(np.float32)


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
    from transformers import CLIPVisionModel, Dinov2Model, SiglipVisionModel
    from scipy.stats import wilcoxon
    import timm

    t0 = time.time()

    grids = {"vjepa": 32, "dinov3": 16, "dinov2": 16, "siglip": 16,
             "clip": 16, "videomae": 14, "dinov3_tagg": 16,
             "dinov3_perframe": 16}
    n_pairs = len(SEQ_NAMES)

    shift_maps = {k: np.zeros((n_pairs, g, g), dtype=np.float32)
                  for k, g in grids.items()}
    per_pair_scalar = {k: np.zeros(n_pairs, dtype=np.float64) for k in grids}

    # Load all frames once
    print(f"[*] loading frames for {n_pairs} SDSD-indoor pairs ...", flush=True)
    lq_frames, hq_frames = {}, {}
    for seq in SEQ_NAMES:
        lq_frames[seq] = load_pair_frames(seq, "input")
        hq_frames[seq] = load_pair_frames(seq, "GT")
    print(f"    {time.time()-t0:.1f}s", flush=True)

    def run_style_A(enc_key, feat_fn, model):
        print(f"[*] encoder={enc_key} (T-mean features first)", flush=True)
        for i, seq in enumerate(SEQ_NAMES):
            lq = feat_fn(model, lq_frames[seq])
            hq = feat_fn(model, hq_frames[seq])
            s = per_patch_shift(lq, hq)
            shift_maps[enc_key][i] = s
            per_pair_scalar[enc_key][i] = float(s.mean())
            print(f"    {seq:>9s}  mean_shift={s.mean():.4f}", flush=True)

    # ---- V-JEPA ----
    print("[*] loading V-JEPA ...", flush=True)
    vjepa = build_vjepa()
    run_style_A("vjepa", vjepa_tavg_feats, vjepa)
    del vjepa; torch.cuda.empty_cache()

    # ---- DINOv3 (image + tagg + per-frame all from same forward passes) ----
    print("[*] loading DINOv3 ViT-L/16 (timm) ...", flush=True)
    dino3 = timm.create_model(DINO3_ID, pretrained=True, num_classes=0).to(DEVICE).eval()
    n_prefix = int(getattr(dino3, "num_prefix_tokens", 5))
    for i, seq in enumerate(SEQ_NAMES):
        lq_pf = dinov3_per_frame_feats(dino3, lq_frames[seq], n_prefix)   # (T,g,g,D)
        hq_pf = dinov3_per_frame_feats(dino3, hq_frames[seq], n_prefix)
        # style A = T-mean features first
        lq_tavg = lq_pf.mean(axis=0); hq_tavg = hq_pf.mean(axis=0)
        s_A = per_patch_shift(lq_tavg, hq_tavg)
        shift_maps["dinov3"][i]      = s_A
        shift_maps["dinov3_tagg"][i] = s_A
        per_pair_scalar["dinov3"][i]      = float(s_A.mean())
        per_pair_scalar["dinov3_tagg"][i] = float(s_A.mean())
        # style B = per-frame shift then mean
        s_B = per_frame_shift_then_avg(lq_pf, hq_pf)
        shift_maps["dinov3_perframe"][i]     = s_B
        per_pair_scalar["dinov3_perframe"][i] = float(s_B.mean())
        print(f"    {seq:>9s}  A={s_A.mean():.4f}  B(perframe)={s_B.mean():.4f}",
              flush=True)
    del dino3; torch.cuda.empty_cache()

    # ---- DINOv2 ----
    print("[*] loading DINOv2 ...", flush=True)
    dino = Dinov2Model.from_pretrained(DINO_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    run_style_A("dinov2", dinov2_tavg_feats, dino); del dino; torch.cuda.empty_cache()

    # ---- CLIP ----
    print("[*] loading CLIP ...", flush=True)
    clip = CLIPVisionModel.from_pretrained(CLIP_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    run_style_A("clip", clip_tavg_feats, clip); del clip; torch.cuda.empty_cache()

    # ---- SigLIP ----
    print("[*] loading SigLIP 2 ...", flush=True)
    siglip = SiglipVisionModel.from_pretrained(SIGLIP2_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
    run_style_A("siglip", siglip_tavg_feats, siglip); del siglip; torch.cuda.empty_cache()

    # ---- VideoMAE ----
    videomae_failed = None
    try:
        from transformers import VideoMAEModel
        print("[*] loading VideoMAE-large ...", flush=True)
        vm = VideoMAEModel.from_pretrained(VMAE_ID, cache_dir=CACHE_DIR).to(DEVICE).eval()
        run_style_A("videomae", videomae_tavg_feats, vm)
        del vm; torch.cuda.empty_cache()
    except Exception as e:
        videomae_failed = f"{type(e).__name__}: {e}"
        print(f"[WARN] VideoMAE failed: {videomae_failed}", flush=True)
        per_pair_scalar["videomae"] = None

    # ============================================================= summary
    encoder_order = ["vjepa", "dinov3", "dinov3_tagg", "dinov3_perframe",
                     "dinov2", "siglip", "clip", "videomae"]
    labels = {
        "vjepa":           "V-JEPA-2.1",
        "dinov3":          "DINOv3",
        "dinov3_tagg":     "DINOv3-temporal-agg",
        "dinov3_perframe": "DINOv3-per-frame",
        "dinov2":          "DINOv2",
        "siglip":          "SigLIP2",
        "clip":            "CLIP",
        "videomae":        "VideoMAE-L",
    }

    per_seq_out = {}
    for i, seq in enumerate(SEQ_NAMES):
        per_seq_out[seq] = {}
        for k in encoder_order:
            v = per_pair_scalar[k]
            per_seq_out[seq][k] = float(v[i]) if v is not None else None

    v_vec = per_pair_scalar["vjepa"]

    summary = {}
    for k in encoder_order:
        label = labels[k]
        vals = per_pair_scalar[k]
        if vals is None:
            summary[k] = {"label": label, "status": "missing",
                          "reason": videomae_failed if k == "videomae" else "N/A"}
            continue
        vals = np.asarray(vals, dtype=np.float64)
        m = float(vals.mean()); s = float(vals.std(ddof=1))
        lo, hi = bootstrap_ci_mean(vals)
        entry = {
            "label": label,
            "mean":  m,
            "std":   s,
            "ci95_lo": lo, "ci95_hi": hi,
            "n": int(len(vals)),
            "per_pair_values": vals.tolist(),
        }
        if k != "vjepa":
            # Lower shift = MORE invariant. So V-JEPA "beats" competitor iff
            # V-JEPA has LOWER shift. One-sided Wilcoxon H1: V-JEPA < competitor
            try:
                wres = wilcoxon(v_vec, vals, alternative="less",
                                zero_method="pratt")
                entry["wilcoxon_p_vs_vjepa_less"] = float(wres.pvalue)
                entry["wilcoxon_stat"]            = float(wres.statistic)
            except ValueError as e:
                entry["wilcoxon_p_vs_vjepa_less"] = None
                entry["wilcoxon_error"]           = str(e)
            entry["cohens_d_paired_vjepa_minus"] = paired_cohens_d(v_vec, vals)
            entry["delta_mean_vjepa_minus"]      = float((v_vec - vals).mean())
            dlo, dhi = bootstrap_ci_delta(v_vec.tolist(), vals.tolist())
            entry["delta_ci95_lo"] = dlo
            entry["delta_ci95_hi"] = dhi
        summary[k] = entry

    out = {
        "meta": {
            "task": "single-frame invariance (rho_L2 relative per-patch shift)",
            "metric": "rho_patch = ||phi_LQ - phi_HQ||_2 / ||phi_HQ||_2, "
                      "mean over (h,w) grid per pair",
            "aggregation_style_default": "A_Tmean_features_then_shift",
            "n_pairs": len(SEQ_NAMES),
            "pairs": SEQ_NAMES,
            "encoders": encoder_order,
            "crop": CROP, "clip_len": CLIP_LEN,
            "note": ("dinov3_tagg is style-A on DINOv3 per-frame features "
                     "(= T-mean features first, then shift). "
                     "dinov3_perframe is style-B (per-frame shift, then mean). "
                     "The two share the same underlying DINOv3 forward passes; "
                     "'dinov3' == 'dinov3_tagg' by construction and is kept "
                     "as the canonical image-encoder number."),
        },
        "per_seq": per_seq_out,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=2)

    # Save all shift maps for downstream viz
    save_kwargs = {"seq_names": np.array(SEQ_NAMES)}
    for k, arr in shift_maps.items():
        if k == "videomae" and per_pair_scalar["videomae"] is None:
            continue
        save_kwargs[f"{k}_shift"] = arr
    np.savez(OUT_NPZ, **save_kwargs)

    # ---- print compact table ----
    print("\n" + "=" * 96)
    print(f"{'encoder':<22s} {'mean_rho':>9s} {'std':>7s} {'95% CI':>18s}"
          f" {'delta_v':>10s} {'p_wilc(V<)':>11s} {'cohen d':>9s}")
    print("-" * 96)
    for k in encoder_order:
        e = summary[k]
        if e.get("status") == "missing":
            print(f"{labels[k]:<22s} MISSING ({e.get('reason','')})")
            continue
        ci = f"[{e['ci95_lo']:.4f},{e['ci95_hi']:.4f}]"
        if k == "vjepa":
            print(f"{labels[k]:<22s} {e['mean']:>9.5f} {e['std']:>7.4f} {ci:>18s}   ---")
        else:
            print(f"{labels[k]:<22s} {e['mean']:>9.5f} {e['std']:>7.4f} {ci:>18s}"
                  f" {e['delta_mean_vjepa_minus']:>+10.5f}"
                  f" {e.get('wilcoxon_p_vs_vjepa_less', float('nan')):>11.4g}"
                  f" {e['cohens_d_paired_vjepa_minus']:>+9.4f}")
    print("=" * 96)
    print(f"[SAVE] {OUT_JSON}")
    print(f"[SAVE] {OUT_NPZ}")
    print(f"[DONE] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
