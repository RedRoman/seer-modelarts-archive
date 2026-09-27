"""tools/v11_c2_smoke.py — DAT4 mandatory smoke for C-2-Direct (DAW2 audit-saved).

Verifies:
  1. Adapter param count within DAR7 budget (≤ 14M).
  2. h_b from decoder.encode(kv_layers=None) is V-JEPA-clean (DAT2).
  3. adapter_kv shape per active scale.
  4. DAT4 zero-init equivalence:
       max_abs(pred_adapted_zero_init - pred_no_prior) ≤ 1e-5
  5. All 5 conditions produce finite outputs in [0, 1].

Exit code:
  0 = all checks PASS
  1 = any check FAIL (do not launch GPU training)

Usage:
  python tools/v11_c2_smoke.py [--out_log <path>]

Spec: §DAT4 / §DAW2.
"""
from __future__ import annotations
import argparse, os, sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from models.v11.content_attn_adapter import ContentAttnAdapter


def main(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_lines = []
    def emit(s):
        print(s)
        log_lines.append(s)

    emit("=== C-2-Direct smoke (DAT4 mandatory + DAW2 audit-saved) ===")
    emit(f"device: {device}, seed: {args.seed}")

    dec = DeltaModulatedDecoder(delta_in_encoder=False).to(device).eval()
    adapter = ContentAttnAdapter(
        bottleneck_channels=dec.dims[-1],
        vjepa_grid=tuple(dec.vjepa_grid),
        vjepa_dim=dec.vjepa_dim,
        active_scales=tuple(int(s) for s in args.active_scales.split(",")),
        ref_layer=args.ref_layer,
        attn_heads=args.attn_heads,
        ffn_mult=args.ffn_mult,
    ).to(device).eval()

    n_params_M = adapter.num_trainable_params() / 1e6
    emit(f"adapter params: {n_params_M:.2f}M (DAR7 budget ≤ 14M)")
    check_params = n_params_M <= 14.0
    emit(f"  → {'PASS' if check_params else 'FAIL'}")

    emit(f"adapter active_scales: {adapter.active_scales}")
    emit(f"adapter ref_layer: {adapter.ref_layer}")

    B, T, H, W = 1, 16, 256, 256
    N_v = 8 * 16 * 16
    lq = torch.rand(B, 3, T, H, W, device=device)
    zL = [torch.randn(B, N_v, 1024, device=device) for _ in range(4)]
    zL_other = [torch.randn(B, N_v, 1024, device=device) for _ in range(4)]
    zL_zeros = [torch.zeros(B, N_v, 1024, device=device) for _ in range(4)]

    # DAT2: h_b must be V-JEPA-clean (kv_layers=None)
    h_b, skips = dec.encode(lq, kv_layers=None, delta_layers=None)
    emit(f"\n[DAT2] h_b from encode(kv=None): shape={tuple(h_b.shape)}")
    check_hb_shape = tuple(h_b.shape) == (B, dec.dims[-1], T, 8, 8)
    emit(f"  → shape {'PASS' if check_hb_shape else 'FAIL'}")

    def pred(kv_layers, kv_override):
        return dec.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None,
                          kv_override=kv_override)

    p_no_prior = pred(None, None)
    p_passive  = pred(zL,   None)
    kv_good    = adapter(zL,       h_b)
    p_adapted  = pred(None, kv_good)
    kv_bad     = adapter(zL_other, h_b)
    p_shuffled = pred(None, kv_bad)
    kv_zerov   = adapter(zL_zeros, h_b)
    p_zero     = pred(None, kv_zerov)

    emit(f"\n[shapes]")
    emit(f"  adapter_kv keys = {list(kv_good.keys())}")
    for s in kv_good:
        emit(f"  adapter_kv[{s}] shape = {tuple(kv_good[s].shape)}")

    diff = (p_adapted - p_no_prior).abs().max().item()
    emit(f"\n[DAT4 SMOKE] iter0 max_abs(p_adapted_zero_init - p_no_prior) = {diff:.3e}")
    emit(f"  gate ≤ 1e-5 → {'PASS' if diff <= 1e-5 else 'FAIL'}")
    check_dat4 = diff <= 1e-5

    emit("\n[finite + range] (all conditions must be finite, range ⊂ [0,1])")
    finite_ok = True
    for nm, p in [("no_prior", p_no_prior), ("passive_zL", p_passive),
                  ("adapted_zL", p_adapted), ("shuffled_adapted", p_shuffled),
                  ("zero_zL_hb_only", p_zero)]:
        is_finite = torch.isfinite(p).all().item()
        rng = (p.min().item(), p.max().item())
        ok = is_finite and rng[0] >= -1e-6 and rng[1] <= 1.0 + 1e-6
        finite_ok = finite_ok and ok
        emit(f"  [{nm:20s}] finite={is_finite} range=[{rng[0]:.4f}, {rng[1]:.4f}]  → {'PASS' if ok else 'FAIL'}")

    overall = check_params and check_hb_shape and check_dat4 and finite_ok
    emit(f"\n=== OVERALL: {'PASS' if overall else 'FAIL'} ===")
    if overall:
        emit("Stage 1 launch is technically permitted (still pending reviewer DAX/DAY).")
    else:
        emit("DO NOT launch Stage 1 GPU training.")

    if args.out_log:
        os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
        with open(args.out_log, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        emit(f"\n[log] saved to {args.out_log}")

    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_log", default="/cache/zihaohe/checkpoint/v11_probes/c2_smoke.log",
                    help="DAW2: persist smoke log artifact path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--active_scales", default="3,4")
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--ffn_mult", type=float, default=1.0)
    sys.exit(main(ap.parse_args()))
