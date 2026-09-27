"""Create tools/v17_wanvj_bridge_sigma_train.py from the stable v3 trainer by
adding EXACTLY ONE mechanism: a boundary-vanishing stochastic term on the bridge.

    deterministic (v10, current):  z_t = (1-t) z_lq + t z_hq
    stochastic    (v17, RIVER):    z_t = (1-t) z_lq + t z_hq + sigma_0 sin(pi t) eps

v_target is UNCHANGED (still z_hq - z_lq) -- this matches RIVER Eq.3, where the
drift is supervised against the endpoint displacement even though the input is
noise-perturbed.

sigma(0)=sigma(1)=0, so the 1-NFE inference path (t=0) is bit-identical to v10.
--bridge_sigma 0.0 reproduces v10 exactly (branch is skipped).

Run from the vjepa_llve repo root:  python make_v17.py
"""
import re
import shutil
import sys

SRC = "tools/v11_wanvj_phase2_v3_train.py"
DST = "tools/v17_wanvj_bridge_sigma_train.py"

src = open(SRC).read()
orig_len = len(src)

# ---------------------------------------------------------------- patch 1
# Inject the noise right after the deterministic interpolation, and make the
# per-step noise magnitude observable so we can PROVE the branch fired.
old1 = """            tv = t.view(B, 1, 1, 1, 1)
            z_t = (1.0 - tv) * z_lq + tv * z_hq
            dove_active = False
        v_target = z_hq - z_lq"""
new1 = """            tv = t.view(B, 1, 1, 1, 1)
            z_t = (1.0 - tv) * z_lq + tv * z_hq
            # ---- v17: RIVER-style stochastic bridge -------------------------
            # Boundary-vanishing noise turns the deterministic linear flow into
            # a genuine bridge pinned at both endpoints. sigma(0)=sigma(1)=0, so
            # the t=0 inference path is untouched.
            if args.bridge_sigma > 0.0:
                sigma_t = args.bridge_sigma * torch.sin(torch.pi * tv)
                _bridge_eps = torch.randn_like(z_t)
                z_t = z_t + sigma_t * _bridge_eps
                _sig_std = float((sigma_t * _bridge_eps).std())
            # -----------------------------------------------------------------
            dove_active = False
        v_target = z_hq - z_lq"""
assert src.count(old1) == 1, f"patch1 anchor count = {src.count(old1)}"
src = src.replace(old1, new1)

# ---------------------------------------------------------------- patch 2
# _sig_std must exist on every path (incl. the DOVE branch, which we leave alone).
old2 = """        B = z_lq.shape[0]
        # DOVE training mixin"""
new2 = """        B = z_lq.shape[0]
        _sig_std = 0.0   # v17: actual injected-noise std this step (0 => none)
        # DOVE training mixin"""
assert src.count(old2) == 1, f"patch2 anchor count = {src.count(old2)}"
src = src.replace(old2, new2)

# ---------------------------------------------------------------- patch 3
# Surface it in the training log -- "installed" is not "fired".
old3 = """                  f"haste={int(haste_active)} eta={eta_h:.1f}h")"""
new3 = """                  f"sg={_sig_std:.4f} "
                  f"haste={int(haste_active)} eta={eta_h:.1f}h")"""
assert src.count(old3) == 1, f"patch3 anchor count = {src.count(old3)}"
src = src.replace(old3, new3)

# ---------------------------------------------------------------- patch 4
old4 = """    ap.add_argument("--dove_prob", type=float, default=0.0,"""
new4 = """    ap.add_argument("--bridge_sigma", type=float, default=0.0,
                    help="v17: stochastic bridge noise scale sigma_0 in "
                         "z_t += sigma_0*sin(pi*t)*eps. 0.0 = deterministic "
                         "linear bridge (identical to v10).")
    ap.add_argument("--dove_prob", type=float, default=0.0,"""
assert src.count(old4) == 1, f"patch4 anchor count = {src.count(old4)}"
src = src.replace(old4, new4)

# ---------------------------------------------------------------- patch 5
old5 = """        f.write(f"DOVE t={args.dove_t}\\n")"""
new5 = """        f.write(f"DOVE t={args.dove_t}\\n")
        f.write(f"Bridge sigma_0: {args.bridge_sigma} "
                f"(sigma(t)=sigma_0*sin(pi*t); 0.0 = deterministic)\\n")"""
assert src.count(old5) == 1, f"patch5 anchor count = {src.count(old5)}"
src = src.replace(old5, new5)

open(DST, "w").write(src)
print(f"wrote {DST}  ({orig_len} -> {len(src)} chars, +{len(src)-orig_len})")

# ---------------------------------------------------------------- verify
import ast
ast.parse(src)
print("syntax OK")

for name, pat in [
    ("bridge_sigma flag", r'--bridge_sigma'),
    ("noise injection",   r'z_t = z_t \+ sigma_t \* _bridge_eps'),
    ("sin schedule",      r'torch\.sin\(torch\.pi \* tv\)'),
    ("log field",         r'sg=\{_sig_std'),
]:
    n = len(re.findall(pat, src))
    print(f"  {name:20s} occurrences={n}  {'OK' if n >= 1 else 'FAIL'}")

# The deterministic line must still be there exactly once, untouched.
n = src.count("z_t = (1.0 - tv) * z_lq + tv * z_hq")
print(f"  {'base interp kept':20s} occurrences={n}  {'OK' if n == 1 else 'FAIL'}")

# v_target must NOT have changed.
n = src.count("v_target = z_hq - z_lq")
print(f"  {'v_target unchanged':20s} occurrences={n}  {'OK' if n == 1 else 'FAIL'}")
