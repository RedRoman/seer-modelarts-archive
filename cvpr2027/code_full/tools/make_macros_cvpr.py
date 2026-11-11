"""Derive the paper's CVPR-era numbers from results/cvpr/ into LaTeX macros.

The ICLR macro file is kept and included first; this one is included after it
and defines only the keys the rewritten text uses, so a stale ICLR key that no
longer has a result behind it prints a red TBD rather than an old number.

Everything here is computed, never typed: means are seed-averaged per sequence,
differences are paired over sequences, and intervals come from the same 10k
bootstrap the pre-registration's ground rule 4 specifies. A macro whose
underlying files are absent is simply not emitted.
"""
from __future__ import annotations

import glob
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results", "cvpr")
OUT = os.path.join(HERE, "tables", "result_macros_cvpr.tex")
RNG = 0
NBOOT = 10000


def seqs(path):
    j = json.load(open(path))
    return {r["name"]: r for r in j["per_sequence"]}


def seed_avg(paths, metric, names):
    S = [seqs(p) for p in paths]
    return np.array([[s[n][metric] for s in S] for n in names]).mean(1)


def common(paths):
    return sorted(set.intersection(*[set(seqs(p)) for p in paths]))


def boot(d):
    rng = np.random.default_rng(RNG)
    i = rng.integers(0, len(d), size=(NBOOT, len(d)))
    return np.percentile(d[i].mean(1), [2.5, 97.5])


M = {}


def emit(key, value):
    M[key] = value


def paired(a_paths, b_paths, metric):
    names = common(a_paths + b_paths)
    d = seed_avg(a_paths, metric, names) - seed_avg(b_paths, metric, names)
    return d.mean(), boot(d), len(names)


def fmt_ci(ci, nd=3):
    return "[%+.*f, %+.*f]" % (nd, ci[0], nd, ci[1])


# ---------------------------------------------------------------- in-domain
P = sorted(glob.glob(f"{RES}/indomain/*pretrained_s*.json"))
N = sorted(glob.glob(f"{RES}/indomain/*none_s*.json"))
if P and N:
    for metric, key, nd in (("psnr_y", "psnr", 3), ("lpips", "lpips", 4)):
        m, ci, n = paired(P, N, metric)
        emit(f"indomain@{key}", "%+.*f" % (nd, m))
        emit(f"indomain@{key}@ci", fmt_ci(ci, nd))
    emit("indomain@n", str(len(common(P + N))))
    emit("indomain@seeds", str(len(P)))
    names = common(P + N)
    emit("indomain@abs@prior", "%.3f" % seed_avg(P, "psnr_y", names).mean())
    emit("indomain@abs@none", "%.3f" % seed_avg(N, "psnr_y", names).mean())
    emit("indomain@abs@prior@lpips", "%.4f" % seed_avg(P, "lpips", names).mean())
    emit("indomain@abs@none@lpips", "%.4f" % seed_avg(N, "lpips", names).mean())

# the random-token control: what the tokens' content is worth
R = sorted(glob.glob(f"{RES}/indomain/*random_s*.json"))
if P and R:
    # matched seeds only, so an unmatched extra seed cannot shift the gap
    pr = [p for p in P if "_s0" in p]
    m, ci, n = paired(pr, R, "psnr_y")
    emit("randomgap@psnr", "%.2f" % abs(m))
    emit("randomgap@psnr@ci", fmt_ci(ci, 2))

# ---------------------------------------------------------------- baseline
rf = f"{RES}/retinexformer.json"
if os.path.exists(rf) and P:
    names = sorted(set(common(P)) & set(seqs(rf)))
    ours = seed_avg(P, "psnr_y", names)
    theirs = np.array([seqs(rf)[n]["psnr_y"] for n in names])
    d = ours - theirs
    emit("vsrf@psnr", "%+.3f" % d.mean())
    emit("vsrf@psnr@ci", fmt_ci(boot(d), 3))
    emit("vsrf@median", "%+.3f" % float(np.median(d)))
    emit("vsrf@won", "%d" % int((d > 0).sum()))
    emit("vsrf@n", "%d" % len(names))
    ol = seed_avg(P, "lpips", names)
    tl = np.array([seqs(rf)[n]["lpips"] for n in names])
    emit("vsrf@lpips", "%+.4f" % (ol - tl).mean())
    emit("vsrf@lpips@ci", fmt_ci(boot(ol - tl), 4))
    emit("rf@abs@psnr", "%.3f" % theirs.mean())
    emit("rf@abs@lpips", "%.4f" % tl.mean())

# ---------------------------------------------------------------- cross-domain
for dom in ("sdsd", "bvi", "lol"):
    dp = sorted(glob.glob(f"{RES}/xdomain/e20_{dom}_pretrained_s*.json"))
    dn = sorted(glob.glob(f"{RES}/xdomain/e20_{dom}_none_s*.json"))
    if not (dp and dn):
        continue
    for metric, key, nd in (("psnr_y", "psnr", 3), ("lpips", "lpips", 4)):
        m, ci, n = paired(dp, dn, metric)
        emit(f"xdom@{dom}@{key}", "%+.*f" % (nd, m))
        emit(f"xdom@{dom}@{key}@ci", fmt_ci(ci, nd))
    emit(f"xdom@{dom}@n", str(len(common(dp + dn))))
    emit(f"xdom@{dom}@seeds", str(len(dp)))

# ---------------------------------------------------------------- injection
B = sorted(glob.glob(f"{RES}/indomain/e17_small_pretrained_s*.json"))
for tag, pat in (("spatial", "e18_spatial_s*"), ("hybrid", "e18_hybrid_s*"),
                 ("edc", "e18_edc_s*"), ("gated", "e21_gated_s*")):
    v = sorted(p for p in glob.glob(f"{RES}/injection/{pat}.json")
               if "randtok" not in p)
    if not (v and B):
        continue
    # match seeds so the comparison is within-seed
    sv = {os.path.basename(p).split("_s")[-1][0] for p in v}
    bb = [p for p in B if os.path.basename(p).split("_s")[-1][0] in sv]
    for metric, key, nd in (("psnr_y", "psnr", 3), ("lpips", "lpips", 4)):
        m, ci, n = paired(v, bb, metric)
        emit(f"inj@{tag}@{key}", "%+.*f" % (nd, m))
        emit(f"inj@{tag}@{key}@ci", fmt_ci(ci, nd))

# gate audit: the two numbers that showed the gate never closed
for s in (0, 1):
    f = f"{RES}/injection/gate_audit_s{s}.json"
    if os.path.exists(f):
        g = json.load(open(f))["gate_mean"]
        emit(f"gate@s{s}@real", "%.4f" % g["real tokens"])
        emit(f"gate@s{s}@random", "%.4f" % g["random tokens"])

# ---------------------------------------------------------------- H17
for tag in ("e24_tdcos", "e22_tdelta"):
    v = sorted(glob.glob(f"{RES}/h17/{tag}_s*.json"))
    v = [p for p in v if "tdprobe" not in p and "order" not in p]
    if not (v and B):
        continue
    short = "tdcos" if "e24" in tag else "tdrel"
    for metric, key, nd in (("psnr_y", "psnr", 3), ("lpips", "lpips", 4)):
        m, ci, n = paired(v, B, metric)
        emit(f"h17@{short}@{key}", "%+.*f" % (nd, m))
        emit(f"h17@{short}@{key}@ci", fmt_ci(ci, nd))
    # Match aggregate.py --temporal: first seed-average each arm's per-sequence
    # tLP and GT tLP, then take their absolute gap. A signed difference can
    # reward a clip that moves further below the target's temporal activity.
    names = common(v + B)
    gv = np.abs(np.array([[seqs(p)[n]["tlp"] - seqs(p)[n]["gt_tlp"] for p in v]
                          for n in names]).mean(1))
    gb = np.abs(np.array([[seqs(p)[n]["tlp"] - seqs(p)[n]["gt_tlp"] for p in B]
                          for n in names]).mean(1))
    emit(f"h17@{short}@tlp", "%+.4f" % (gv - gb).mean())
    emit(f"h17@{short}@tlp@ci", fmt_ci(boot(gv - gb), 4))
    pr = [json.load(open(p.replace(".json", "_tdprobe.json")))["mean"]
          for p in v if os.path.exists(p.replace(".json", "_tdprobe.json"))]
    if pr:
        emit(f"h17@{short}@cos", "%+.3f" % np.mean([x["cos_pred"] for x in pr]))
        emit(f"h17@{short}@mag", "%.3f" % np.mean([x["mag_pred"] for x in pr]))

o = f"{RES}/h17/e24_tdcos_order.json"
if os.path.exists(o):
    pc = json.load(open(o))["per_clip"]
    a = np.mean([r["intact/pretrained"] for r in pc])
    b = np.mean([r["shuffled/pretrained"] for r in pc])
    emit("ordersens", "%.3f" % abs(a - b))

# ---------------------------------------------------------------- E2 / H16' / H18
# Same three-seed full-token baseline for all three, matched by sequence; the
# variants have two seeds each, so the per-seed sign is reported separately in
# the prereg rather than folded into a macro.
B3 = sorted(glob.glob(f"{RES}/indomain/*pretrained_s*.json"))
for tag, pat in (("e2deep", "wave/e2c_deep_s*.json"),
                 ("cgate", "wave/e25_cgate_s[01].json"),
                 ("grank64", "e26/e26_grank64_s*.json")):
    v = sorted(glob.glob(f"{RES}/{pat}"))
    if not (v and B3):
        continue
    for metric, key, nd in (("psnr_y", "psnr", 3), ("lpips", "lpips", 4)):
        m, ci, n = paired(v, B3, metric)
        emit(f"{tag}@{key}", "%+.*f" % (nd, m))
        emit(f"{tag}@{key}@ci", fmt_ci(ci, nd))

# H16'(i): what random tokens cost the corruption-trained gate
g = sorted(glob.glob(f"{RES}/wave/e25_cgate_s[01].json"))
r = sorted(glob.glob(f"{RES}/wave/e25_cgate_randtok_s*.json"))
if g and r:
    m, ci, n = paired(r, g, "psnr_y")
    emit("cgate@randcost", "%.2f" % abs(m))
    emit("cgate@randcost@ci", fmt_ci(ci, 2))

# ---------------------------------------------------------------- sufficiency
sf = sorted(glob.glob(f"{RES}/sufficiency/token_sufficiency_s*.json"))
if sf:
    per = [json.load(open(p))["per_clip"] for p in sf]
    base = np.mean([np.array(p["identity"]["psnr"]) for p in per], 0)
    for tname, key in (("rank 4", "rank4"), ("spatial mean", "spatial"),
                       ("temporal mean", "temporal"), ("rank 16", "rank16")):
        if tname not in per[0]:
            continue
        v = np.mean([np.array(p[tname]["psnr"]) for p in per], 0)
        d = v - base
        ci = boot(d)
        if ci[0] < 0:
            emit(f"suff@{key}", "%.2f" % abs(ci[0]))          # the loss bound
        emit(f"suff@{key}@d", "%+.3f" % d.mean())
        emit(f"suff@{key}@ci", fmt_ci(ci, 3))
    emit("suff@seeds", str(len(sf)))

gs = sorted(glob.glob(f"{RES}/sufficiency/global_sufficiency_s*.json"))
if gs:
    per = [json.load(open(p))["per_clip"] for p in gs]
    base = np.mean([np.array(p["identity"]["psnr"]) for p in per], 0)
    for k in (4, 16, 64):
        for kind in ("clip", "global"):
            v = np.mean([np.array(p[f"{kind} rank {k}"]["psnr"]) for p in per], 0)
            d = v - base
            emit(f"gsuff@{kind}{k}", "%+.3f" % d.mean())
            emit(f"gsuff@{kind}{k}@ci", fmt_ci(boot(d), 3))
    emit("gsuff@seeds", str(len(gs)))

with open(OUT, "w") as fh:
    fh.write("% AUTO-GENERATED by tools/make_macros_cvpr.py -- do not edit.\n")
    fh.write("% Derived from results/cvpr/; a key with no result is not emitted\n")
    fh.write("% and prints a red TBD through \\R.\n")
    for k in sorted(M):
        fh.write("\\expandafter\\def\\csname res@%s\\endcsname{%s}\n" % (k, M[k]))
print("wrote %s with %d macros" % (OUT, len(M)))
for k in sorted(M):
    print("  %-28s %s" % (k, M[k]))
