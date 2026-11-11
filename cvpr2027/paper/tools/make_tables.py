"""Build every results table and number macro of the paper from the eval JSONs.

Nothing in tables/ is edited by hand: each cell is recomputed from per-sequence
results, so all methods on a split are averaged over the same sequence list
(checked below). Missing results render as a red TBD, never as a guess.

Usage: python tools/make_tables.py   (after tools/sync_results.sh)
Outputs: tables/*.tex, tables/result_macros.tex, tables/PROVENANCE.md
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
NEW = os.path.join(RES, "seer_iclr_evals")
OLD = os.path.join(RES, "server")
OUT = os.path.join(ROOT, "tables")

SPLITS = ["L0", "L1", "L2", "L3", "L4", "L5"]
VIDEO = ["L0", "L1", "L2"]
SPLIT_NAME = {"L0": "SDSD-in", "L1": "SDSD-out", "L2": "BVI-RLV",
              "L3": "LOL-v1", "L4": "LOLv2-Syn", "L5": "LOLv2-Real"}
SEER_KEY = {"L0": "L0_SDSD-indoor", "L1": "L1_SDSD-outdoor", "L2": "L2_BVI-RLV",
            "L3": "L3_LOL-v1", "L4": "L4_LOLv2-Syn", "L5": "L5_LOLv2-Real"}
V2_PREFIX = {"L1": "SDSD_outdoor", "L2": "BVI_RLV", "L3": "LOL_v1",
             "L4": "LOLv2_Syn", "L5": "LOLv2_Real"}
TEMP_KEY = {"L0": "L0_SDSD-indoor", "L1": "L1_SDSD-outdoor", "L2": "L2_BVI-RLV"}
CLEAN6 = ["pair50", "pair52", "pair58", "pair60", "pair68", "pair70"]
LOWER = {"lpips", "dists", "tlp", "mabd", "flicker", "warp_err"}
FMT = {"psnr_rgb": "{:.2f}", "lpips": "{:.3f}", "dists": "{:.3f}", "clipiqa": "{:.3f}",
       "tlp": "{:.3f}", "mabd": "{:.4f}", "flicker": "{:.3f}", "warp_err": "{:.4f}"}
SEEDS = [0, 1, 2]

PROV: List[str] = []
MACROS: "OrderedDict[str, str]" = OrderedDict()
WARN: List[str] = []


def jload(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-sequence loaders. Each returns {split: {seq: {metric: value}}}.
# ---------------------------------------------------------------------------
def merge(dst: dict, split: str, seqs: Dict[str, dict]) -> None:
    for s, m in seqs.items():
        dst.setdefault(split, {}).setdefault(s, {}).update(
            {k: v for k, v in m.items() if isinstance(v, (int, float))})


def load_baseline(name: str) -> dict:
    out: dict = {}
    fx = jload(os.path.join(NEW, "baselines_L0_fixedpairing_eval.json"))
    if fx and name in fx["per_seq"]["L0_SDSD-indoor"]:
        merge(out, "L0", fx["per_seq"]["L0_SDSD-indoor"][name])
    if name == "STCD":
        st = jload(os.path.join(NEW, "STCD_unified_eval.json"))
        for sp, key in SEER_KEY.items():
            if sp != "L0" and key in st["per_seq"]:
                merge(out, sp, st["per_seq"][key])
    else:
        v2 = jload(os.path.join(OLD, f"{name}_eval_v2.json"))
        if v2:
            for sp, pre in V2_PREFIX.items():
                merge(out, sp, {k.split("__", 1)[1]: v for k, v in v2["per_seq"].items()
                                if k.startswith(pre + "__")})
    tm = jload(os.path.join(NEW, "baselines_temporal_unified_v2.json"))
    if tm:
        for sp, key in TEMP_KEY.items():
            if name in tm["per_seq"].get(key, {}):
                merge(out, sp, tm["per_seq"][key][name])
    PROV.append(f"{name}: L0 baselines_L0_fixedpairing_eval.json; L1-L5 "
                f"{'STCD_unified_eval.json' if name == 'STCD' else name + '_eval_v2.json'}; "
                f"temporal baselines_temporal_unified_v2.json")
    return out


def load_retinexformer_retrained(tag: str) -> Optional[dict]:
    """tag in {union, indoor}: Retinexformer retrained on Seer's split."""
    six = jload(os.path.join(NEW, f"Retinexformer_{tag}_seersplit_eval_v2.json"))
    l0 = jload(os.path.join(NEW, f"Retinexformer_{tag}_seersplit_L0seer_eval_v2.json"))
    if not six or not l0:
        return None
    out: dict = {}
    merge(out, "L0", {k.split("__", 1)[1]: v for k, v in l0["per_seq"].items()})
    for sp, pre in V2_PREFIX.items():
        merge(out, sp, {k.split("__", 1)[1]: v for k, v in six["per_seq"].items()
                        if k.startswith(pre + "__")})
    tm = jload(os.path.join(NEW, "Retinexformer_seersplit_temporal.json"))
    label = {"union": "Retinexformer-union-seersplit", "indoor": "Retinexformer-indoor-seersplit"}[tag]
    if tm:
        for sp, key in TEMP_KEY.items():
            if label in tm["per_seq"].get(key, {}):
                merge(out, sp, tm["per_seq"][key][label])
    PROV.append(f"Retinexformer retrained ({tag}, Seer split): "
                f"Retinexformer_{tag}_seersplit_eval_v2.json + _L0seer_eval_v2.json")
    return out


def load_seer_json(path: str, variant: str) -> Optional[dict]:
    d = jload(path)
    if not d:
        return None
    out: dict = {}
    for sp, key in SEER_KEY.items():
        if key in d["per_seq"] and variant in d["per_seq"][key]:
            merge(out, sp, d["per_seq"][key][variant])
    return out


def load_seer_runs(prefix: str, variant: str, ckpt: str = "") -> List[dict]:
    """All available seeds of a controlled run family, e.g. ctrl_vjepa, variant HMP-raw."""
    runs = []
    for s in SEEDS:
        suffix = f"_{ckpt}" if ckpt else ""
        p = os.path.join(NEW, f"{prefix}_s{s}{suffix}_eval_unified.json")
        r = load_seer_json(p, variant)
        if r:
            runs.append(r)
            PROV.append(f"{prefix}_s{s}{suffix} [{variant}]: {os.path.basename(p)}")
    return runs


def load_gt_temporal() -> dict:
    tm = jload(os.path.join(NEW, "baselines_temporal_unified_v2.json"))
    out: dict = {}
    for name in ("GT", "LQ"):
        out[name] = {}
        for sp, key in TEMP_KEY.items():
            if tm and name in tm["per_seq"].get(key, {}):
                merge(out[name], sp, tm["per_seq"][key][name])
    fx = jload(os.path.join(NEW, "baselines_L0_fixedpairing_eval.json"))
    if fx and "LQ" in fx["per_seq"]["L0_SDSD-indoor"]:
        merge(out["LQ"], "L0", fx["per_seq"]["L0_SDSD-indoor"]["LQ"])
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
REF_SEQS: Dict[str, List[str]] = {}


def agg(run: Optional[dict], split: str, metric: str, subset: Optional[List[str]] = None,
        who: str = "") -> Optional[float]:
    if not run or split not in run:
        return None
    seqs = sorted(run[split])
    if subset is not None:
        seqs = [s for s in seqs if s in subset]
    vals = [run[split][s][metric] for s in seqs if metric in run[split][s]]
    if not vals or len(vals) != len(seqs):
        return None
    if subset is None:
        ref = REF_SEQS.setdefault(split, seqs)
        if ref != seqs:
            WARN.append(f"{who} {split}: sequence list differs from reference "
                        f"({len(seqs)} vs {len(ref)})")
    return float(np.mean(vals))


def agg_seeds(runs: List[dict], split: str, metric: str, subset=None, who=""):
    vals = [agg(r, split, metric, subset, who) for r in runs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None), len(vals)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
TBD = r"\tbd"


def fmt(v: Optional[float], metric: str, std: Optional[float] = None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return TBD
    s = FMT[metric].format(v)
    if metric != "psnr_rgb" and s.startswith("0."):
        s = s[1:]
    if std is not None:
        t = FMT[metric].format(std)
        if metric != "psnr_rgb" and t.startswith("0."):
            t = t[1:]
        s += r"{\tiny$\pm$" + t + "}"
    return s


def macro(name: str, value: str) -> None:
    MACROS[name] = value


def rank_marks(values: List[Optional[float]], metric: str) -> List[str]:
    idx = [i for i, v in enumerate(values) if v is not None]
    order = sorted(idx, key=lambda i: values[i], reverse=metric not in LOWER)
    marks = [""] * len(values)
    if order:
        marks[order[0]] = "best"
    if len(order) > 1:
        marks[order[1]] = "second"
    return marks


def decorate(cell: str, mark: str) -> str:
    if cell == TBD or not mark:
        return cell
    return r"\textbf{" + cell + "}" if mark == "best" else r"\underline{" + cell + "}"


def write(name: str, body: str) -> None:
    with open(os.path.join(OUT, name), "w") as f:
        f.write("% AUTO-GENERATED by tools/make_tables.py -- do not edit by hand.\n")
        f.write(body)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
class Row:
    def __init__(self, label: str, key: str, runs: List[dict], train: str = "",
                 rank: bool = True, group: str = ""):
        self.label, self.key, self.runs, self.train = label, key, runs, train
        self.rank, self.group = rank, group

    def get(self, split, metric, subset=None):
        m, s, n = agg_seeds(self.runs, split, metric, subset, self.label)
        return m, s, n


def build_rows():
    rows = []
    base = [("Zero-DCE", "zerodce", "unsup."), ("SCI", "sci", "unsup."),
            ("HVI-CIDNet", "hvicidnet", "LOL"), ("DarkIR", "darkir", "LOL-Blur"),
            ("URCSA", "urcsa", "SDSD+LOL"), ("STCD", "stcd", "SDSD-in"),
            ("Retinexformer", "retinexformer", "SDSD")]
    for name, key, train in base:
        file_name = {"zerodce": "Zero-DCE", "sci": "SCI", "hvicidnet": "HVI-CIDNet",
                     "darkir": "DarkIR", "urcsa": "URCSA", "stcd": "STCD",
                     "retinexformer": "Retinexformer"}[key]
        rows.append(Row(name + (r"$^{\dagger}$" if key in ("retinexformer", "urcsa", "stcd") else ""),
                        key, [load_baseline(file_name)], train, group="released"))
    rt = load_retinexformer_retrained("union")
    rows.append(Row(r"Retinexformer (retrained)", "retinexformerRT", [rt] if rt else [],
                    r"SDSD$^{\ast}$", group="retrained"))
    rows.append(Row(r"Pixel branch w/o V-JEPA", "novjepa",
                    load_seer_runs("ctrl_novjepa", "HMP-raw"), r"SDSD$^{\ast}$", group="ours"))
    rows.append(Row(r"\textbf{Seer}", "seer", load_seer_runs("ctrl_vjepa", "HMP-raw"),
                    r"SDSD$^{\ast}$", group="ours"))
    rows.append(Row(r"\textbf{Seer}\,+\,CP-src", "seercp", load_seer_runs("ctrl_vjepa", "HMP+CPfit"),
                    r"SDSD$^{\ast}$", group="ours"))
    return rows


def table_main(rows: List[Row]) -> None:
    metrics = ["psnr_rgb", "lpips"]
    cols = [(sp, m) for sp in SPLITS for m in metrics]
    cells = {r.key: [r.get(sp, m) for sp, m in cols] for r in rows}
    marks = {}
    for j, (sp, m) in enumerate(cols):
        vals = [cells[r.key][j][0] if r.rank else None for r in rows]
        for r, mk in zip(rows, rank_marks(vals, m)):
            marks.setdefault(r.key, []).append(mk)
    lines = [r"\begin{tabular}{l l " + " ".join(["cc"] * len(SPLITS)) + "}", r"\toprule",
             "& Train & " + " & ".join(
                 r"\multicolumn{2}{c}{\textbf{%s} %s}" % (sp, SPLIT_NAME[sp]) for sp in SPLITS) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (3 + 2 * i, 4 + 2 * i) for i in range(len(SPLITS))),
             "Method & data & " + " & ".join(["PSNR$\\uparrow$ & LPIPS$\\downarrow$"] * len(SPLITS)) + r" \\",
             r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r.group != prev:
            lines.append(r"\midrule")
        prev = r.group
        out = []
        for j, (sp, m) in enumerate(cols):
            mean, std, n = cells[r.key][j]
            out.append(decorate(fmt(mean, m, std), marks[r.key][j]))
            macro(f"{r.key}@{sp}@{m}", fmt(mean, m))
        lines.append(f"{r.label} & {r.train} & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_main.tex", "\n".join(lines) + "\n")


def table_extra_metrics(rows: List[Row]) -> None:
    metrics = ["dists", "clipiqa"]
    cols = [(sp, m) for sp in SPLITS for m in metrics]
    lines = [r"\begin{tabular}{l " + " ".join(["cc"] * len(SPLITS)) + "}", r"\toprule",
             "& " + " & ".join(r"\multicolumn{2}{c}{\textbf{%s}}" % sp for sp in SPLITS) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(SPLITS))),
             "Method & " + " & ".join(["DISTS$\\downarrow$ & CLIPIQA$\\uparrow$"] * len(SPLITS)) + r" \\",
             r"\midrule"]
    vals = {r.key: [r.get(sp, m) for sp, m in cols] for r in rows}
    marks = {}
    for j, (sp, m) in enumerate(cols):
        for r, mk in zip(rows, rank_marks([vals[r.key][j][0] if r.rank else None for r in rows], m)):
            marks.setdefault(r.key, []).append(mk)
    for r in rows:
        out = [decorate(fmt(vals[r.key][j][0], m, vals[r.key][j][1]), marks[r.key][j])
               for j, (sp, m) in enumerate(cols)]
        for j, (sp, m) in enumerate(cols):
            macro(f"{r.key}@{sp}@{m}", fmt(vals[r.key][j][0], m))
        lines.append(f"{r.label} & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_extra_metrics.tex", "\n".join(lines) + "\n")


def table_clean6(rows: List[Row]) -> None:
    """L0 restricted to the 6 sequences no method was trained on."""
    metrics = ["psnr_rgb", "lpips", "dists", "tlp", "warp_err"]
    vals = {r.key: [r.get("L0", m, CLEAN6) for m in metrics] for r in rows}
    marks = {}
    for j, m in enumerate(metrics):
        for r, mk in zip(rows, rank_marks([vals[r.key][j][0] if r.rank else None for r in rows], m)):
            marks.setdefault(r.key, []).append(mk)
    lines = [r"\begin{tabular}{l ccccc}", r"\toprule",
             r"Method & PSNR$\uparrow$ & LPIPS$\downarrow$ & DISTS$\downarrow$ & tLP$\downarrow$ & WE$\downarrow$ \\",
             r"\midrule"]
    for r in rows:
        out = [decorate(fmt(vals[r.key][j][0], m, vals[r.key][j][1]), marks[r.key][j])
               for j, m in enumerate(metrics)]
        for j, m in enumerate(metrics):
            macro(f"{r.key}@L0c6@{m}", fmt(vals[r.key][j][0], m))
        lines.append(f"{r.label} & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_clean6.tex", "\n".join(lines) + "\n")


def table_temporal(rows: List[Row], ref: dict) -> None:
    metrics = ["tlp", "warp_err", "mabd", "flicker"]
    cols = [(sp, m) for sp in VIDEO for m in metrics]
    allrows = [Row("GT", "gt", [ref["GT"]], rank=False, group="ref"),
               Row("LQ input", "lq", [ref["LQ"]], rank=False, group="ref")] + rows
    vals = {r.key: [r.get(sp, m) for sp, m in cols] for r in allrows}
    marks = {}
    for j, (sp, m) in enumerate(cols):
        for r, mk in zip(allrows, rank_marks([vals[r.key][j][0] if r.rank else None for r in allrows], m)):
            marks.setdefault(r.key, []).append(mk)
    lines = [r"\begin{tabular}{l " + " ".join(["cccc"] * len(VIDEO)) + "}", r"\toprule",
             "& " + " & ".join(r"\multicolumn{4}{c}{\textbf{%s} %s}" % (sp, SPLIT_NAME[sp])
                               for sp in VIDEO) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * i, 5 + 4 * i) for i in range(len(VIDEO))),
             "Method & " + " & ".join(["tLP & WE & MABD & flk"] * len(VIDEO)) + r" \\", r"\midrule"]
    prev = None
    for r in allrows:
        if prev is not None and r.group != prev:
            lines.append(r"\midrule")
        prev = r.group
        out = [decorate(fmt(vals[r.key][j][0], m, vals[r.key][j][1]), marks[r.key][j])
               for j, (sp, m) in enumerate(cols)]
        for j, (sp, m) in enumerate(cols):
            macro(f"{r.key}@{sp}@{m}", fmt(vals[r.key][j][0], m))
        lines.append(f"{r.label} & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_temporal.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Controlled V-JEPA study and ablations
# ---------------------------------------------------------------------------
def paired_delta(a: List[dict], b: List[dict], split: str, metric: str, subset=None):
    """Seed-level delta mean(b) - mean(a) and a sequence-bootstrap CI on seed-averaged outputs."""
    if not a or not b:
        return None, None, None
    seqs = sorted(set.intersection(*[set(r[split]) for r in a + b if split in r]))
    if subset is not None:
        seqs = [s for s in seqs if s in subset]
    if not seqs:
        return None, None, None
    A = np.mean([[r[split][s][metric] for s in seqs] for r in a], axis=0)
    B = np.mean([[r[split][s][metric] for s in seqs] for r in b], axis=0)
    d = B - A
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (10000, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def table_temporal_compact(rows: List[Row], ref: dict) -> None:
    """Main-text temporal table: tLP and motion-compensated warping error only."""
    metrics = ["tlp", "warp_err"]
    cols = [(sp, m) for sp in VIDEO for m in metrics]
    allrows = [Row("GT", "gt", [ref["GT"]], rank=False, group="ref")] + rows
    vals = {r.key: [r.get(sp, m) for sp, m in cols] for r in allrows}
    marks = {}
    for j, (sp, m) in enumerate(cols):
        for r, mk in zip(allrows, rank_marks([vals[r.key][j][0] if r.rank else None for r in allrows], m)):
            marks.setdefault(r.key, []).append(mk)
    lines = [r"\begin{tabular}{l " + " ".join(["cc"] * len(VIDEO)) + "}", r"\toprule",
             "& " + " & ".join(r"\multicolumn{2}{c}{\textbf{%s}}" % sp for sp in VIDEO) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(VIDEO))),
             "Method & " + " & ".join(["tLP & WE"] * len(VIDEO)) + r" \\", r"\midrule"]
    prev = None
    for r in allrows:
        if prev is not None and r.group != prev:
            lines.append(r"\midrule")
        prev = r.group
        out = [decorate(fmt(vals[r.key][j][0], m, vals[r.key][j][1]), marks[r.key][j])
               for j, (sp, m) in enumerate(cols)]
        lines.append(f"{r.label} & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_temporal_compact.tex", "\n".join(lines) + "\n")


def table_control() -> None:
    fam = OrderedDict([
        ("novjepa", ("No V-JEPA (zero tokens)", "ctrl_novjepa", "")),
        ("randvjepa", ("Random-init V-JEPA", "abl_randvjepa", "")),
        ("vjepa", ("Pretrained V-JEPA (Seer)", "ctrl_vjepa", "")),
    ])
    metrics = [("L0", "psnr_rgb"), ("L0", "lpips"), ("L1", "psnr_rgb"), ("L1", "lpips"),
               ("L2", "psnr_rgb"), ("L2", "lpips"), ("L0", "tlp"), ("L0", "warp_err")]
    lines = [r"\begin{tabular}{l c " + "c" * len(metrics) + "}", r"\toprule",
             r"& & \multicolumn{2}{c}{\textbf{L0}} & \multicolumn{2}{c}{\textbf{L1}} & "
             r"\multicolumn{2}{c}{\textbf{L2}} & \multicolumn{2}{c}{\textbf{L0} temporal} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
             r"V-JEPA tokens & seeds & PSNR & LPIPS & PSNR & LPIPS & PSNR & LPIPS & tLP & WE \\",
             r"\midrule"]
    runs = {}
    for key, (label, prefix, ck) in fam.items():
        variant = "HMP-raw"
        runs[key] = load_seer_runs(prefix, variant)
        cells = []
        for sp, m in metrics:
            mean, std, n = agg_seeds(runs[key], sp, m, who=label)
            cells.append(fmt(mean, m, std))
            macro(f"ctrl{key}@{sp}@{m}", fmt(mean, m))
        macro(f"ctrl{key}@nseeds", str(len(runs[key])))
        lines.append(f"{label} & {len(runs[key]) or TBD} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for key, label in (("novjepa", r"$\Delta$ pretrained $-$ none"),
                       ("randvjepa", r"$\Delta$ pretrained $-$ random")):
        cells = []
        for sp, m in metrics:
            d, lo, hi = paired_delta(runs[key], runs["vjepa"], sp, m)
            if d is None:
                cells.append(TBD)
                continue
            sig = lo > 0 or hi < 0
            c = ("{:+.2f}" if m == "psnr_rgb" else "{:+.3f}" if m != "warp_err" else "{:+.4f}").format(d)
            cells.append(r"\textbf{" + c + "}" if sig else c)
            macro(f"delta{key}@{sp}@{m}", c)
            macro(f"delta{key}@{sp}@{m}@ci", ("[{:+.2f}, {:+.2f}]" if m == "psnr_rgb" else
                                              "[{:+.3f}, {:+.3f}]").format(lo, hi))
        lines.append(f"{label} & & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_control.tex", "\n".join(lines) + "\n")


ABLATIONS = OrderedDict([
    ("m1active", "M1 gate opened ($g_0{=}0.5$)"),
    ("m3soft", "M3 soft router init ($c{=}1$)"),
    ("noretinex", r"w/o Retinex aux.\ losses ($\lambda_R{=}\lambda_S{=}0$)"),
    ("noigab", "w/o IGAB"),
    ("nolic", "w/o LIC"),
    ("nobottleneck", "w/o temporal mixer + CFIM"),
])


def table_ablation() -> None:
    ref = load_seer_runs("ctrl_vjepa", "HMP-raw", "step_750")
    refcp = load_seer_runs("ctrl_vjepa", "HMP+CPfit", "step_750")
    metrics = [("L0", "psnr_rgb"), ("L0", "lpips"), ("L1", "psnr_rgb"), ("L1", "lpips"),
               ("L2", "psnr_rgb"), ("L2", "lpips"), ("L0", "tlp")]
    lines = [r"\begin{tabular}{l " + "c" * len(metrics) + "}", r"\toprule",
             r"& \multicolumn{2}{c}{\textbf{L0}} & \multicolumn{2}{c}{\textbf{L1}} & "
             r"\multicolumn{2}{c}{\textbf{L2}} & \textbf{L0} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
             r"Variant (750 it.) & PSNR & LPIPS & PSNR & LPIPS & PSNR & LPIPS & tLP \\", r"\midrule"]
    cells = []
    for sp, m in metrics:
        mean, std, n = agg_seeds(ref, sp, m, who="ref750")
        cells.append(fmt(mean, m, std))
        macro(f"abl@ref@{sp}@{m}", fmt(mean, m))
    lines.append(f"Seer, full ({len(ref) or TBD} seeds) & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for key, label in ABLATIONS.items():
        runs = [r for r in [load_seer_json(os.path.join(NEW, f"abl_{key}_s{s}_eval_unified.json"),
                                           "HMP-raw") for s in SEEDS] if r]
        cells = []
        for sp, m in metrics:
            d, lo, hi = paired_delta(ref, runs, sp, m)
            if d is None:
                cells.append(TBD)
                continue
            c = ("{:+.2f}" if m == "psnr_rgb" else "{:+.3f}").format(d)
            cells.append(r"\textbf{" + c + "}" if (lo > 0 or hi < 0) else c)
            macro(f"abl@{key}@{sp}@{m}", c)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    # CP as a post-hoc row on the same reference checkpoints (source-calibrated alpha)
    cells = []
    for sp, m in metrics:
        d, lo, hi = paired_delta(ref, refcp, sp, m)
        cells.append(TBD if d is None else ("{:+.2f}" if m == "psnr_rgb" else "{:+.3f}").format(d))
    lines.append(r"\midrule" + "\n" + r"+ CP-src post-processing & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Encoder swap (archived July runs, original code) + invariance
# ---------------------------------------------------------------------------
def table_encoder() -> None:
    inv = jload(os.path.join(RES, "aaai_data", "invariance_v2.json"))
    ref = load_seer_json(os.path.join(OLD, "reference_v35stage4_L0to5_full_metrics_v2.json"), "HMP-raw")
    refcp = load_seer_json(os.path.join(OLD, "reference_v35stage4_L0to5_full_metrics_v2.json"), "HMP+CP")
    encs = OrderedDict([("vjepa", ("V-JEPA 2.1 (video, latent pred.)", None)),
                        ("videomae", ("VideoMAE (video, pixel recon.)", "videomae")),
                        ("dinov3", ("DINOv3 (image)", "dinov3")),
                        ("dinov2", ("DINOv2 (image)", "dinov2")),
                        ("clip", ("CLIP (image-text)", "clip")),
                        ("siglip2", ("SigLIP\\,2 (image-text)", "siglip"))])
    lines = [r"\begin{tabular}{l c cc cc cc}", r"\toprule",
             r"& Invariance & \multicolumn{2}{c}{\textbf{L0}} & \multicolumn{2}{c}{\textbf{L1}} & "
             r"\multicolumn{2}{c}{\textbf{L2}} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Frozen encoder & $\bar\rho_{\mathrm{patch}}\downarrow$ & PSNR & LPIPS & PSNR & LPIPS & PSNR & LPIPS \\",
             r"\midrule"]
    for key, (label, invkey) in encs.items():
        if key == "vjepa":
            raw, cp = ref, refcp
            rho = inv["summary"]["vjepa"]["mean"]
        else:
            f = os.path.join(OLD, f"encswap_{key}_L012.json")
            raw, cp = load_seer_json(f, "HMP-raw"), load_seer_json(f, "HMP+CP")
            rho = inv["summary"][invkey]["mean"]
        cells = [fmt(rho, "lpips")]
        for sp in ("L0", "L1"):
            cells += [fmt(agg(raw, sp, "psnr_rgb"), "psnr_rgb"), fmt(agg(raw, sp, "lpips"), "lpips")]
        cells += [fmt(agg(raw, "L2", "psnr_rgb"), "psnr_rgb"), fmt(agg(raw, "L2", "lpips"), "lpips")]
        for sp in ("L0", "L1"):
            macro(f"enc{key}@{sp}@psnr_rgb", fmt(agg(raw, sp, "psnr_rgb"), "psnr_rgb"))
            macro(f"enc{key}@{sp}@lpips", fmt(agg(raw, sp, "lpips"), "lpips"))
        macro(f"enc{key}@L2@psnr_rgb", fmt(agg(raw, "L2", "psnr_rgb"), "psnr_rgb"))
        macro(f"enc{key}@L2@lpips", fmt(agg(raw, "L2", "lpips"), "lpips"))
        macro(f"inv@{key}", "{:.3f}".format(rho))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_encoder.tex", "\n".join(lines) + "\n")
    for k, v in inv["summary"].items():
        if "wilcoxon_p_vs_vjepa_less" in v:
            pv = v["wilcoxon_p_vs_vjepa_less"]
            if pv < 1e-3:
                mant, ex = "{:.1e}".format(pv).split("e")
                macro(f"inv@p@{k}", r"%s{\times}10^{%d}" % (mant, int(ex)))
            else:
                macro(f"inv@p@{k}", "{:.2f}".format(pv))
            macro(f"inv@d@{k}", "{:.2f}".format(v["cohens_d_paired_vjepa_minus"]))
    PROV.append("encoder swap: encswap_*_L012.json + reference_v35stage4_L0to5_full_metrics_v2.json "
                "(July runs, original code); invariance: aaai_data/invariance_v2.json")


def table_membership() -> None:
    m = jload(os.path.join(NEW, "train_membership.json"))
    if not m:
        return
    show = [("Retinexformer", "Retinexformer$^{\\dagger}$"), ("URCSA", "URCSA$^{\\dagger}$"),
            ("STCD", "STCD$^{\\dagger}$")]
    split_keys = list(m["methods"]["Retinexformer"]["splits"].keys())
    lines = [r"\begin{tabular}{l ccc}", r"\toprule",
             r"Released checkpoint & L0 (15) & STCD-12 & L1 (25) \\", r"\midrule"]
    for name, label in show:
        sp = m["methods"].get(name, {}).get("splits", {})
        cells = []
        for k in split_keys:
            s = sp.get(k, {})
            if not s:
                cells.append("?")
                continue
            lo, hi = s.get("n_seen", 0), s.get("n_seen_upper", s.get("n_seen", 0))
            unk = s.get("n_unknown", 0)
            if unk and unk == s.get("n"):
                cells.append("unknown")
            else:
                cells.append(("{}".format(lo) if lo == hi else "{}--{}".format(lo, hi)) + ("" if not unk else "?"))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_membership.tex", "\n".join(lines) + "\n")
    PROV.append("membership: train_membership.json")


def table_cp_transfer() -> None:
    """Delta of CP-src / CP-tgt over raw for every method and split (same calibration protocol as Seer)."""
    b = jload(os.path.join(NEW, "baselines_cp_eval.json"))
    methods = ["Zero-DCE", "SCI", "HVI-CIDNet", "DarkIR", "URCSA", "STCD", "Retinexformer"]
    runs = {}
    if b:
        for mth in methods:
            runs[mth] = {}
            for var in ("raw", "CP-src", "CP-tgt"):
                r = {}
                for sp, key in SEER_KEY.items():
                    if key in b["per_seq"] and mth in b["per_seq"][key]:
                        merge(r, sp, b["per_seq"][key][mth][var])
                runs[mth][var] = [r]
        PROV.append("CP transfer: baselines_cp_eval.json")
    runs["w/o V-JEPA"] = {"raw": load_seer_runs("ctrl_novjepa", "HMP-raw"),
                          "CP-src": load_seer_runs("ctrl_novjepa", "HMP+CPfit"),
                          "CP-tgt": load_seer_runs("ctrl_novjepa", "HMP+CPxfit")}
    runs["Seer"] = {"raw": load_seer_runs("ctrl_vjepa", "HMP-raw"),
                    "CP-src": load_seer_runs("ctrl_vjepa", "HMP+CPfit"),
                    "CP-tgt": load_seer_runs("ctrl_vjepa", "HMP+CPxfit")}
    lines = [r"\begin{tabular}{l l " + "c" * len(SPLITS) + "}", r"\toprule",
             r"Method & CP & " + " & ".join(r"\textbf{%s}" % sp for sp in SPLITS) + r" \\", r"\midrule"]
    for mth in methods + ["w/o V-JEPA", "Seer"]:
        if mth not in runs:
            continue
        for var in ("CP-src", "CP-tgt"):
            cells = []
            for sp in SPLITS:
                d, lo, hi = paired_delta(runs[mth]["raw"], runs[mth][var], sp, "psnr_rgb")
                if d is None:
                    cells.append(TBD)
                    continue
                c = "{:+.2f}".format(d)
                cells.append(r"\textbf{" + c + "}" if (lo > 0 or hi < 0) else c)
                macro(f"cpx@{mth}@{var}@{sp}", c)
            label = (r"\textbf{Seer}" if mth == "Seer" else mth) if var == "CP-src" else ""
            lines.append(f"{label} & {var[3:]} & " + " & ".join(cells) + r" \\")
        if mth != "Seer":
            lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_cp_transfer.tex", "\n".join(lines) + "\n")


def load_gtmean_seer(prefix: str, variant: str) -> List[dict]:
    runs = []
    for s_ in SEEDS:
        r = load_seer_json(os.path.join(NEW, f"{prefix}_s{s_}_gtmean_eval.json"), variant)
        if r:
            runs.append(r)
    return runs


def table_gtmean() -> None:
    """Raw vs GT-mean (exposure-normalised) scores on all splits; diagnostic only."""
    b = jload(os.path.join(NEW, "baselines_gtmean_eval.json"))
    rows = []
    names = [("Zero-DCE", "Zero-DCE"), ("SCI", "SCI"), ("HVI-CIDNet", "HVI-CIDNet"), ("DarkIR", "DarkIR"),
             ("URCSA", "URCSA$^{\\dagger}$"), ("STCD", "STCD$^{\\dagger}$"),
             ("Retinexformer", "Retinexformer$^{\\dagger}$"),
             ("Retinexformer-union-seersplit", "Retinexformer (retrained)")]
    for key, label in names:
        r = {}
        if b:
            for sp, k in SEER_KEY.items():
                if k in b["per_seq"] and key in b["per_seq"][k]:
                    merge(r, sp, b["per_seq"][k][key]["gtmean"])
        rows.append((label, key, [r] if r else []))
    rows.append(("Pixel branch w/o V-JEPA", "novjepa", load_gtmean_seer("ctrl_novjepa", "HMP-gtmean")))
    rows.append((r"\textbf{Seer}", "seer", load_gtmean_seer("ctrl_vjepa", "HMP-gtmean")))
    cols = [(sp, m) for sp in SPLITS for m in ("psnr_rgb", "lpips")]
    vals = {k: [agg_seeds(rr, sp, m, who="gtmean-" + k) for sp, m in cols] for _, k, rr in rows}
    marks = {}
    for j, (sp, m) in enumerate(cols):
        for (_, k, _), mk in zip(rows, rank_marks([vals[k][j][0] for _, k, _ in rows], m)):
            marks.setdefault(k, []).append(mk)
    lines = [r"\begin{tabular}{l " + "cc" * len(SPLITS) + "}", r"\toprule",
             "& " + " & ".join(r"\multicolumn{2}{c}{\textbf{%s}}" % sp for sp in SPLITS) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(SPLITS))),
             "Method (GT-mean) & " + " & ".join(["PSNR & LPIPS"] * len(SPLITS)) + r" \\", r"\midrule"]
    for label, k, rr in rows:
        if k == "novjepa":
            lines.append(r"\midrule")
        cells = [decorate(fmt(vals[k][j][0], m, vals[k][j][1]), marks[k][j]) for j, (sp, m) in enumerate(cols)]
        for j, (sp, m) in enumerate(cols):
            macro(f"gm@{k}@{sp}@{m}", fmt(vals[k][j][0], m))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    # paired delta Seer - no V-JEPA under GT-mean
    A, B = load_gtmean_seer("ctrl_novjepa", "HMP-gtmean"), load_gtmean_seer("ctrl_vjepa", "HMP-gtmean")
    cells = []
    for sp, m in cols:
        d, lo, hi = paired_delta(A, B, sp, m)
        if d is None:
            cells.append(TBD)
            continue
        c = ("{:+.2f}" if m == "psnr_rgb" else "{:+.3f}").format(d)
        cells.append(r"\textbf{" + c + "}" if (lo > 0 or hi < 0) else c)
        macro(f"gmdelta@{sp}@{m}", c)
    lines.append(r"\midrule" + "\n" + r"$\Delta$ Seer $-$ w/o V-JEPA & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_gtmean.tex", "\n".join(lines) + "\n")
    PROV.append("GT-mean: baselines_gtmean_eval.json, ctrl_*_s*_gtmean_eval.json")


def table_adapt() -> None:
    """Few-shot adaptation on 4 BVI-RLV *training* scenes; evaluated on the L2 test split and L0."""
    arms = OrderedDict([("novjepa", ("w/o V-JEPA", "ctrl_novjepa", "adapt_novjepa", [0, 1])),
                        ("vjepa", (r"\textbf{Seer}", "ctrl_vjepa", "adapt_vjepa", [0, 2]))])
    steps = [("0", None), ("100", "step_100"), ("200", "final")]
    got = {}
    for key, (label, cprefix, aprefix, seeds) in arms.items():
        for st, ck in steps:
            runs = []
            for s_ in seeds:
                f = (os.path.join(NEW, f"{cprefix}_s{s_}_eval_unified.json") if ck is None
                     else os.path.join(NEW, f"{aprefix}_s{s_}_{ck}_eval_unified.json"))
                r = load_seer_json(f, "HMP-raw")
                if r:
                    runs.append(r)
            got[(key, st)] = runs
    metrics = [("L2", "psnr_rgb"), ("L2", "lpips"), ("L2", "tlp"), ("L0", "psnr_rgb"), ("L0", "lpips")]
    lines = [r"\begin{tabular}{l c ccc cc}", r"\toprule",
             r"& & \multicolumn{3}{c}{\textbf{L2} BVI-RLV test} & \multicolumn{2}{c}{\textbf{L0} (source)} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"Model & steps & PSNR & LPIPS & tLP & PSNR & LPIPS \\", r"\midrule"]
    for key, (label, _, _, _) in arms.items():
        for st, _ in steps:
            cells = []
            for sp, m in metrics:
                mean, std, n = agg_seeds(got[(key, st)], sp, m, who=f"adapt-{key}-{st}")
                cells.append(fmt(mean, m, std))
                macro(f"adapt@{key}@{st}@{sp}@{m}", fmt(mean, m))
            lines.append(f"{label if st == '0' else ''} & {st} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    for st, _ in steps:
        cells = []
        for sp, m in metrics:
            d, lo, hi = paired_delta(got[("novjepa", st)], got[("vjepa", st)], sp, m)
            if d is None:
                cells.append(TBD)
                continue
            c = ("{:+.2f}" if m == "psnr_rgb" else "{:+.3f}").format(d)
            cells.append(r"\textbf{" + c + "}" if (lo > 0 or hi < 0) else c)
            macro(f"adaptdelta@{st}@{sp}@{m}", c)
        lines.append(f"$\\Delta$ Seer $-$ w/o & {st} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_adapt.tex", "\n".join(lines) + "\n")
    PROV.append("adaptation: adapt_{vjepa,novjepa}_s*_{step_100,final}_eval_unified.json + ctrl_* zero-shot")


def table_factorial() -> None:
    """Training-data control on Seer's split, raw outputs.
    Retinexformer is trained from scratch on each split. There is no indoor-only Seer:
    the archived v35_S4_INDOOR_ONLY_ft was initialised from the union-trained Stage 2
    (v35_S3_INDOOR_ONLY log: init v35_stage2/best_y512.pt), so it has seen SDSD-outdoor
    and is deliberately NOT shown. Seer cell: archived v35_stage4 (original code)."""
    rt_in, rt_un = load_retinexformer_retrained("indoor"), load_retinexformer_retrained("union")
    cells = OrderedDict([
        (("Retinexformer", "SDSD-in"), [rt_in] if rt_in else []),
        (("Retinexformer", "SDSD-in+out"), [rt_un] if rt_un else []),
        (("Seer w/o V-JEPA", "SDSD-in+out"), load_seer_runs("ctrl_novjepa", "HMP-raw")),
        (("Seer", "SDSD-in+out"), load_seer_runs("ctrl_vjepa", "HMP-raw")),
    ])
    metrics = [(sp, m) for sp in ("L0", "L1", "L2") for m in ("psnr_rgb", "lpips")]
    lines = [r"\begin{tabular}{ll cccccc}", r"\toprule",
             r"& & \multicolumn{2}{c}{\textbf{L0}} & \multicolumn{2}{c}{\textbf{L1}} & \multicolumn{2}{c}{\textbf{L2}} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Architecture & Training data & PSNR & LPIPS & PSNR & LPIPS & PSNR & LPIPS \\", r"\midrule"]
    for (arch, data), runs in cells.items():
        out = []
        for sp, m in metrics:
            v, sd, n = agg_seeds(runs, sp, m, who=f"{arch}-{data}")
            out.append(fmt(v, m, sd))
            macro(f"fac@{arch}@{data}@{sp}@{m}", fmt(v, m))
        lines.append(f"{arch} & {data} & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_factorial.tex", "\n".join(lines) + "\n")
    PROV.append("data control: Retinexformer_{indoor,union}_seersplit_*; Seer / w/o V-JEPA: ctrl_* runs; "
                "indoor-only Seer excluded (initialised from union-trained Stage 2)")


def table_cp() -> None:
    """Raw vs the three CP calibrations on every split (Seer, final checkpoints)."""
    variants = [("HMP-raw", "Raw prediction", "raw"),
                ("HMP+CPfit", r"CP-src ($\alpha^\star$ on SDSD train clips)", "src"),
                ("HMP+CPxfit", r"CP-tgt ($\alpha^\star$ cross-fitted on target)", "tgt"),
                ("HMP+CP", r"CP, archived $\alpha{=}(.50,.50,.68)$", "arch")]
    runs = {v: load_seer_runs("ctrl_vjepa", v) for v, _, _ in variants}
    head = [r"\begin{tabular}{l " + "cc" * len(SPLITS) + "}", r"\toprule",
            "& " + " & ".join(r"\multicolumn{2}{c}{\textbf{%s}}" % sp for sp in SPLITS) + r" \\",
            "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(SPLITS))),
            "Seer output & " + " & ".join(["PSNR & LPIPS"] * len(SPLITS)) + r" \\", r"\midrule"]
    body = {}
    for v, label, key in variants:
        cells = []
        for sp in SPLITS:
            for m in ("psnr_rgb", "lpips"):
                mean, std, n = agg_seeds(runs[v], sp, m, who=label)
                cells.append(fmt(mean, m, std))
                macro(f"cp{key}@{sp}@{m}", fmt(mean, m))
        body[key] = f"{label} & " + " & ".join(cells) + r" \\"
    tail = [r"\bottomrule", r"\end{tabular}"]
    # main text: the three calibrations defined in Sec. 4.3; appendix adds the archived alpha
    write("tab_cp.tex", "\n".join(head + [body["raw"], body["src"], body["tgt"]] + tail) + "\n")
    write("tab_cp_archived.tex", "\n".join(head + [body["raw"], body["src"], body["tgt"],
                                                  r"\midrule", body["arch"]] + tail) + "\n")
    # fitted alphas, averaged over seeds
    alphas = []
    for s_ in SEEDS:
        d = jload(os.path.join(NEW, f"ctrl_vjepa_s{s_}_eval_unified.json"))
        if d and "cp_alpha_fit" in d:
            alphas.append(d["cp_alpha_fit"]["raw"])
    if alphas:
        a = np.mean(alphas, axis=0)
        macro("cp@alphasrc", "({:.2f}, {:.2f}, {:.2f})".format(*a))


def write_macros() -> None:
    lines = [r"% AUTO-GENERATED by tools/make_tables.py -- do not edit by hand.",
             r"\providecommand{\tbd}{\textcolor{red}{\textbf{TBD}}}",
             r"\providecommand{\R}[1]{\ifcsname res@#1\endcsname\csname res@#1\endcsname\else\tbd\fi}"]
    for k, v in MACROS.items():
        lines.append(r"\expandafter\def\csname res@" + k + r"\endcsname{" + v + "}")
    write("result_macros.tex", "\n".join(lines) + "\n")
    with open(os.path.join(OUT, "PROVENANCE.md"), "w") as f:
        f.write("# Result provenance (auto-generated)\n\n")
        for p in PROV:
            f.write(f"- {p}\n")
        f.write("\n## Warnings\n\n" + ("\n".join(f"- {w}" for w in WARN) or "- none") + "\n")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = build_rows()
    table_main(rows)
    table_extra_metrics(rows)
    table_clean6(rows)
    trows = [r for r in rows if r.key not in ("zerodce", "sci")]
    gtt = load_gt_temporal()
    table_temporal(trows, gtt)
    table_temporal_compact(trows, gtt)
    table_control()
    table_ablation()
    table_encoder()
    table_membership()
    table_cp()
    table_factorial()
    table_cp_transfer()
    table_gtmean()
    table_adapt()
    write_macros()
    n_tbd = sum(v == TBD for v in MACROS.values())
    print(f"{len(MACROS)} macros, {n_tbd} pending (TBD); warnings: {len(WARN)}")
    for w in WARN:
        print("  WARN", w)


if __name__ == "__main__":
    main()
