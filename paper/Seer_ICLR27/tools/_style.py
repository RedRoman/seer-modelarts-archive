"""Shared matplotlib style for Seer paper figures.

Sanitized for AAAI style: no Type-3 fonts, no external font files, single sans
family, minimal chartjunk. All figures share the same palette.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "vjepa":   "#6E90B8",  # dusty blue — Seer / video world model
    "clip":    "#B4A69A",  # warm gray  — image-level prior
    "dino":    "#D6A15A",  # soft amber — the third comparator
    "seer":    "#6E90B8",  # dusty blue — main model
    "seer_cp": "#D6A15A",  # soft amber — full model with chroma preserve
    "baseline": "#B4A69A",  # warm gray — baselines
    "highlight": "#4E8B6A",  # deep mint — highlight for chosen point
    "reference": "#9A9A9A",  # medium gray — reference lines
    "ink":      "#1B1B1B",  # near-black — text and ink
}


def apply_style():
    """Apply the shared publication style."""
    mpl.rcParams.update({
        # fonts: TrueType embedded (Type 3 forbidden by AAAI kit)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        # sizes tuned for column-width printing (fig will be ~3.3 in wide)
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # geometry
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # remove decorations
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        # tight bounding
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 300,
    })
