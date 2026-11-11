"""Assemble recorded, unchanged middle-frame crops into the CVPR figure."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / "figures" / "qualitative_cvpr"
METHODS = (("lq", "Low light"), ("retinexformer", "Retinexformer"),
           ("none", "No prior"), ("ours", "Seer"), ("gt", "Ground truth"))


def main():
    metadata = json.loads((SOURCE / "qualitative.json").read_text())
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    assert provenance["panel_seed"] == 0
    assert all(v["mode"] == "saved" for v in provenance["reconstruction"].values())
    margins = json.loads((SOURCE / "selection_margins.json").read_text())
    ordered = sorted(margins, key=margins.get)
    expected = [ordered[round(p / 100 * (len(ordered) - 1))] for p in (10, 35, 60, 85)]
    if metadata["picked"] != expected:
        raise ValueError("recorded sequence selection does not match the stated percentile rule")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "font.size": 7, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(4, 5, figsize=(7.0, 6.05))
    files = []
    for row, (name, percentile) in enumerate(zip(metadata["picked"], (10, 35, 60, 85))):
        safe = name.replace("/", "__")
        for col, (tag, label) in enumerate(METHODS):
            path = SOURCE / f"{safe}__{tag}.png"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == provenance["outputs"][path.name]["sha256"], path
            files.append({"path": str(path.relative_to(PAPER)),
                          "sha256": digest})
            ax = axes[row, col]
            ax.imshow(Image.open(path))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=8, pad=7)
            if tag != "gt":
                value = metadata["per_image"][name][tag]
                ax.set_xlabel(f"{value['psnr_y']:.2f} dB / {value['lpips']:.3f}",
                              fontsize=6.8, labelpad=3)
            else:
                ax.set_xlabel("PSNR-Y / LPIPS", fontsize=6.8, labelpad=3)
            if col == 0:
                short = name.split("/")[-1]
                ax.set_ylabel(f"{percentile}th percentile\n{short}", fontsize=7, labelpad=5)
    fig.subplots_adjust(left=.085, right=.995, top=.955, bottom=.025, hspace=.21, wspace=.045)
    output = PAPER / "figures" / "fig_qualitative_cvpr"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.02)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    (SOURCE / "panel_manifest.json").write_text(json.dumps({
        "source_host": "31051", "source_path": provenance["arguments"]["out"],
        "selection": metadata["selection"], "metrics": "single middle frame, not clip averages",
        "checkpoint_identity": {k: v for k, v in provenance["sources"].items()
                                if k.endswith("checkpoint")},
        "provenance_sha256": hashlib.sha256((SOURCE / "provenance.json").read_bytes()).hexdigest(),
        "files": files,
    }, indent=2) + "\n")
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
