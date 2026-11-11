"""Frames for the qualitative figure, from the same evaluator as the tables.

The ICLR-era `dump_qualitative.py` runs the old 512-crop stack with the old
baselines and the pre-de-leak split, so its numbers would not match this
paper's tables. This one goes through `eval_e4` and `eval_baseline` so a number
printed under a crop is the same number the table reports for that sequence.

Sequences are chosen by a stated rule rather than by eye: the test sequences at
the 10th, 35th, 60th and 85th percentile of our margin over the retrained
Retinexformer. That deliberately includes a sequence we lose on, because the
margin's median is near zero and a figure made of wins would misrepresent the
distribution the paper reports.
"""
from __future__ import annotations

import argparse, json, os, sys, hashlib, datetime, socket, importlib.util
import numpy as np
import torch
from PIL import Image

CODE = "/cache/zihaohe/seer_takeover_20261111/code_full"
for p in (CODE, os.path.join(CODE, "tools"),
          "/temp/zihaohe/seer_cvpr/code/baselines"):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_e4 import (build_model, held_out_sequences, run_clip,  # noqa: E402
                     psnr_y_uint8, lpips_of)
BASELINE_LOADER = "/cache/zihaohe/seer_cvpr/qual_regen/eval_baseline.py"
_baseline_spec = importlib.util.spec_from_file_location("verified_baseline_loader", BASELINE_LOADER)
EB = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(EB)
assert os.path.realpath(sys.modules["eval_e4"].__file__) == os.path.realpath(os.path.join(CODE, "tools/eval_e4.py"))

PCTS = (10, 35, 60, 85)


def save(arr_u8, path):
    Image.fromarray(arr_u8).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours", required=True, help="converged prior checkpoint")
    ap.add_argument("--none", required=True, help="converged no-prior checkpoint")
    ap.add_argument("--retinexformer", required=True, help="retrained .pth")
    ap.add_argument("--margins", required=True,
                    help="JSON with our and Retinexformer per-sequence psnr_y")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    torch.set_num_threads(4)
    if a.device.startswith("npu"):
        import torch_npu  # noqa: F401
    dev = torch.device(a.device)
    rf_dev = torch.device("cpu")  # Retinexformer was only ever run on CUDA; CPU avoids unsupported NPU ops
    amp = torch.bfloat16

    M = json.load(open(a.margins))
    order = sorted(M.items(), key=lambda kv: kv[1])
    picks = []
    for pc in PCTS:
        i = int(round(pc / 100 * (len(order) - 1)))
        if order[i][0] not in picks:
            picks.append(order[i][0])
    print("picked by margin percentile %s:" % (PCTS,))
    for n in picks:
        print("  %-28s margin %+.3f dB" % (n, M[n]))

    models = {}
    for tag, ck in (("ours", a.ours), ("none", a.none)):
        models[tag] = build_model(ck, dev, reconstruction="saved")
    rf_model, rf_mult = EB.LOADERS["retinexformer"](a.retinexformer, rf_dev)

    rows = {}
    for name, lq, gt, kind in held_out_sequences("sdsd", "test", 16, a.crop):
        if name not in picks:
            continue
        mid = lq.shape[0] // 2
        out = {"lq": lq, "gt": gt}
        for tag in ("ours", "none"):
            pb, bb, refiner, exp_head, arm, clip_len, _ = models[tag]
            out[tag] = run_clip(pb, bb, refiner, exp_head, arm, lq,
                                clip_len, dev, amp)
        out["retinexformer"] = EB.run_frames(rf_model, rf_mult, lq, rf_dev)

        rows[name] = {}
        safe = name.replace("/", "__")
        for tag, img in out.items():
            f = img[mid]
            save(f, os.path.join(a.out, f"{safe}__{tag}.png"))
            if tag != "gt":
                # lpips_of wants (B,3,H,W) float tensors in [0,1], the same
                # form eval_e4 feeds it, so the figure's numbers and the
                # table's come from identical inputs
                def to_t(u8):
                    return (torch.from_numpy(np.ascontiguousarray(u8[None]))
                            .permute(0, 3, 1, 2).float().div(255.))
                rows[name][tag] = {
                    "psnr_y": psnr_y_uint8(f, gt[mid]),
                    "lpips": lpips_of(to_t(f), to_t(gt[mid]), dev)}
        print("%-28s " % name + "  ".join(
            "%s %.2f/%.3f" % (t, v["psnr_y"], v["lpips"])
            for t, v in rows[name].items()))

    meta = {"crop": a.crop, "frame": "middle of the centre 16-frame clip",
            "selection": "test sequences at margin percentiles %s over the "
                         "retrained Retinexformer" % (PCTS,),
            "picked": picks, "per_image": rows}
    json.dump(meta, open(os.path.join(a.out, "qualitative.json"), "w"), indent=1)

    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return {"path": os.path.realpath(path), "bytes": os.stat(path).st_size, "sha256": h.hexdigest()}
    sources = {
        "ours_checkpoint": a.ours, "none_checkpoint": a.none, "retinexformer_checkpoint": a.retinexformer,
        "margins": a.margins, "generator": __file__, "baseline_loader": BASELINE_LOADER,
        "selection_provenance": os.path.join(os.path.dirname(a.margins), "qualitative_selection_provenance.json"),
        "frozen_encoder_checkpoint": os.path.join(os.environ["SEER_ROOT"], "weights/vjepa2_1_vitl_dist_vitG_384.pt"),
        "eval_e4": os.path.join(CODE, "tools/eval_e4.py"),
        "checkpoint_config": os.path.join(CODE, "tools/checkpoint_config.py"),
        "pixel_branch_v2": os.path.join(CODE, "models/pixel_branch_v2.py"),
        "retinexformer_arch": "/temp/zihaohe/seer_cvpr/code/ext_methods/Retinexformer/basicsr/models/archs/RetinexFormer_arch.py",
        "splits_cvpr": os.path.join(os.environ["SEER_ROOT"], "datasets/splits_cvpr.json"),
    }
    provenance = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "arguments": vars(a),
        "selection_basis": "Three-seed mean prior PSNR-Y minus retrained Retinexformer, 39 test sequences",
        "panel_seed": 0, "cpu_threads": 4, "retinexformer_device": "cpu",
        "npu_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "reconstruction": {tag: models[tag][0].eval_reconstruction for tag in ("ours", "none")},
        "sources": {key: digest(path) for key, path in sources.items()},
        "outputs": {name: digest(os.path.join(a.out, name)) for name in sorted(os.listdir(a.out)) if name.endswith((".png", ".json"))},
        "source_modifications": "Isolated saved evaluator import, explicit checkpoint reconstruction, four CPU threads, and provenance recording. Selection and frame generation logic retained.",
    }
    with open(os.path.join(a.out, "provenance.json"), "x") as f:
        json.dump(provenance, f, indent=2)

    print("\nwrote", os.path.join(a.out, "qualitative.json"))


if __name__ == "__main__":
    main()
