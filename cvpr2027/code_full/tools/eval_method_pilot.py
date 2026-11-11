"""Reload a self-contained method-pilot checkpoint and score frozen validation."""
from pathlib import Path
import argparse
import sys

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from train_method_pilot import digest, evaluate, held_out_sequences, write_json
from eval_e4 import SPLITS, VJEPA_CKPT, VJEPA_REPO
from models.pixel_branch_v2 import SeerV2PixelBranch
from models.pixel_branch_fusion import SeerFusionPixelBranch
from models.vjepa_backbone import VJepaBackboneLocal21


def load_pilot(path, dev):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    metadata = saved["provenance"]
    base = metadata["base_reconstruction"]
    if (base["arch"] != "v2" or base["branch"] != "small" or base["arm"] != "pretrained"
            or base["token_proj"] != "none" or base["clip_len"] != 16):
        raise ValueError("Unsupported pilot base; this loader does not silently fall back")
    variant = saved["args"]["variant"]
    fusion = metadata["fusion_config"]
    if (variant == "baseline") != (fusion is None):
        raise ValueError("Checkpoint variant and fusion configuration disagree")
    kwargs = dict(dims=(20, 40, 80, 160, 320), num_blocks=(2, 4, 4, 6, 8),
                  heads=(2, 4, 4, 8, 8), vjepa_grid=(8, 16, 16),
                  use_gradient_checkpoint=False, xattn=base["xattn"],
                  kv_mode=base["kv"], layer_mode=base["layers"])
    if fusion is None:
        model = SeerV2PixelBranch(**kwargs)
    else:
        if fusion["fusion_mode"] != variant:
            raise ValueError("Saved guidance mode does not match the checkpoint variant")
        model = SeerFusionPixelBranch(**kwargs, **fusion)
    model.load_state_dict(saved["pixel_branch"], strict=True)
    model = model.to(dev).eval()
    backbone = VJepaBackboneLocal21(
        local_repo=VJEPA_REPO, local_ckpt=VJEPA_CKPT, feature_layers=tuple(base["taps"]),
        image_size=256, frames_per_clip=16, patch_size=16, tubelet_size=2, freeze=True).to(dev).eval()
    return model, backbone, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "npu"], default="npu")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.device == "npu":
        import torch_npu  # noqa: F401
    torch.set_num_threads(4)
    dev = torch.device(args.device + ":0")
    model, backbone, saved = load_pilot(args.checkpoint, dev)
    split_sha = digest(SPLITS)
    if split_sha != saved["provenance"]["split_sha256"]:
        raise ValueError("Current validation split differs from the training provenance")
    domain = saved["args"].get("domain", "sdsd")
    clips = list(held_out_sequences(domain, "val", 16, 256))
    result = evaluate(model, backbone, clips, dev, saved["step"], saved["args"]["variant"], domain)
    result["reload_verification"] = {"checkpoint_sha256": digest(args.checkpoint),
                                     "evaluation_source_sha256": digest(__file__),
                                     "split_sha256": split_sha,
                                     "base_reconstruction": saved["provenance"]["base_reconstruction"],
                                     "fusion_config": saved["provenance"]["fusion_config"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(result["metrics"])


if __name__ == "__main__":
    main()
