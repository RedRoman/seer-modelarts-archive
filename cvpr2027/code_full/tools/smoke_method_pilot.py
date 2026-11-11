"""One real training example: initialization identity, finite gradients and update."""
from pathlib import Path
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_method_pilot import PlannedSDSD, build_model, state_digest, charbonnier_loss
from models.pixel_branch_fusion import SeerFusionPixelBranch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True)
    p.add_argument("--device", default="npu")
    p.add_argument("--out", required=True)
    p.add_argument("--trainable", choices=["all", "fusion"], default="all")
    args = p.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    torch.set_num_threads(4)
    dev = torch.device(args.device + ":0")
    root = Path(os.environ["SEER_ROOT"])
    ds = PlannedSDSD(root / "datasets/splits_cvpr.json", root, 2, 20261111, 256, 16)
    a, repeat = ds[0], ds[0]
    assert torch.equal(a["lq"], repeat["lq"]) and torch.equal(a["gt"], repeat["gt"])
    model, bb, _, _, _, _, _ = build_model(args.init, dev)
    low, gt = a["lq"][None].to(dev), a["gt"][None].to(dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
        tokens = [z.float() for z in bb(low)["feats"]]
        reference = model(low, tokens)["pred_pixels"]
    checks = {"data_repeat_equal": True, "sample_name": a["name"],
              "trainable": args.trainable, "modes": {}}
    initial = None
    for mode in ("local", "prior"):
        torch.manual_seed(9000)
        candidate = SeerFusionPixelBranch(
            dims=(20, 40, 80, 160, 320), num_blocks=(2, 4, 4, 6, 8), heads=(2, 4, 4, 8, 8),
            vjepa_grid=(8, 16, 16), use_gradient_checkpoint=True, fusion_mode=mode,
            fusion_scale=3, fusion_radius=2, fusion_offset=2, fusion_tap=2)
        missing, unexpected = candidate.load_state_dict(model.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("temporal_fusion.") for k in missing)
        fingerprint = state_digest(candidate.temporal_fusion.state_dict())
        if initial is not None:
            assert fingerprint == initial
        initial = fingerprint
        candidate = candidate.to(dev).eval()
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            prediction = candidate(low, tokens)["pred_pixels"]
        max_error = float((prediction - reference).abs().max())
        assert max_error == 0., max_error
        if args.trainable == "fusion":
            for name, parameter in candidate.named_parameters():
                parameter.requires_grad_(name.startswith("temporal_fusion."))
        base_before = state_digest({k: v for k, v in candidate.state_dict().items()
                                    if not k.startswith("temporal_fusion.")})
        candidate.train()
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            result = candidate(low, tokens)
        loss = charbonnier_loss(result["pred_pixels"].float(), gt)
        loss.backward()
        grad = candidate.temporal_fusion.output_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0
        grad_norm = float(grad.float().norm())
        optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad],
                                      lr=2e-4)
        optimizer.step()
        assert torch.isfinite(candidate.temporal_fusion.output_proj.weight).all()
        updated = state_digest(candidate.temporal_fusion.state_dict())
        assert updated != fingerprint, "Fusion parameters did not update"
        base_after = state_digest({k: v for k, v in candidate.state_dict().items()
                                   if not k.startswith("temporal_fusion.")})
        if args.trainable == "fusion":
            assert base_after == base_before, "Frozen restoration state changed"
            assert all(p.grad is None for n, p in candidate.named_parameters()
                       if not n.startswith("temporal_fusion."))
        checks["modes"][mode] = {"initial_prediction_max_error": max_error,
                                 "fusion_initial_sha256": fingerprint,
                                 "fusion_updated_sha256": updated,
                                 "base_state_before_sha256": base_before,
                                 "base_state_after_sha256": base_after,
                                 "first_loss": float(loss), "output_projection_grad_norm": grad_norm}
        del candidate, optimizer, result, prediction, loss, grad
    Path(args.out).write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
