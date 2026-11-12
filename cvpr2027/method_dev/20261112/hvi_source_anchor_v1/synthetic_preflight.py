"""Actual fixed HVI CPU/NPU comparison using synthetic float inputs only."""
import argparse
import datetime
import json
from pathlib import Path
import torch
from cache_hvi_source import (STATE_SHA, digest, load_hvi, predict_float_base,
                              state_digest, tensor_digest, write_new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads((args.release / "config.json").read_text())
    torch.set_num_threads(4)
    low = torch.rand((1, 3, 2, 32, 32), generator=torch.Generator().manual_seed(731), dtype=torch.float32)
    cpu = load_hvi(args.release, torch.device("cpu"))
    output_cpu = predict_float_base(cpu, low, torch.device("cpu"))
    assert state_digest(cpu.state_dict()) == STATE_SHA
    import torch_npu
    device = torch.device("npu:0")
    npu = load_hvi(args.release, device)
    output_npu = predict_float_base(npu, low, device)
    assert state_digest(npu.state_dict()) == STATE_SHA
    differences = (output_cpu - output_npu).abs()
    maximum, mean = float(differences.max()), float(differences.mean())
    assert maximum <= config["synthetic_cpu_npu_max_abs_tolerance"], maximum
    assert mean <= config["synthetic_cpu_npu_mean_abs_tolerance"], mean
    assert output_npu.shape == low.shape and torch.isfinite(output_npu).all()
    write_new(args.out, {"purpose": "Synthetic CPU/NPU strict191-key float comparison; no real inputs or labels",
        "passed": True, "maximum_absolute_difference": maximum, "mean_absolute_difference": mean,
        "input_sha256": tensor_digest(low), "cpu_prediction_sha256": tensor_digest(output_cpu),
        "npu_prediction_sha256": tensor_digest(output_npu), "shape": list(low.shape), "dtype": "float32",
        "both_model_states_unchanged": True, "model_state_sha256": STATE_SHA,
        "config_sha256": digest(args.release / "config.json"), "adapter_sha256": digest(args.release / "cache_hvi_source.py"),
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
