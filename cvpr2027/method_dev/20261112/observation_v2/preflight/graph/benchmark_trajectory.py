"""Synthetic CPU performance only; does not evaluate real data or model quality."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import time

import torch

ROOT = Path(__file__).resolve().parent
source = ROOT / "models/trajectory_reconstruction.py"
spec = importlib.util.spec_from_file_location("trajectory_benchmark", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(20261112)
report = {
    "purpose": "Synthetic CPU timing only, no research performance results",
    "torch": torch.__version__, "threads": torch.get_num_threads(),
    "backend_autoload": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "batch": 1, "frames": 16, "supervised_frames": list(range(0, 16, 2)),
    "warmup_steps": 2, "timed_steps": 3, "optimizer": "Adam lr=0.001",
    "mode": "conditioned", "results": [],
}
for size in (64, 128, 256):
    gc.collect()
    model = module.TrajectoryReconstruction()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    base = torch.rand(1, 3, 16, size, size)
    corrections = torch.randn(1, 2, 3, 16, size, size) * .04
    features = torch.randn(1, 16, 19, size // 8, size // 8)
    details = torch.randn_like(base) * .08
    flow = torch.full((1, 2, 2, 16, size, size), .25)
    mask = torch.ones(1, 2, 1, 16, size, size)
    mask[:, 0, :, 0] = 0
    mask[:, 1, :, -1] = 0
    mask[..., -1, :] = 0
    mask[..., -1] = 0
    target = torch.rand_like(base)
    timings = []
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        predicted = model(base, corrections, features, details, flow, mask)["pred_pixels"]
        loss = (predicted[:, :, ::2] - target[:, :, ::2]).square().mean()
        forward = time.perf_counter()
        loss.backward()
        optimizer.step()
        end = time.perf_counter()
        if step >= 2:
            timings.append({"forward_s": forward - start, "backward_optimizer_s": end - forward,
                            "total_s": end - start})
        print(json.dumps({"size": size, "step": step, "elapsed_s": end - start}), flush=True)
    result = {"size": size, "timings": timings,
              "median_step_s": statistics.median(x["total_s"] for x in timings),
              "mean_step_s": statistics.mean(x["total_s"] for x in timings),
              "peak_process_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "parameters": sum(x.numel() for x in model.parameters())}
    report["results"].append(result)
    (ROOT / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
    del model, optimizer, base, corrections, features, details, flow, mask, target, predicted, loss
print("BENCHMARK_COMPLETE", flush=True)
