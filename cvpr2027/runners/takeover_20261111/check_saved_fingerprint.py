import json
import sys
import torch
torch.set_num_threads(4)
sys.path.insert(0, "/cache/zihaohe/seer_takeover_20261111/code_full/tools")
from eval_e4 import build_model
checkpoint = "/cache/zihaohe/seer_cvpr/runs_e17/e17_small_random_s0/final_cpu.pt"
pb, bb, *_ = build_model(checkpoint, torch.device("cpu"), reconstruction="saved")
metadata = pb.eval_reconstruction
print(json.dumps(metadata, indent=2), flush=True)
observed = metadata["random_encoder_fingerprint"]
expected = 1016.862549
if abs(observed - expected) > 0.00001:
    raise RuntimeError(f"fingerprint mismatch: {observed} vs {expected}")
print("FINGERPRINT_MATCH", flush=True)
