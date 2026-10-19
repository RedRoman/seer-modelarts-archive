#!/usr/bin/env python3
"""Print HBM MiB in use on each NPU (chip row of npu-smi info)."""
import re, subprocess, sys
txt = subprocess.run(["npu-smi", "info"], capture_output=True, text=True).stdout
cur, out = None, {}
for ln in txt.splitlines():
    m = re.match(r"\|\s+(\d+)\s+\|\s+910B3", ln) or re.match(r"\|\s+(\d+)\s+910B3", ln)
    if m:
        cur = int(m.group(1)); continue
    if cur is not None:
        m2 = re.search(r"(\d+)\s*/\s*(\d+)\s*\|?\s*$", ln.rstrip("|").rstrip())
        if m2 and int(m2.group(2)) > 1000:
            out[cur] = int(m2.group(1)); cur = None
if len(sys.argv) > 1:
    print(out.get(int(sys.argv[1]), -1))
else:
    for k in sorted(out): print(f"NPU{k} {out[k]} MiB")
