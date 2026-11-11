#!/usr/bin/env bash
# V-JEPA invariance probe (GO/NO-GO before training).
set -euo pipefail
cd "$(dirname "$0")/.."
python probe.py --config configs/probe_sdsd.yaml "$@"
echo ""
echo "Probe done. See output_dir/summary.txt"
