#!/usr/bin/env bash
# Evaluate a v5 checkpoint on SDSD test set.
# Usage: bash scripts/eval.sh <ckpt_path> [overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <ckpt_path> [overrides...]"
    exit 1
fi
CKPT="$1"; shift
python eval.py --config configs/eval_sdsd.yaml ckpt="${CKPT}" "$@"
