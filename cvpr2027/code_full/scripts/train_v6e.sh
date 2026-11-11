#!/usr/bin/env bash
# v6.E: PMRF — Retinex DiT + CFIM + flow start = V-JEPA(v6.0(LQ)).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/cache/zihaohe/hf_cache}
CUDNN_LIB_DIR="$(python -c 'import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + "/lib")')"
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
python train_v6e.py --config configs/train_v6e.yaml "$@"
