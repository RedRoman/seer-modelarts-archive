#!/usr/bin/env bash
# v9.0: V-JEPA-grounded Retinex pixel branch with local illumination cues.
# Per locked spec (G4): no Predictor, no Prompt Bank, no Pixel Parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/cache/zihaohe/hf_cache}
CUDNN_LIB_DIR="$(python -c 'import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + "/lib")')"
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
python train_v9.py --config configs/train_v9.yaml "$@"
