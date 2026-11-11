#!/usr/bin/env bash
# v9.0b: PSNR-targeted finetune from v9.0/best.pt. R5/R8/Q1 lock spec.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/cache/zihaohe/hf_cache}
CUDNN_LIB_DIR="$(python -c 'import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + "/lib")')"
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
python train_v9_0b.py --config configs/train_v9_0b.yaml "$@"
