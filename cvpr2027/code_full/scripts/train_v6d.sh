#!/usr/bin/env bash
# v6.D: Retinex Latent DiT + CFIM + EMA + sliding-best eval.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/cache/zihaohe/hf_cache}
# Force conda env's nvidia-cudnn-cu12 to win over the (broken) system cuDNN.
CUDNN_LIB_DIR="$(python -c 'import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + "/lib")')"
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
python train_v6d.py --config configs/train_v6d.yaml "$@"
