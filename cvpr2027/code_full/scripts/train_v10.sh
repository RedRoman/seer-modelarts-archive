#!/usr/bin/env bash
# Launch script for V-JEPA-LLVE v10 first training run.
# Bakes in the CN-F cuDNN library-resolution fix so backward never falls back
# to the system's incomplete libcudnn_cnn_infer.so.8.
#
# Spec: V10_DESIGN_SPEC.md (CU-approved 2026-05-03).
# Config: configs/train_v10.yaml (lr=2e-5, accum=4, total_iters=5000).
set -euo pipefail

REPO="/home/ma-user/workspace/zihaohe/vjepa_llve"
CONDA_ENV="/cache/zihaohe/conda_envs/vjepa2-pip"
CUDNN_DIR="${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cudnn/lib"

# CN-F fix: prepend conda's bundled cuDNN so all sublibs resolve from one wheel.
export LD_LIBRARY_PATH="${CUDNN_DIR}:${LD_LIBRARY_PATH:-}"

# Default to GPU 0 unless caller overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Activate env (login shells only — tmux usually inherits).
source "/home/ma-user/miniconda3/bin/activate" "${CONDA_ENV}"

cd "${REPO}"

CONFIG="${1:-configs/train_v10.yaml}"
shift || true

echo "[train_v10.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train_v10.sh] LD_LIBRARY_PATH (head): ${LD_LIBRARY_PATH%%:*}"
echo "[train_v10.sh] config: ${CONFIG}"
echo "[train_v10.sh] extra args: $*"

exec python train_v10.py --config "${CONFIG}" "$@"
