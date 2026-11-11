#!/usr/bin/env bash
set -euo pipefail
cd /home/ma-user/workspace/zihaohe/vjepa_llve
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/cache/zihaohe/hf_cache}
CUDNN_LIB_DIR="$(python -c 'import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + "/lib")')"
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
stdbuf -oL -eL python -u tools/audit_llvestcd.py 2>&1 | tee /cache/zihaohe/checkpoint/audit_llvestcd_run.log
