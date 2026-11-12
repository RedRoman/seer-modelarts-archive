#!/bin/bash
set -euo pipefail
task_root=/cache/zihaohe/seer_predictive_credit_probe_20261112_v1
release_root="$task_root/smoke_release_v1"
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export ASCEND_RT_VISIBLE_DEVICES=6
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
npu-smi info > "$task_root/checks/critic_npu_before.txt"
python3 - <<'PY'
from pathlib import Path
text=Path('/cache/zihaohe/seer_predictive_credit_probe_20261112_v1/checks/critic_npu_before.txt').read_text()
if 'No running processes found in NPU 6' not in text:
    raise RuntimeError('Physical NPU6 is occupied; refusing to modify other processes')
PY
cd "$release_root"
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -m unittest -v test_critic_export > "$task_root/checks/critic_cpu_tests.log" 2>&1
/cache/zihaohe/seer_cvpr/venv_npu/bin/python critic_export.py \
  --release "$release_root" --mode smoke --output "$task_root/checks/critic_synthetic_smoke.json" \
  > "$task_root/checks/critic_synthetic_smoke.log" 2>&1
npu-smi info > "$task_root/checks/critic_npu_after_smoke.txt"
