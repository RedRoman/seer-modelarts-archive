#!/bin/bash
set -euo pipefail
task_root=/cache/zihaohe/seer_color_contract_20261112
release_root="$task_root/export_release_v1"
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
npu-smi info > "$task_root/checks/export_npu_before.txt"
python3 - <<'PY'
from pathlib import Path
text=Path('/cache/zihaohe/seer_color_contract_20261112/checks/export_npu_before.txt').read_text()
if 'No running processes found in NPU 7' not in text:
    raise RuntimeError('Physical NPU 7 is occupied; refusing export without touching any process')
PY
cd "$release_root"
/cache/zihaohe/seer_cvpr/venv_npu/bin/python cache_hvi_rgb_source.py \
  --release "$release_root" --mode export --device npu \
  --bgr-evidence /cache/zihaohe/seer_source_input_audit_20261112_v2/source_color_receipt_closed.json \
  --report "$task_root/checks/export_outputs_verification.json" \
  > "$task_root/checks/export_source.log" 2>&1
npu-smi info > "$task_root/checks/export_npu_after.txt"
