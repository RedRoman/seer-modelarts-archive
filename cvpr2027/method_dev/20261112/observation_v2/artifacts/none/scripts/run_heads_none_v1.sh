#!/bin/bash
set -eo pipefail
TASK_ROOT=/cache/zihaohe/seer_observation_dev_20261112/method_release_v1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export PYTHONPATH="$TASK_ROOT/release/code_full:$TASK_ROOT/release/code_full/tools:$SEER_ROOT/code/baselines:${PYTHONPATH:-}"
python3 -B - <<'PY'
import time,pathlib
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112')
log=r/'method_release_v1/logs/prepare_none_v1.log'
deadline=time.monotonic()+1800
print('Waiting for successful complete prepare136; no partial cache training',flush=True)
while True:
 text=log.read_text()
 if 'Traceback (most recent call last)' in text:raise RuntimeError('Prepare failed; no training launched')
 if '\nEND ' in text:
  assert (r/'prepared/none/prepared_manifest.json').is_file()
  break
 if time.monotonic()>deadline:raise TimeoutError('Prepare not complete after30min; no training launched')
 time.sleep(5)
PY
npu-smi info > "$TASK_ROOT/checks/npu_before_cpu_training_none.txt"
/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -B "$TASK_ROOT/scripts/run_source_heads_v1.py" none
npu-smi info > "$TASK_ROOT/checks/npu_after_cpu_training_none.txt"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
