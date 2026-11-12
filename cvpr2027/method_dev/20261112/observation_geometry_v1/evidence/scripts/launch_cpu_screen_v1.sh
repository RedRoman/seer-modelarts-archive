#!/bin/bash
set -u
set -o noclobber
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export SEER_ROOT=/temp/zihaohe/seer_cvpr
ROOT=/cache/zihaohe/seer_observation_geometry_20261112
/cache/zihaohe/seer_cvpr/venv_npu/bin/python "$ROOT/scripts/preflight_and_launch_v1.py" > "$ROOT/launch_v1.log" 2>&1
result=$?
printf '%s\n' "$result" > "$ROOT/launch_exit_v1.txt"
exit "$result"
