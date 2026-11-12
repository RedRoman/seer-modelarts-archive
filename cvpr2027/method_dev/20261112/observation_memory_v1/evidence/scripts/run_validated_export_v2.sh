#!/bin/bash
set -e
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
R=/cache/zihaohe/seer_observation_memory_20261112
P=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
cd "$R/release_v2/code_full"
"$P" -m unittest discover -s tests -p test_cache_observation_memory.py -v > "$R/checks/export_cpu_tests_v2.log" 2>&1
"$P" "$R/scripts/cpu_real_npu_load_v2.py" > "$R/checks/cpu_real_npu_load_v2.log" 2>&1
printf 'preflight PASS\n'
bash "$R/scripts/run_export_v2.sh" > "$R/logs/export_tokens_v2.log" 2>&1
