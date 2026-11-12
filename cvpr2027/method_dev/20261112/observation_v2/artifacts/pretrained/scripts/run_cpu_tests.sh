#!/bin/bash
set -eo pipefail
TASK_ROOT=/cache/zihaohe/seer_observation_dev_20261112/method_release_v1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export PYTHONPATH="$TASK_ROOT/release/code_full:$TASK_ROOT/release/code_full/tools:$SEER_ROOT/code/baselines:${PYTHONPATH:-}"
cd "$TASK_ROOT/release/code_full"
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -B -m unittest discover -s tests -v > "$TASK_ROOT/logs/cpu_tests.log" 2>&1
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -B tools/train_observation_v2.py --help > "$TASK_ROOT/logs/train_help.txt" 2>&1
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -B tools/eval_observation_v2.py --help > "$TASK_ROOT/logs/eval_help.txt" 2>&1
python3 -B - <<'PY'
import pathlib,json,re,hashlib
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112/method_release_v1')
s=(r/'logs/cpu_tests.log').read_text();m=re.search(r'Ran (\d+) tests',s)
assert m and int(m[1])>107 and '\nOK\n' in s and 'skipped=' not in s
with (r/'checks/cpu_result.json').open('x') as f:json.dump({'tests':int(m[1]),'skips':0,'status':'PASS','log_sha256':hashlib.sha256(s.encode()).hexdigest()},f)
print(m[0]+'; PASS; no skips',flush=True)
PY
