#!/bin/bash
set -eo pipefail
TASK_ROOT=/cache/zihaohe/seer_observation_dev_20261112/method_release_v1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export PYTHONPATH="$TASK_ROOT/release/code_full:$TASK_ROOT/release/code_full/tools:$SEER_ROOT/code/baselines:${PYTHONPATH:-}"
/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -B - <<'PY'
import pathlib,json,hashlib,os,datetime
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112/method_release_v1')
assert json.loads((r/'checks/cpu_result.json').read_text())['tests']==133
assert json.loads((r/'checks/adjacent_none_received.json').read_text())['all_files_sha_verified']
m=json.loads((r/'release/source_manifest.json').read_text())
for p,h in m['files'].items():assert hashlib.sha256((r/'release/code_full'/p).read_bytes()).hexdigest()==h,p
record={'arm':'none','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_archive_sha256':'6483532a2423d8427b07157e5eca463c378d43c6c51a5b70838f423a3692e996','protocol_sha256':'a57534fa7cd37fb2a81175de89f929830044ae7dcb9f52e71f6c6cb83590a9f2','source_cache':'/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_n_s0','adjacent_cache':str(r.parent/'adjacent_flow/none'),'output':str(r.parent/'prepared/none'),'environment':{k:os.environ.get(k) for k in ['TORCH_DEVICE_BACKEND_AUTOLOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH']},'wrapper_sha256':hashlib.sha256((r/'scripts/run_prepare_none_v1.sh').read_bytes()).hexdigest()}
with (r/'checks/prepare_none_launch.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
PY
/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -B "$TASK_ROOT/release/code_full/tools/train_observation_v2.py" prepare --cache /cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_n_s0 --adjacent /cache/zihaohe/seer_observation_dev_20261112/adjacent_flow/none --splits "$SEER_ROOT/datasets/splits_cvpr.json" --out /cache/zihaohe/seer_observation_dev_20261112/prepared/none
date -u '+END %Y-%m-%dT%H:%M:%SZ'
