#!/bin/bash
set -e
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TASKROOT=/cache/zihaohe/seer_observation_memory_20261112
TASKCODE="$TASKROOT/release_v2/code_full"
TASKPY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
export PYTHONPATH="$TASKCODE:$TASKCODE/tools:${PYTHONPATH:-}"
cd "$TASKCODE"
"$TASKPY" - <<'INNER'
import subprocess,json,hashlib,os,time,importlib.metadata
from pathlib import Path
r=Path('/cache/zihaohe/seer_observation_memory_20261112')
s=subprocess.check_output(['npu-smi','info'],text=True)
(r/'checks/export_v2_npu_before.txt').write_text(s)
assert 'No running processes found in NPU 6 ' in s,'Device6 is occupied; refusing allocation'
m=json.loads((r/'release_v2/source_manifest.json').read_text())
assert all(hashlib.sha256((r/'release_v2/code_full'/p).read_bytes()).hexdigest()==h for p,h in m.items())
x={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'device_physical':6,'device_logical':0,'source_manifest_sha256':hashlib.sha256((r/'release_v2/source_manifest.json').read_bytes()).hexdigest(),'exporter_sha256':m['tools/cache_observation_memory.py'],'environment':{k:os.environ.get(k) for k in ('TORCH_DEVICE_BACKEND_AUTOLOAD','ASCEND_RT_VISIBLE_DEVICES','PYTHONPATH','LD_LIBRARY_PATH','ASCEND_HOME_PATH','SEER_ROOT','OMP_NUM_THREADS')},'versions':{k:importlib.metadata.version(k) for k in ('torch','torch-npu','numpy')},'target_inference':False}
with (r/'checks/export_v2_launch.json').open('x') as f:json.dump(x,f,indent=2)
INNER
if "$TASKPY" tools/cache_observation_memory.py --cache /cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0 --out "$TASKROOT/tokens_source_v1" --seer-root /temp/zihaohe/seer_cvpr --device npu; then status=0; else status=$?; fi
npu-smi info > "$TASKROOT/checks/export_v2_npu_after.txt"
printf '%s\n' "$status" > "$TASKROOT/checks/export_v2_exit_code.txt"
exit "$status"
