#!/bin/bash
set -eo pipefail
TASK_ROOT=/cache/zihaohe/seer_observation_dev_20261112/tempretinex_reference_v1
RUNTIME_ROOT=/cache/zihaohe/seer_observation_dev_20261112/runtime_overlay_v1
source <(sed "s|/cache/hanbin/ascend_private/cann-8.5.2|$RUNTIME_ROOT/private_cann/cann-8.5.2|g" "$RUNTIME_ROOT/private_cann/cann-8.5.2/set_env.sh")
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="$RUNTIME_ROOT/overlay/torch_home"
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:$RUNTIME_ROOT/overlay/site-packages:$SEER_ROOT/code/baselines:$SEER_ROOT/code/baselines/common:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=1
npu-smi info > "$TASK_ROOT/checks/npu_before_v1.txt"
/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -B - <<'PY'
import pathlib,json,hashlib,datetime,os
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112/tempretinex_reference_v1')
assert 'No running processes found in NPU 1' in (r/'checks/npu_before_v1.txt').read_text()
m=json.loads((r/'source_manifest.json').read_text())
for p,h in m['files'].items():assert hashlib.sha256((r/'code_full'/p).read_bytes()).hexdigest()==h,p
assert hashlib.sha256((r/'checks/REFERENCE_PROTOCOL.md').read_bytes()).hexdigest()=='88ba65c293cb25253de65bd83c4b1f4b32480f070864ec409a4d2fd382c3d1ec'
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_npu':1,'source_archive_sha256':'bb6281cfcb71b27c265c8feff833b5b035588cbc377bda6eb12612f43d14e2d4','reference_protocol_sha256':'88ba65c293cb25253de65bd83c4b1f4b32480f070864ec409a4d2fd382c3d1ec','wrapper_sha256':hashlib.sha256((r/'scripts/run_reference_v1.sh').read_bytes()).hexdigest(),'environment':{k:os.environ.get(k) for k in ['ASCEND_RT_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','ASCEND_HOME_PATH','LD_LIBRARY_PATH','TORCH_HOME']}}
with (r/'checks/launch_v1.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v),flush=True)
PY
/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -B "$TASK_ROOT/scripts/npu_smoke_v1.py"
/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -B "$TASK_ROOT/code_full/tools/eval_tempretinex_reference.py" --upstream-root "$TASK_ROOT/inputs" --seer-root "$SEER_ROOT" --device npu --out "$TASK_ROOT/results/bvi_val16_did_reference.json"
npu-smi info > "$TASK_ROOT/checks/npu_after_v1.txt"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
