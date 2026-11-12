#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
TASK_ROOT=/cache/zihaohe/seer_observation_dev_20261112/method_release_v1
PHASE_ROOT=/cache/zihaohe/seer_observation_dev_20261112
export PYTHONPATH="$TASK_ROOT/release/code_full:$TASK_ROOT/release/code_full/tools:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export ASCEND_RT_VISIBLE_DEVICES=7
cd "$TASK_ROOT/release/code_full"
"$SEER_PYTHON" -B - <<'PY'
import pathlib,json,hashlib,subprocess,datetime,os,importlib.metadata
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112');task=r/'method_release_v1'
assert not (r/'results/bvi_val16_none_observation_v2.json').exists()
m=json.loads((task/'release/source_manifest.json').read_text())
for p,h in m['files'].items():assert hashlib.sha256((task/'release/code_full'/p).read_bytes()).hexdigest()==h,p
h=json.loads((task/'checks/heads_none_receipt.json').read_text());assert h['n_heads']==21
for p,v in h['files'].items():assert hashlib.sha256((r/'target_inputs/none'/p).read_bytes()).hexdigest()==v['sha256'],p
import eval_e4
import common.eval_lib as common
mp=pathlib.Path(common.__file__).resolve()
assert hashlib.sha256(mp.read_bytes()).hexdigest()=='a26ff0c81f0e6fa380ca26ac87dc2197c66248d3b6945e0da1c99e5b3d3f451d'
record={'arm':'none','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'31051','physical_npu':7,'source_archive_sha256':'6483532a2423d8427b07157e5eca463c378d43c6c51a5b70838f423a3692e996','source_manifest_sha256':hashlib.sha256((task/'release/source_manifest.json').read_bytes()).hexdigest(),'heads_archive_sha256':h['archive_sha256'],'metric_file':str(mp),'metric_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'versions':{k:importlib.metadata.version(k) for k in ['torch','torch-npu','pyiqa','numpy','torchvision']},'python':os.environ['SEER_PYTHON'],'wrapper_sha256':hashlib.sha256((task/'scripts/run_eval_none_31051_v1.sh').read_bytes()).hexdigest(),'runtime_env_sha256':hashlib.sha256(pathlib.Path('/temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh').read_bytes()).hexdigest(),'source_protocol_sha256':'a57534fa7cd37fb2a81175de89f929830044ae7dcb9f52e71f6c6cb83590a9f2','scope':'All31 fixed methods; VAL16 only; final1000; no tuning/selection','environment':{k:os.environ.get(k) for k in ['ASCEND_RT_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','ASCEND_HOME_PATH','LD_LIBRARY_PATH']}}
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
with (task/'checks/npu_before_eval_none_31051.txt').open('x') as f:f.write(smi)
assert 'No running processes found in NPU 7' in smi
with (task/'checks/eval_none_31051_launch.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
PY
"$SEER_PYTHON" -B tools/eval_observation_v2.py --init /temp/zihaohe/seer_cvpr/method_dev_20261111/inputs/e17_small_none_s0_final_cpu.pt --raft-weights "$RAFT_WEIGHTS" --prepared-manifest "$PHASE_ROOT/target_inputs/none/prepared/none/prepared_manifest.json" --heads "$PHASE_ROOT/target_inputs/none/runs/none/"* --v1-references "$PHASE_ROOT/inputs/v1_references/none/raw_detail_s0" "$PHASE_ROOT/inputs/v1_references/none/raw_detail_s1" "$PHASE_ROOT/inputs/v1_references/none/raw_detail_s2" --seer-root "$SEER_ROOT" --device npu --out "$PHASE_ROOT/results/bvi_val16_none_observation_v2.json"
npu-smi info > "$TASK_ROOT/checks/npu_after_eval_none_31051.txt"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
