#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/cache/hanbin/ascend_private/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd
SOURCE_ROOT=/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export ASCEND_RT_VISIBLE_DEVICES=''
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export SEER_SOURCE_FAMILY="$1"
export SEER_SELECTOR_SEED="$2"
cd "$TASK_ROOT/code_full"
"$SEER_PYTHON" -B - <<'CHECK'
import pathlib,hashlib,json,os,datetime,platform,sys,torch
parameter=torch.nn.Parameter(torch.ones(2));optimizer=torch.optim.Adam([parameter],lr=1e-3);parameter.square().sum().backward();optimizer.step();assert parameter.device.type=='cpu' and 'torch_npu' not in sys.modules
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd')
cache=root.parents[1]/'cache_sdsd_n_s0'
family=os.environ['SEER_SOURCE_FAMILY'];assert family in ('restored','restored_detail','raw_detail')
summary=json.loads((cache/'cache_summary.json').read_text())
assert summary['n']==136 and summary['oracle_computed'] is False
assert summary['base_state_unchanged'] is True and summary['raft_state_unchanged'] is True
m=json.loads((root/'source_manifest.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((root/'code_full'/n).read_bytes()).hexdigest()==h,n
seed=int(os.environ['SEER_SELECTOR_SEED']);assert seed in (0,1,2)
assert not (root.parents[1]/'runs_none_cpu_v2'/f'{family}_s{seed}').exists()
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':platform.node(),'source_archive_sha256':'c65c0b4919cd8befdd850f9f96e5ce35565a45c3b3f5335ae2c50d00bebac14c','wrapper_sha256':hashlib.sha256((root/'tools/run_none_train_cpu_v2_c65c0b.sh').read_bytes()).hexdigest(),'cache_provenance_sha256':hashlib.sha256((cache/'provenance.json').read_bytes()).hexdigest(),'cache_summary_sha256':hashlib.sha256((cache/'cache_summary.json').read_bytes()).hexdigest(),'family':family,'seed':seed,'device':'cpu','threads':4,'steps':1000,'batch_frames':8,'lr':1e-3,'modes':['direct_mse','direct_mse_normalized'],'queue':'Nine independent family x seed jobs concurrent; 4 CPU threads each','scope':'Source TRAIN136 only; final internal source holdout once; no target data','environment':{k:os.environ.get(k) for k in ['TORCH_DEVICE_BACKEND_AUTOLOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES']}}
dest=root/'checks'/f'train_cpu_v2_launch_{family}_s{seed}.json'
with dest.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
CHECK
  date -u '+START %Y-%m-%dT%H:%M:%SZ'
  "$SEER_PYTHON" -B tools/train_source_selector.py --cache "$SOURCE_ROOT/cache_sdsd_n_s0" --out "$SOURCE_ROOT/runs_none_cpu_v2/${SEER_SOURCE_FAMILY}_s${SEER_SELECTOR_SEED}" --candidate-mode "$SEER_SOURCE_FAMILY" --seed "$SEER_SELECTOR_SEED" --device cpu > "$TASK_ROOT/logs/train_cpu_v2_${SEER_SOURCE_FAMILY}_s${SEER_SELECTOR_SEED}.log" 2>&1
  date -u '+END %Y-%m-%dT%H:%M:%SZ'
