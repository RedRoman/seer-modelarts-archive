#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/selector_val_v1
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=7
cd "$TASK_ROOT/code_full"
test ! -e "$TASK_ROOT/results/bvi_val16_s0.json"
"$SEER_PYTHON" -B - <<'CHECK'
import hashlib,pathlib,json,tarfile,subprocess,os,platform,datetime
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/selector_val_v1')
p=root/'incoming/source_selector_val_v1.tgz'
h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h=='f2cf65cfdb0a79e9cfc071127d2275b003f8d3a0e369800987625f0a5aa4ec1e'
files={}
with tarfile.open(p) as arc:
 for m in arc.getmembers():
  if not m.isfile() or any(part.startswith('.') for part in pathlib.PurePosixPath(m.name).parts):continue
  s=hashlib.sha256(arc.extractfile(m).read()).hexdigest()
  assert hashlib.sha256((root/'code_full'/m.name).read_bytes()).hexdigest()==s,m.name
  files[m.name]=s
assert len(files)==205
assert files['tools/eval_correction_selector.py']=='5d99715af4ba3084c342a0b6086e278da969fc2669515eef6b8046b74d204853'
train=json.loads((root.parent/'selector_v1/checks/source_manifest.json').read_text())
assert files.keys()==train['files'].keys()
changed=[n for n in files if files[n]!=train['files'][n]]
assert changed==['tools/eval_correction_selector.py'],changed
(root/'checks/source_manifest.json').write_text(json.dumps({'archive_sha256':h,'files':files,'changed_from_selector_training':changed},indent=2)+'\n')
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
(root/'checks/npu_before.txt').write_text(smi)
assert 'No running processes found in NPU 7' in smi
record={'host':'huawei-modelarts-31051','hostname':platform.node(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_archive_sha256':h,'source_manifest_sha256':hashlib.sha256((root/'checks/source_manifest.json').read_bytes()).hexdigest(),
 'wrapper_sha256':hashlib.sha256((root/'tools/run_val_s0_f2cf65.sh').read_bytes()).hexdigest(),
 'python':os.environ['SEER_PYTHON'],'device':'npu:0','physical_device':7,
 'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES']},
 'selectors':'/cache/zihaohe/seer_method_dev_20261111/selector_v1/runs/s0',
 'base':'/cache/zihaohe/seer_cvpr/runs_e12/e12_bvi_pretrained_s0/final_cpu.pt',
 'raft':'/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth',
 'out':str(root/'results/bvi_val16_s0.json'),'policy':'All three predeclared final1000 arms; frozen BVI VAL16; no test; guards fail closed.'}
(root/'launch_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print('205 source hashes verified; only evaluator differs from training; NPU7 process-free',flush=True)
CHECK
date -u '+START %Y-%m-%dT%H:%M:%SZ'
"$SEER_PYTHON" -B tools/eval_correction_selector.py --selectors /cache/zihaohe/seer_method_dev_20261111/selector_v1/runs/s0 --init /cache/zihaohe/seer_cvpr/runs_e12/e12_bvi_pretrained_s0/final_cpu.pt --raft-weights /temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth --out "$TASK_ROOT/results/bvi_val16_s0.json" --device npu
date -u '+END %Y-%m-%dT%H:%M:%SZ'
