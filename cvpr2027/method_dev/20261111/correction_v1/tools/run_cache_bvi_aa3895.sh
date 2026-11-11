#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/correction_v1
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=7
cd "$TASK_ROOT/code_full"
test ! -e "$TASK_ROOT/cache_bvi_train48"
"$SEER_PYTHON" -B - <<'CHECK'
import hashlib,pathlib,subprocess,json,tarfile
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/correction_v1')
assert 'No running processes found in NPU 7' in subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
p=root/'incoming/source_correction_v1.tgz'; assert hashlib.sha256(p.read_bytes()).hexdigest()=='aa3895124732263c82d33b2067f5d05f78dcb55240b329b57e8c1e4ccdcaf947'
files={}
with tarfile.open(p) as arc:
 for m in arc.getmembers():
  if not m.isfile() or any(part.startswith('.') for part in pathlib.PurePosixPath(m.name).parts): continue
  h=hashlib.sha256(arc.extractfile(m).read()).hexdigest(); assert hashlib.sha256((root/'code_full'/m.name).read_bytes()).hexdigest()==h,m.name; files[m.name]=h
(root/'checks/source_manifest.json').write_text(json.dumps({'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':files},indent=2)+'\n')
print('Verified source files',len(files),flush=True)
CHECK
date -u '+START %Y-%m-%dT%H:%M:%SZ'
"$SEER_PYTHON" -B tools/cache_correction_oracle.py --init /cache/zihaohe/seer_cvpr/runs_e12/e12_bvi_pretrained_s0/final_cpu.pt --raft-weights /temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth --out "$TASK_ROOT/cache_bvi_train48" --data-seed 20261113 --patch-size 8 --offset 2 --flow-steps 12 --flow-batch 2 --max-sequences 48 --device npu
date -u '+END %Y-%m-%dT%H:%M:%SZ'
