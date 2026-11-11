#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=7
cd "$TASK_ROOT/code_full"
"$SEER_PYTHON" -B - <<'CHECK'
import pathlib,hashlib,json,subprocess,os,datetime,platform
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd')
out=root.parents[1]/'cache_sdsd_p_s0';assert not out.exists(),out
pre=json.loads((root/'checks/preflight_result.json').read_text());assert pre['status']=='PASS' and pre['tests']==74 and pre['skips']==0
m=json.loads((root/'source_manifest.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((root/'code_full'/n).read_bytes()).hexdigest()==h,n
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
base='/temp/zihaohe/seer_cvpr/method_dev_20261111/inputs/e17_small_pretrained_s0_final_cpu.pt'
raft='/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth'
assert digest(base)=='5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740'
assert digest(raft)=='01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27'
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15);(root/'checks/npu_before_cache.txt').write_text(smi)
assert 'No running processes found in NPU 7' in smi
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'huawei-modelarts-31051','hostname':platform.node(),'source_archive_sha256':'c65c0b4919cd8befdd850f9f96e5ce35565a45c3b3f5335ae2c50d00bebac14c','source_manifest_sha256':digest(root/'source_manifest.json'),'wrapper_sha256':digest(root/'tools/run_source_cache_c65c0b.sh'),'preflight':pre,'base':base,'base_sha256':digest(base),'raft':raft,'raft_sha256':digest(raft),'out':str(out),'data_seed':20261113,'crop':256,'clip_len':16,'device':'npu:0','physical_npu':7,'python':os.environ['SEER_PYTHON'],'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES']},'scope':'SDSD TRAIN136 only; source labels only; no target/no oracle'}
(root/'checks/cache_launch_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print('Source/input hashes and NPU7 free verified',flush=True)
CHECK
date -u '+START %Y-%m-%dT%H:%M:%SZ'
"$SEER_PYTHON" -B tools/cache_source_corrections.py --init /temp/zihaohe/seer_cvpr/method_dev_20261111/inputs/e17_small_pretrained_s0_final_cpu.pt --raft-weights /temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth --out /cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0 --data-seed 20261113 --device npu
npu-smi info > "$TASK_ROOT/checks/npu_after_cache.txt"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
