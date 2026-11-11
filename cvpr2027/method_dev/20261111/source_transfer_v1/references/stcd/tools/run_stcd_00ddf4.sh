#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/stcd_transfer_v1/releases/v1_00ddf46c4536
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=7
cd "$TASK_ROOT/code_full"
"$SEER_PYTHON" -B - <<'CHECK'
import pathlib,hashlib,json,subprocess,os,datetime,platform
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/stcd_transfer_v1/releases/v1_00ddf46c4536')
out=root.parents[1]/'results/stcd_indoor_release_bvi_val16.json';assert not out.exists(),out
cache=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0/cache_summary.json')
s=json.loads(cache.read_text());assert s['n']==136 and s['base_state_unchanged'] is True and s['raft_state_unchanged'] is True
m=json.loads((root/'source_manifest.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((root/'code_full'/n).read_bytes()).hexdigest()==h,n
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15);(root/'checks/npu_before_stcd.txt').write_text(smi)
assert 'No running processes found in NPU 7' in smi
checkpoint='/temp/zihaohe/seer_cvpr/code/ext_methods/LLVE_STCD/pretrained_model/SDSD_indoor_model/indoor_G.pth'
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':platform.node(),'source_archive_sha256':'00ddf46c453685600f10d1bf8c4e480eb313e90cb0d5286b6f3d8abd9ef86031','source_manifest_sha256':hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256((root/'tools/run_stcd_00ddf4.sh').read_bytes()).hexdigest(),'checkpoint':checkpoint,'checkpoint_sha256':hashlib.sha256(pathlib.Path(checkpoint).read_bytes()).hexdigest(),'out':str(out),'physical_npu':7,'device':'npu:0','chunk':1,'precision':'fp32','scope':'Released SDSD indoor model, BVI VAL16 reference, center16/crop256/lastdup, GT metrics only','environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES']}}
with (root/'checks/stcd_launch_manifest.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
CHECK
date -u '+START %Y-%m-%dT%H:%M:%SZ'
"$SEER_PYTHON" -B tools/eval_stcd_transfer.py --stcd-root /temp/zihaohe/seer_cvpr/code/ext_methods/LLVE_STCD --checkpoint /temp/zihaohe/seer_cvpr/code/ext_methods/LLVE_STCD/pretrained_model/SDSD_indoor_model/indoor_G.pth --seer-root /temp/zihaohe/seer_cvpr --device npu --chunk 1 --out /cache/zihaohe/seer_method_dev_20261111/stcd_transfer_v1/results/stcd_indoor_release_bvi_val16.json
npu-smi info > "$TASK_ROOT/checks/npu_after_stcd.txt"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
