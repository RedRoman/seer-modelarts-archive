#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd
SOURCE_ROOT=/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$TASK_ROOT/code_full"
"$SEER_PYTHON" -B - <<'CHECK'
import pathlib,hashlib,json,subprocess,os,datetime,platform
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd')
source=root.parents[1]
out=source/'results/bvi_val16_none_s0s1s2.json';assert not out.exists(),out
m=json.loads((root/'source_manifest.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((root/'code_full'/n).read_bytes()).hexdigest()==h,n
records=[]
for family in ('restored','restored_detail','raw_detail'):
 for seed in range(3):
  run=source/'runs_none_cpu_v2'/f'{family}_s{seed}'
  hold=json.loads((run/'internal_holdout.json').read_text());assert hold['oracle_computed'] is False
  for mode in ('direct_mse','direct_mse_normalized'):
   p=run/f'selector_{mode}.pt';h=hashlib.sha256(p.read_bytes()).hexdigest()
   side=json.loads(p.with_suffix('.sha256.json').read_text());assert side['checkpoint_sha256']==h
   rows=[json.loads(s) for s in (run/f'curve_{mode}.jsonl').read_text().splitlines()]
   assert len(rows)==1000 and [x['step'] for x in rows]==list(range(1,1001))
   records.append({'family':family,'seed':seed,'mode':mode,'path':str(p),'sha256':h,'steps':1000})
stcd=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/stcd_transfer_v1/results/stcd_indoor_release_bvi_val16.json')
assert stcd.is_file(),'Independent STCD reference must finish first'
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15);(root/'checks/npu_before_none_eval.txt').write_text(smi)
assert 'No running processes found in NPU 7' in smi
import importlib.metadata
import eval_e4
import common.eval_lib as metric_library
metric_path=pathlib.Path(metric_library.__file__).resolve()
metric_runtime={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metric_file':str(metric_path),'metric_sha256':hashlib.sha256(metric_path.read_bytes()).hexdigest(),'versions':{k:importlib.metadata.version(k) for k in ['torch','torch-npu','pyiqa']},'python':os.environ['SEER_PYTHON'],'collection':'Immediately before this target evaluation; STCD earlier result records its own actual metric source hash'}
with (root/'checks/metric_runtime_none.json').open('x') as f:json.dump(metric_runtime,f,indent=2)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':platform.node(),'source_archive_sha256':'c65c0b4919cd8befdd850f9f96e5ce35565a45c3b3f5335ae2c50d00bebac14c','wrapper_sha256':hashlib.sha256((root/'tools/run_none_eval_cpu_v2_c65c0b.sh').read_bytes()).hexdigest(),'checkpoints':records,'out':str(out),'physical_npu':7,'device':'npu:0','scope':'All predeclared 18 final heads, one shared base/RAFT computation, target BVI VAL16 metrics only; no target tuning/selection/oracle','environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES']}}
with (root/'checks/none_eval_launch_manifest.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
CHECK
date -u '+START %Y-%m-%dT%H:%M:%SZ'
"$SEER_PYTHON" -B tools/eval_source_selector.py --selectors "$SOURCE_ROOT/runs_none_cpu_v2/restored_s0" "$SOURCE_ROOT/runs_none_cpu_v2/restored_s1" "$SOURCE_ROOT/runs_none_cpu_v2/restored_s2" "$SOURCE_ROOT/runs_none_cpu_v2/restored_detail_s0" "$SOURCE_ROOT/runs_none_cpu_v2/restored_detail_s1" "$SOURCE_ROOT/runs_none_cpu_v2/restored_detail_s2" "$SOURCE_ROOT/runs_none_cpu_v2/raw_detail_s0" "$SOURCE_ROOT/runs_none_cpu_v2/raw_detail_s1" "$SOURCE_ROOT/runs_none_cpu_v2/raw_detail_s2" --init /temp/zihaohe/seer_cvpr/method_dev_20261111/inputs/e17_small_none_s0_final_cpu.pt --raft-weights /temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth --out "$SOURCE_ROOT/results/bvi_val16_none_s0s1s2.json" --device npu
npu-smi info > "$TASK_ROOT/checks/npu_after_none_eval.txt"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
