#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
export SEER_SELECTOR_SEED="${1:?seed required}"
[[ "$SEER_SELECTOR_SEED" == 1 || "$SEER_SELECTOR_SEED" == 2 ]]
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/selector_v1
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
cd "$TASK_ROOT/code_full"
test ! -e "$TASK_ROOT/runs/s${SEER_SELECTOR_SEED}"
"$SEER_PYTHON" -B - <<'CHECK'
import hashlib,pathlib,json,platform,os,datetime
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/selector_v1')
seed=int(os.environ['SEER_SELECTOR_SEED'])
assert seed in (1,2)
log=(root/'logs/cpu_tests.log').read_text()
assert 'Ran 59 tests' in log and '\nOK\n' in log and 'CPU_TESTS_COMPLETE' in log
assert 'skipped=' not in log
manifest=json.loads((root/'checks/source_manifest.json').read_text())
for name,h in manifest['files'].items():
 assert hashlib.sha256((root/'code_full'/name).read_bytes()).hexdigest()==h,name
record={'host':'huawei-modelarts-31051','hostname':platform.node(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_archive_sha256':manifest['archive_sha256'],'source_manifest_sha256':hashlib.sha256((root/'checks/source_manifest.json').read_bytes()).hexdigest(),
 'cpu_tests_sha256':hashlib.sha256((root/'logs/cpu_tests.log').read_bytes()).hexdigest(),
 'wrapper_sha256':hashlib.sha256((root/'tools/run_selector_replicate_35e7a7.sh').read_bytes()).hexdigest(),
 'python':os.environ['SEER_PYTHON'],'device':'cpu','cpu_threads':4,
 'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH']},
 'cache':'/cache/zihaohe/seer_method_dev_20261111/correction_v1/cache_bvi_train48',
 'out':str(root/f'runs/s{seed}'),'args':{'steps':1000,'lr':.001,'seed':seed,'batch_frames':8,'device':'cpu'},
 'policy':'Fixed final step; three matched arms; only one internal holdout evaluation; no formal VAL/test evaluation.'}
(root/f'launch_manifest_s{seed}.json').write_text(json.dumps(record,indent=2)+'\n')
print('Launch provenance recorded',flush=True)
CHECK
date -u '+START %Y-%m-%dT%H:%M:%SZ'
"$SEER_PYTHON" -B tools/train_correction_selector.py --cache /cache/zihaohe/seer_method_dev_20261111/correction_v1/cache_bvi_train48 --out "$TASK_ROOT/runs/s${SEER_SELECTOR_SEED}" --steps 1000 --lr 1e-3 --seed "$SEER_SELECTOR_SEED" --batch-frames 8 --device cpu
date -u '+END %Y-%m-%dT%H:%M:%SZ'
