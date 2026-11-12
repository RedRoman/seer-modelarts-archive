#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
REPLAY_ROOT=/cache/zihaohe/seer_baseline_replay_20261112
REPLAY_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
"$REPLAY_PYTHON" -B - <<'PY'
from pathlib import Path
import datetime,json,subprocess,os,hashlib
root=Path('/cache/zihaohe/seer_baseline_replay_20261112')
release=root/'release_v1'
python='/cache/zihaohe/seer_cvpr/venv_npu/bin/python'
def free_card(label):
    smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
    (root/'checks_v1'/('npu_before_'+label+'.txt')).write_text(smi)
    assert 'No running processes found in NPU 7' in smi
for method in ('hvi_cidnet','darkir','retinexformer'):
    assert json.loads((root/'smoke_v1'/({'hvi_cidnet':'hvi_cpu'}.get(method,method+'_cpu')+'.json')).read_text())['state_unchanged']
free_card('retinexformer_smoke')
smoke=root/'smoke_v1/retinexformer_npu.json'
assert not smoke.exists()
with (root/'logs_v1/retinexformer_npu_smoke.log').open('x') as log:
    subprocess.run([python,'-B',str(release/'run_replay.py'),'--release',str(release),'--method','retinexformer','--model-device','npu','--metric-device','npu','--smoke','--out',str(smoke)],stdout=log,stderr=subprocess.STDOUT,check=True)
for method in ('hvi_cidnet','retinexformer'):
    assert json.loads((root/'smoke_v1'/(method+'_npu.json')).read_text())['state_unchanged']
decision={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'choices':{'hvi_cidnet':'NPU inference','darkir':'CPU inference; original complex FFT absolute operation unsupported on this NPU stack; no network changes','retinexformer':'NPU inference'},
    'metric_device':'physical NPU7/logical npu:0, same existing v2 scorer','scores_seen':False,
    'adapter_sha256':hashlib.sha256((release/'run_replay.py').read_bytes()).hexdigest()}
with (root/'checks_v1/device_decision_before_scores.json').open('x') as f:json.dump(decision,f,indent=2)
for method,device in [('hvi_cidnet','npu'),('darkir','cpu'),('retinexformer','npu')]:
    free_card(method)
    result=root/'results_v1'/(method+'_bvi_val16.json')
    assert not result.exists()
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':method,'model_device':device,'physical_npu':7,'out':str(result),'gt_policy':'score only'}
    with (root/'checks_v1'/(method+'_launch.json')).open('x') as f:json.dump(record,f,indent=2)
    with (root/'logs_v1'/(method+'_eval.log')).open('x') as log:
        subprocess.run([python,'-B',str(release/'run_replay.py'),'--release',str(release),'--method',method,'--model-device',device,'--metric-device','npu','--out',str(result)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',method,flush=True)
(root/'checks_v1/npu_after_all.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True,timeout=15))
with (root/'checks_v1/all_done.json').open('x') as f:json.dump({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'methods':3},f,indent=2)
PY
