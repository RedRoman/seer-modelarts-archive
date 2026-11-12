#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
ANCHOR_ROOT=/cache/zihaohe/seer_hvi_source_anchor_20261112
ANCHOR_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
"$ANCHOR_PYTHON" -B - <<'PY'
from pathlib import Path
import sys,subprocess,json,datetime,hashlib
root=Path('/cache/zihaohe/seer_hvi_source_anchor_20261112')
release=root/'release_v1'
sys.path.insert(0,str(release))
from cache_hvi_source import read_lq_only,digest
# Production storage-reader compatibility: LQ only; no model output or score.
name,scene,low,lineage=read_lq_only('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0/clip_000.pt')
assert low.shape==(1,3,16,256,256)
with (root/'checks_v1/real_source_lq_reader.json').open('x') as f:
    json.dump({'purpose':'LQ storage compatibility only; no real prediction or score','name':name,'scene':scene,'lineage':lineage},f,indent=2)
def free_card(label):
    smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
    with (root/'checks_v1'/('npu_before_'+label+'.txt')).open('x') as f:f.write(smi)
    assert 'No running processes found in NPU 7' in smi
python='/cache/zihaohe/seer_cvpr/venv_npu/bin/python'
free_card('synthetic_parity')
with (root/'logs_v1/synthetic_parity.log').open('x') as log:
    subprocess.run([python,'-B',str(release/'synthetic_preflight.py'),'--release',str(release),'--out',str(root/'checks_v1/synthetic_cpu_npu_parity.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
assert json.loads((root/'checks_v1/synthetic_cpu_npu_parity.json').read_text())['passed'] is True
free_card('source_export')
with (root/'checks_v1/source_launch.json').open('x') as f:
    json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'source TRAIN136 float HVI only; no GT pixels or scores','physical_npu':7,'release_manifest_sha256':digest(release/'release_manifest.json'),'wrapper_sha256':digest(root/'run_source_v1.sh'),'scores_computed':False},f,indent=2)
with (root/'logs_v1/source_export.log').open('x') as log:
    subprocess.run([python,'-B',str(release/'cache_hvi_source.py'),'--source-cache','/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0','--release',str(release),'--out',str(root/'cache_v1'),'--device','npu'],stdout=log,stderr=subprocess.STDOUT,check=True)
with (root/'checks_v1/npu_after_source.txt').open('x') as f:f.write(subprocess.check_output(['npu-smi','info'],text=True,timeout=15))
with (root/'checks_v1/all_done.json').open('x') as f:json.dump({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_outputs':136,'training':False,'scores_computed':False},f,indent=2)
print('SOURCE CACHE COMPLETE',flush=True)
PY
