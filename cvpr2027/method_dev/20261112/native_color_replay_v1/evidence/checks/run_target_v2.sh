#!/bin/bash
set -eo pipefail
ARM="$1"
PHYSICAL="$2"
if [[ "$ARM" != "pretrained" && "$ARM" != "none" ]]; then exit 2; fi
if [[ "$PHYSICAL" != "6" && "$PHYSICAL" != "7" ]]; then exit 2; fi
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
ROOT=/cache/zihaohe/seer_native_color_replay_20261112
RELEASE="$ROOT/release_v2"
export PYTHONPATH="$RELEASE/code_full:$RELEASE/code_full/tools:${PYTHONPATH:-}"
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -B - "$ROOT" "$ARM" "$PHYSICAL" "$0" <<'PY'
import datetime,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);arm=sys.argv[2]
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
p=root/'metric_dependencies_v1.json'
assert sha(p)=='b31ba19caf18f7dd8433b434762e729f770eba7a3707d1c63800708d9a6585eb'
for path,digest in json.loads(p.read_text())['files'].items():assert sha(path)==digest,path
auth=root/'target_authorization_v2.json';a=json.loads(auth.read_text())
assert a['source_compatibility_checks_passed']==4 and a['authorized_stage']=='target_bvi_val16'
assert not (root/f'target_{arm}_v2.json').exists()
row={'arm':arm,'physical_npu':int(sys.argv[3]),'authorization_sha256':sha(auth),
 'metric_dependency_manifest_sha256':sha(p),'controller_sha256':sha(sys.argv[4]),
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_pixel_reads_before_launch':0,
 'scope':'Exact frozen two-arm native-color compatibility replay; no fitting or selection'}
with (root/f'target_launch_{arm}_v2.json').open('x') as f:json.dump(row,f,indent=2);f.write('\n')
print(json.dumps(row),flush=True)
PY
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -B "$RELEASE/phase/replay_native.py" \
  --release "$RELEASE" --stage target --arm "$ARM" --out "$ROOT/target_${ARM}_v2.json" \
  --authorization "$ROOT/target_authorization_v2.json" \
  --source-pretrained "$ROOT/source_pretrained_v2.json" --source-none "$ROOT/source_none_v2.json"
