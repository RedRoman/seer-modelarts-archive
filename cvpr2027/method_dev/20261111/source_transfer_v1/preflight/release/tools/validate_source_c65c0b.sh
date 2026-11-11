#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
cd "$TASK_ROOT/code_full"
"$SEER_PYTHON" -B - <<'CHECK'
import pathlib,json,hashlib,ast
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd')
assert hashlib.sha256((root/'incoming/source_transfer_v1.tgz').read_bytes()).hexdigest()=='c65c0b4919cd8befdd850f9f96e5ce35565a45c3b3f5335ae2c50d00bebac14c'
m=json.loads((root/'source_manifest.json').read_text())
for n,h in m['files'].items():
 data=(root/'code_full'/n).read_bytes();assert hashlib.sha256(data).hexdigest()==h,n;ast.parse(data,filename=n)
assert len(m['files'])==230
for n in m['excluded_paths']:assert not (root/'code_full'/n).exists(),n
(root/'checks/ast_source_check.json').write_text(json.dumps({'source_files_verified':230,'ast_parsed':230,'excluded':m['excluded_paths']},indent=2)+'\n')
print('230 source files SHA and AST verified; STCD excluded',flush=True)
CHECK
"$SEER_PYTHON" -B -m unittest discover -s tests -p 'test_*.py' -v > "$TASK_ROOT/logs/cpu_tests.log" 2>&1
"$SEER_PYTHON" -B tools/cache_source_corrections.py --help > "$TASK_ROOT/logs/help_cache.txt" 2>&1
"$SEER_PYTHON" -B tools/train_source_selector.py --help > "$TASK_ROOT/logs/help_train.txt" 2>&1
"$SEER_PYTHON" -B tools/eval_source_selector.py --help > "$TASK_ROOT/logs/help_eval.txt" 2>&1
"$SEER_PYTHON" -B - <<'CHECK'
import pathlib,json,re,hashlib,datetime
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/releases/v1_c65c0b4919cd')
log=(root/'logs/cpu_tests.log').read_text();match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',log)
assert match and int(match[1])==74 and '\nOK\n' in log and 'skipped=' not in log
result={'status':'PASS','tests':int(match[1]),'skips':0,'seconds':float(match[2]),'cpu_threads':4,'device':'cpu','ast_files':230,'help_entrypoints':3,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_log_sha256':hashlib.sha256((root/'logs/cpu_tests.log').read_bytes()).hexdigest()}
(root/'checks/preflight_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
CHECK
