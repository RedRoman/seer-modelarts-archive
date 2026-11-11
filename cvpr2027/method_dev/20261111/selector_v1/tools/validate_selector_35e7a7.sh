#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
export SEER_PYTHON=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/selector_v1
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
cd "$TASK_ROOT/code_full"
"$SEER_PYTHON" -B - <<'CHECK'
import hashlib,pathlib,json,tarfile
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/selector_v1')
p=root/'incoming/source_selector_v1.tgz'
archive_hash=hashlib.sha256(p.read_bytes()).hexdigest()
assert archive_hash=='35e7a704e2deb5aa111fa7d88586965ff61a961b926566958858ef3a376cc7ec'
files={}
with tarfile.open(p) as arc:
 for m in arc.getmembers():
  if not m.isfile() or any(part.startswith('.') for part in pathlib.PurePosixPath(m.name).parts): continue
  h=hashlib.sha256(arc.extractfile(m).read()).hexdigest()
  assert hashlib.sha256((root/'code_full'/m.name).read_bytes()).hexdigest()==h,m.name
  files[m.name]=h
assert len(files)==205,len(files)
(root/'checks/source_manifest.json').write_text(json.dumps({'archive_sha256':archive_hash,'files':files},indent=2)+'\n')
print('Verified source files',len(files),flush=True)
CHECK
"$SEER_PYTHON" -B -m unittest discover -s tests -p 'test_*.py' -v
date -u '+CPU_TESTS_COMPLETE %Y-%m-%dT%H:%M:%SZ'
