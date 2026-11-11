#!/bin/bash
set -eo pipefail
export SEER_CANN_ROOT=/cache/hanbin/ascend_private/cann-8.5.2
source /temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/pilot_v2
export PYTHONPATH="$TASK_ROOT/code_full:$TASK_ROOT/code_full/tools:${PYTHONPATH:-}"
cd "$TASK_ROOT/code_full"
export ASCEND_RT_VISIBLE_DEVICES=4
test ! -e "$TASK_ROOT/checks/smoke_fusion.json"
"$SEER_PYTHON" -B - "$ASCEND_RT_VISIBLE_DEVICES" <<'CHECK'
import hashlib,pathlib,subprocess,sys
assert ('No running processes found in NPU '+sys.argv[1]) in subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
p=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/pilot_v2/incoming/source_pilot_v2.tgz')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='25948bb0ef40c46199cdbb1957826516c992494fbc155637d289c262fffc5351'
p=pathlib.Path('/temp/zihaohe/seer_cvpr/method_dev_20261111/inputs/e17_small_pretrained_s0_final_cpu.pt')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740'
CHECK
date -u '+START %Y-%m-%dT%H:%M:%SZ'
"$SEER_PYTHON" -B tools/smoke_method_pilot.py --init "$SEER_BASELINE_CKPT" --device npu --trainable fusion --out "$TASK_ROOT/checks/smoke_fusion.json"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
