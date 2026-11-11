#!/bin/bash
set -eo pipefail
TASK_ROOT=/cache/zihaohe/seer_method_dev_20261111/selector_val_v1
for seed in 1 2; do
 bash "$TASK_ROOT/tools/run_val_replicate_f2cf65.sh" "$seed" > "$TASK_ROOT/logs/eval_bvi_val16_s${seed}.log" 2>&1
done
date -u '+QUEUE_COMPLETE %Y-%m-%dT%H:%M:%SZ'
