#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/cann-8.5.2/set_env.sh
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/temp/zihaohe/seer_cvpr/code/baselines:/temp/zihaohe/seer_cvpr/code/baselines/common:${PYTHONPATH:-}
TASK_ROOT=/cache/zihaohe/seer_takeover_20261111
OUT="$TASK_ROOT/results/e17_small_random_s0_saved.json"
test ! -e "$OUT"
cd "$TASK_ROOT/code_full"
date -u '+START %Y-%m-%dT%H:%M:%SZ'
sha256sum tools/eval_e4.py tools/checkpoint_config.py models/pixel_branch_v2.py models/token_projection.py
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -B tools/eval_e4.py \
  --ckpt /cache/zihaohe/seer_cvpr/runs_e17/e17_small_random_s0/final_cpu.pt \
  --heldout sdsd --split test --crop 256 --device npu \
  --reconstruction saved --out "$OUT"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
