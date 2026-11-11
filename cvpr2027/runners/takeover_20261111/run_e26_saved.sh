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
BASIS=/cache/zihaohe/seer_cvpr/results/token_basis_train.json
cd "$TASK_ROOT/code_full"
test ! -e "$TASK_ROOT/results/e26_grank64_s0_saved.json"
test ! -e "$TASK_ROOT/results/e26_grank64_s1_saved.json"
date -u '+START %Y-%m-%dT%H:%M:%SZ'
sha256sum tools/eval_e4.py tools/checkpoint_config.py models/pixel_branch_v2.py models/token_projection.py "$BASIS"
for SEER_SEED in 0 1; do
  date -u "+SEED $SEER_SEED START %Y-%m-%dT%H:%M:%SZ"
  /cache/zihaohe/seer_cvpr/venv_npu/bin/python -B tools/eval_e4.py \
    --ckpt "/cache/zihaohe/seer_cvpr/runs_e26/e26_grank64_s$SEER_SEED/final.pt" \
    --heldout sdsd --split test --crop 256 --device npu \
    --reconstruction saved --token_basis "$BASIS" \
    --out "$TASK_ROOT/results/e26_grank64_s${SEER_SEED}_saved.json"
done
date -u '+END %Y-%m-%dT%H:%M:%SZ'
