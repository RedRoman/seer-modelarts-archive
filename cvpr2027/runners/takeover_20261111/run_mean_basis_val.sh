#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/cann-8.5.2/set_env.sh
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/temp/zihaohe/seer_cvpr/code/baselines:/temp/zihaohe/seer_cvpr/code/baselines/common:${PYTHONPATH:-}
TASK_ROOT=/cache/zihaohe/seer_takeover_20261111
BASIS=/cache/zihaohe/seer_cvpr/results/token_basis_train.json
cd "$TASK_ROOT/code_full"
test ! -e "$TASK_ROOT/results/mean_basis_val_s0.json"
test ! -e "$TASK_ROOT/results/mean_basis_val_s1.json"
date -u '+START %Y-%m-%dT%H:%M:%SZ'
sha256sum tools/mean_basis_control.py tools/eval_e4.py tools/checkpoint_config.py "$BASIS"
for SEER_SEED in 0 1; do
  date -u "+SEED $SEER_SEED START %Y-%m-%dT%H:%M:%SZ"
  /cache/zihaohe/seer_cvpr/venv_npu/bin/python -B tools/mean_basis_control.py \
    --ckpt "/cache/zihaohe/seer_cvpr/runs_e17/e17_small_pretrained_s$SEER_SEED/final_cpu.pt" \
    --basis "$BASIS" --split val --device npu \
    --out "$TASK_ROOT/results/mean_basis_val_s${SEER_SEED}.json"
done
date -u '+END %Y-%m-%dT%H:%M:%SZ'
