#!/bin/bash
# H18: does low-rank sufficiency survive training? Scored on the same test
# split and evaluator as the three-seed full-token baseline it is compared with.
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
cd /temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
O=/cache/zihaohe/seer_cvpr/results/e26_eval_test
mkdir -p "$O"
for S in 0 1; do
  ASCEND_RT_VISIBLE_DEVICES=$((S+4)) $PY tools/eval_e4.py \
    --ckpt /cache/zihaohe/seer_cvpr/runs_e26/e26_grank64_s$S/final.pt \
    --heldout sdsd --split test --device npu --out "$O/e26_grank64_s$S.json" &
done
wait
echo "=== e26 eval done"
