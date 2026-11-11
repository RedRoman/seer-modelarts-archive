#!/bin/bash
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
cd /temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
for S in 0 1; do
  ASCEND_RT_VISIBLE_DEVICES=$((S+4)) $PY tools/global_sufficiency.py \
    --ckpt /cache/zihaohe/seer_cvpr/runs_e17/e17_small_pretrained_s$S/final_cpu.pt \
    --basis /cache/zihaohe/seer_cvpr/results/token_basis_train.json \
    --device npu:0 --out /cache/zihaohe/seer_cvpr/results/global_sufficiency_s$S.json &
done
wait
echo "=== gsuff done"
