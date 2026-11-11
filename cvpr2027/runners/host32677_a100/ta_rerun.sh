#!/bin/bash
# H17 criterion (ii): does the trained model become sensitive to input frame
# order. The converged baseline reads 0.054 dB; E24 is the arm that is supposed
# to move it, and E22 is scored too because it is the control the fix came from.
set -u
export SEER_ROOT=/cache/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/cache/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
cd /cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
OUT=/cache/zihaohe/seer_cvpr/results/h17_eval_test
for TAG in e24_tdcos e22_tdelta; do
  CUDA_VISIBLE_DEVICES=1 $PY tools/temporal_ablate.py --domain sdsd --split test \
    --device cuda --out "$OUT/${TAG}_order.json" \
    --ckpt_pretrained /cache/zihaohe/seer_cvpr/runs_e19/${TAG}_s0/final.pt \
    --ckpt_none /cache/zihaohe/seer_cvpr/runs_e19/e19_seed2_none/final.pt
done
echo "=== order done"
