#!/bin/bash
# E2 at convergence (taps 23,23,23,23) and H16-prime (corruption-trained gate).
# The gate also gets the random-token pass, which is criterion (i) of H16-prime
# and is inference-only.
set -u
export SEER_ROOT=/cache/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/cache/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
cd /cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
R=/cache/zihaohe/seer_cvpr/runs_e19
O=/cache/zihaohe/seer_cvpr/results/wave_eval_test
mkdir -p "$O"
for S in 0 1; do
  CUDA_VISIBLE_DEVICES=2 $PY tools/eval_e4.py --ckpt $R/e2c_deep_s$S/final.pt \
    --heldout sdsd --split test --device cuda --out "$O/e2c_deep_s$S.json"
  CUDA_VISIBLE_DEVICES=2 $PY tools/eval_e4.py --ckpt $R/e25_cgate_s$S/final.pt \
    --heldout sdsd --split test --device cuda --out "$O/e25_cgate_s$S.json"
  CUDA_VISIBLE_DEVICES=2 $PY tools/eval_e4.py --ckpt $R/e25_cgate_s$S/final.pt \
    --heldout sdsd --split test --device cuda --force_arm random \
    --out "$O/e25_cgate_randtok_s$S.json"
  CUDA_VISIBLE_DEVICES=2 $PY tools/gate_audit.py --ckpt $R/e25_cgate_s$S/final.pt \
    --clips 8 --out "$O/e25_gate_audit_s$S.json"
done
echo "=== wave done"
