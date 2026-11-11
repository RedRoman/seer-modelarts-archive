#!/bin/bash
# Third seed for the in-domain prior-vs-none comparison. At two seeds the PSNR
# effect was +0.601 dB with a CI spanning zero and LPIPS -0.0155 significant;
# ground rule 4 also wants every seed to agree in sign, which two seeds can
# state but three test properly.
set -u
export SEER_ROOT=/cache/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/cache/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
cd /cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
for ARM in pretrained none; do
  CUDA_VISIBLE_DEVICES=3 $PY tools/eval_e4.py \
    --ckpt /cache/zihaohe/seer_cvpr/runs_e19/e19_seed2_$ARM/final.pt \
    --heldout sdsd --split test --device cuda \
    --out /cache/zihaohe/seer_cvpr/results/e17_eval_test/e19_small_${ARM}_s2.json
done
echo "=== done"
