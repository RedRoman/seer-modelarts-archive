#!/bin/bash
# Probe the E22 checkpoint the moment it is written. Amendment 14 registers the
# prediction (mag < 0.5, cos < +0.10 = trivial collapse) before this runs.
CK=/cache/zihaohe/seer_cvpr/runs_e19/e22_tdelta_s0/step_5000.pt
while [ ! -s "$CK" ]; do sleep 120; done
sleep 60   # let the write finish
cd /cache/zihaohe/seer_cvpr/code/vjepa_llve
CUDA_VISIBLE_DEVICES=3 /cache/zihaohe/seer_cvpr/venv_cu/bin/python tools/td_probe.py \
  --ckpt "$CK" --clips 12 --device cuda:0 \
  --out /cache/zihaohe/seer_cvpr/results/td_probe/e22_tdelta_s0_step5000.json
