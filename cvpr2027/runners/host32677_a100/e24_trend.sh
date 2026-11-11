#!/bin/bash
# Probe E24 at every checkpoint it writes, so the question that matters --
# whether the alignment keeps climbing or asymptotes -- is answered by a trend
# rather than by one point. E22's collapse was visible at 5000 and unchanged at
# 10000; if E24's cos is real it should keep moving.
set -u
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
R=/cache/zihaohe/seer_cvpr/runs_e19/e24_tdcos_s0
O=/cache/zihaohe/seer_cvpr/results/td_probe
cd $CODE
for S in 10000 15000 20000; do
  CK=$R/step_$S.pt
  [ $S = 20000 ] && CK=$R/final.pt
  while [ ! -s "$CK" ]; do sleep 300; done
  sleep 90
  CUDA_VISIBLE_DEVICES=2 $PY tools/td_probe.py --ckpt "$CK" --clips 12 \
      --device cuda:0 --out "$O/e24_tdcos_s0_step$S.json" 2>&1 | tail -6
done
