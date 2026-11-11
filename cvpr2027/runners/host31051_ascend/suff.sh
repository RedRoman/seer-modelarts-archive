#!/bin/bash
# Token sufficiency on the other two converged seeds. The screen on seed 2 says
# rank 4 of 1024, and a single spatial mean per tubelet, both lose nothing --
# which would contradict this paper's own title. Before that is written down it
# needs more than one checkpoint and a paired confidence interval.
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
cd /temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
for S in 0 1; do
  ASCEND_RT_VISIBLE_DEVICES=$((S+4)) $PY tools/token_sufficiency.py \
    --ckpt /cache/zihaohe/seer_cvpr/runs_e17/e17_small_pretrained_s$S/final_cpu.pt \
    --clips 39 --device npu:0 \
    --out /cache/zihaohe/seer_cvpr/results/token_sufficiency_s$S.json &
done
wait
echo "=== both done"
