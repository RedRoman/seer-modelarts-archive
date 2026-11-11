#!/bin/bash
# H17's verdict, assembled the moment E22 and E24 finish.
#
# Amendment 13's rule needs two things, and neither is the `cos` the probe
# reports: a temporal metric must improve significantly against the converged
# baseline, and the model must become sensitive to the order of its input
# frames, which currently reads 0.054 dB. Both arms are scored because E22 is
# the only direct measurement of what the misnormalised term does at
# convergence, and reporting the fix without it would hide the comparison that
# motivated the fix.
set -u
export SEER_ROOT=/cache/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/cache/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e19
OUT=/cache/zihaohe/seer_cvpr/results/h17_eval_test
mkdir -p "$OUT"; cd $CODE

for TAG in e22_tdelta e24_tdcos; do
  for S in 0 1; do
    while [ ! -s "$RUNS/${TAG}_s$S/final.pt" ]; do sleep 180; done
  done
done
sleep 120

for TAG in e24_tdcos e22_tdelta; do
  for S in 0 1; do
    CK=$RUNS/${TAG}_s$S/final.pt
    echo "[$(date '+%F %T')] eval ${TAG}_s$S"
    CUDA_VISIBLE_DEVICES=0 $PY tools/eval_e4.py --ckpt "$CK" --heldout sdsd \
      --split test --device cuda --out "$OUT/${TAG}_s$S.json"
    CUDA_VISIBLE_DEVICES=0 $PY tools/td_probe.py --ckpt "$CK" --clips 12 \
      --device cuda:0 --out "$OUT/${TAG}_s${S}_tdprobe.json" 2>&1 | tail -5
  done
  # criterion (ii): order sensitivity, against the no-prior arm of the same
  # protocol so the intervention is prior-specific rather than global
  echo "[$(date '+%F %T')] temporal_ablate $TAG"
  CUDA_VISIBLE_DEVICES=1 $PY tools/temporal_ablate.py --domain sdsd --split test \
    --device cuda --out "$OUT/${TAG}_order.json" \
    --ckpt_pretrained "$RUNS/${TAG}_s0/final.pt" \
    --ckpt_none /cache/zihaohe/seer_cvpr/runs_e19/e19_seed2_none/final.pt
done
echo "[$(date '+%F %T')] === H17 inputs complete"
