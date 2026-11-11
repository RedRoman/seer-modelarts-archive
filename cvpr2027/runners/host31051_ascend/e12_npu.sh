#!/bin/bash
# e12_npu.sh NPU JOB [JOB...]   JOB = "arm:seed"
# E12 / H11: in-domain on BVI, the only genuinely dynamic dataset here
# (flow 2.63 px against SDSD's 0.18). Trains and validates on BVI alone, so the
# question is whether the prior helps at all on real motion, separate from
# whether it transfers.
set -u
NPU=$1; shift
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e12
PROG=/temp/zihaohe/seer_cvpr/results/e12_progress.log
mkdir -p "$RUNS" "$(dirname "$PROG")"; cd $CODE
for JOB in "$@"; do
  ARM=${JOB%%:*}; SEED=${JOB##*:}
  TAG="e12_bvi_${ARM}_s${SEED}"
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] NPU$NPU start $TAG" >> "$PROG"
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/seer_train_e4.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device npu \
      --heldout sdsd --only bvi --vjepa "$ARM" --seed "$SEED" \
      --iters "${ITERS:-2500}" --val_every 500 --save_every 2500 --num_workers 4 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] NPU$NPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] NPU$NPU FAILED $TAG" >> "$PROG"
  fi
done
