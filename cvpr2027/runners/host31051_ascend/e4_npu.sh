#!/bin/bash
# e4_npu.sh NPU_ID JOB [JOB...]    JOB = "heldout:arm:seed"
# Leave-one-domain-out runs for H5 on one Ascend 910B. ITERS from env.
set -u
NPU=$1; shift
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e4
PROG=/temp/zihaohe/seer_cvpr/results/e4_progress.log
mkdir -p "$RUNS" "$(dirname "$PROG")"; cd $CODE
for JOB in "$@"; do
  HO=${JOB%%:*}; R=${JOB#*:}; ARM=${R%%:*}; SEED=${R##*:}
  TAG="e4_${HO}_${ARM}_s${SEED}"
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] skip $TAG (done)" >> "$PROG"; continue
  fi
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] NPU$NPU start $TAG heldout=$HO arm=$ARM seed=$SEED" >> "$PROG"
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/seer_train_e4.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device npu \
      --heldout "$HO" --vjepa "$ARM" --seed "$SEED" \
      --iters "${ITERS:-2500}" --val_every 500 --save_every 2500 --num_workers 4 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] NPU$NPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] NPU$NPU FAILED $TAG" >> "$PROG"
  fi
done
echo "[$(date '+%F %T')] NPU$NPU E4 queue finished" >> "$PROG"
