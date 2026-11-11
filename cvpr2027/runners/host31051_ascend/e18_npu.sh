#!/bin/bash
# e18_npu.sh NPU JOB [JOB...]   JOB = "name:flags(+ separated)"
# Ascend half of the converged architecture wave. Waits for free HBM so queued
# jobs pick up the lanes E17 releases as it finishes.
set -u
NPU=$1; shift
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e18
PROG=/temp/zihaohe/seer_cvpr/results/e18_progress.log
NEED=${NEED:-20000}
mkdir -p "$RUNS" "$(dirname "$PROG")"; cd $CODE
for JOB in "$@"; do
  TAG=${JOB%%:*}; FLAGS=$(echo "${JOB#*:}" | tr '+' ' ')
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  while :; do
    USED=$(python3 /cache/zihaohe/seer_cvpr/nf.py $NPU 2>/dev/null)
    [ -z "$USED" ] && USED=65536
    [ $((65536 - USED)) -ge "$NEED" ] && break
    echo "[$(date '+%F %T')] NPU$NPU wait $TAG (free $((65536-USED)))" >> "$PROG"; sleep 300
  done
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] NPU$NPU start $TAG ($FLAGS)" >> "$PROG"
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/seer_train_cvpr.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device npu --arch v2 \
      --branch small --vjepa pretrained $FLAGS \
      --iters "${ITERS:-20000}" --val_every 1000 --save_every 5000 --num_workers 4 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] NPU$NPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] NPU$NPU FAILED $TAG" >> "$PROG"
  fi
done
