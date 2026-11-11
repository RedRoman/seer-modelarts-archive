#!/bin/bash
# e23_runner.sh GPU JOB [JOB...]   JOB = "exp:arm:variant:seed"
#   exp     e2 | e3
#   arm     pretrained | none | random
#   variant for e2 the tap list (dots for commas, e.g. 23.23.23.23)
#           for e3 full | small
# Waits for free HBM first: these cards are shared and a neighbour taking a
# whole 80 GB has killed a lane outright before.
set -u
GPU=$1; shift
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e23
PROG=$RUNS/e23_progress.log
NEED=${NEED:-34000}
mkdir -p "$RUNS"; cd $CODE
for JOB in "$@"; do
  EXP=${JOB%%:*}; R1=${JOB#*:}; ARM=${R1%%:*}; R2=${R1#*:}; VAR=${R2%%:*}; SEED=${R2##*:}
  TAG="${EXP}_${ARM}_${VAR}_s${SEED}"
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  while :; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    [ -z "$USED" ] && { USED=81920; TOT=81920; }
    FREE=$((TOT - USED))
    [ "$FREE" -ge "$NEED" ] && break
    echo "[$(date '+%F %T')] GPU$GPU wait $TAG: free ${FREE}MiB" >> "$PROG"
    sleep 300
  done
  mkdir -p "$RUNS/$TAG"
  if [ "$EXP" = "e2" ]; then
    EXTRA="--taps ${VAR//./,}"
  else
    EXTRA="--branch $VAR"
  fi
  echo "[$(date '+%F %T')] GPU$GPU start $TAG ($EXTRA)" >> "$PROG"
  CUDA_VISIBLE_DEVICES=$GPU SEER_ROOT=/cache/zihaohe/seer_cvpr $PY tools/seer_train_cvpr.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device cuda --arch v2 \
      --vjepa "$ARM" --seed "$SEED" $EXTRA \
      --iters "${ITERS:-2000}" --val_every 500 --save_every 2000 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] GPU$GPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] GPU$GPU FAILED $TAG" >> "$PROG"
  fi
done
echo "[$(date '+%F %T')] GPU$GPU queue finished" >> "$PROG"
