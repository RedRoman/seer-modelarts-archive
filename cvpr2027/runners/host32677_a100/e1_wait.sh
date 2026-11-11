#!/bin/bash
# e1_wait.sh GPU_ID JOB [JOB...]   JOB = "tag:arch:arm:seed"
# Like e1_cuda.sh but waits for free HBM first: these cards are shared with other
# tenants who occasionally take a whole 80 GB, which killed a lane outright.
set -u
GPU=$1; shift
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_cuda
PROG=$RUNS/e1_cuda_progress.log
NEED=${NEED:-34000}          # MiB that must be free before starting
mkdir -p "$RUNS"; cd $CODE
for JOB in "$@"; do
  TAG=${JOB%%:*}; R1=${JOB#*:}; ARCH=${R1%%:*}; R2=${R1#*:}; ARM=${R2%%:*}; SEED=${R2##*:}
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG (done)" >> "$PROG"; continue; }
  while :; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    [ -z "$USED" ] && { USED=81920; TOT=81920; }
    FREE=$((TOT - USED))
    [ "$FREE" -ge "$NEED" ] && break
    echo "[$(date '+%F %T')] GPU$GPU wait for $TAG: free ${FREE}MiB < ${NEED}" >> "$PROG"
    sleep 300
  done
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] GPU$GPU start $TAG arch=$ARCH arm=$ARM seed=$SEED (free ${FREE}MiB)" >> "$PROG"
  CUDA_VISIBLE_DEVICES=$GPU SEER_ROOT=/cache/zihaohe/seer_cvpr $PY tools/seer_train_cvpr.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device cuda \
      --arch "$ARCH" --vjepa "$ARM" --seed "$SEED" \
      --iters "${ITERS:-2000}" --val_every 500 --save_every 2000 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] GPU$GPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] GPU$GPU FAILED $TAG" >> "$PROG"
  fi
done
echo "[$(date '+%F %T')] GPU$GPU queue finished" >> "$PROG"
