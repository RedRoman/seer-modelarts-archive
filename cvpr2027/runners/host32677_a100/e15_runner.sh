#!/bin/bash
# e15_runner.sh GPU JOB [JOB...]   JOB = "kvmode:seed"
# E15 / H12: does letting each query see beyond its own tubelet recover the
# cross-frame information E14 showed the fusion was discarding?
set -u
GPU=$1; shift
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e15
PROG=$RUNS/e15_progress.log
NEED=${NEED:-34000}
mkdir -p "$RUNS"; cd $CODE
for JOB in "$@"; do
  KV=${JOB%%:*}; SEED=${JOB##*:}
  TAG="e15_${KV}_s${SEED}"
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  while :; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    [ -z "$USED" ] && { USED=81920; TOT=81920; }
    [ $((TOT - USED)) -ge "$NEED" ] && break
    echo "[$(date '+%F %T')] GPU$GPU wait $TAG" >> "$PROG"; sleep 300
  done
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] GPU$GPU start $TAG (kv=$KV)" >> "$PROG"
  CUDA_VISIBLE_DEVICES=$GPU SEER_ROOT=/cache/zihaohe/seer_cvpr \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY tools/seer_train_cvpr.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device cuda --arch v2 \
      --vjepa pretrained --kv "$KV" --seed "$SEED" \
      --iters "${ITERS:-2000}" --val_every 500 --save_every 2000 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] GPU$GPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] GPU$GPU FAILED $TAG" >> "$PROG"
  fi
done
