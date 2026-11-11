#!/bin/bash
# e4_cuda.sh GPU_ID JOB [JOB...]    JOB = "heldout:arm:seed"
# Leave-one-domain-out runs for H5, sequentially on one GPU. ITERS from env.
set -u
GPU=$1; shift
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e4
PROG=$RUNS/e4_progress.log
mkdir -p "$RUNS"; cd $CODE
for JOB in "$@"; do
  HO=${JOB%%:*}; R=${JOB#*:}; ARM=${R%%:*}; SEED=${R##*:}
  TAG="e4_${HO}_${ARM}_s${SEED}"
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] skip $TAG (done)" >> "$PROG"; continue
  fi
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] GPU$GPU start $TAG heldout=$HO arm=$ARM seed=$SEED" >> "$PROG"
  CUDA_VISIBLE_DEVICES=$GPU SEER_ROOT=/cache/zihaohe/seer_cvpr $PY tools/seer_train_e4.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device cuda \
      --heldout "$HO" --vjepa "$ARM" --seed "$SEED" \
      --iters "${ITERS:-2500}" --val_every 500 --save_every 2500 --num_workers 6 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] GPU$GPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] GPU$GPU FAILED $TAG" >> "$PROG"
  fi
done
echo "[$(date '+%F %T')] GPU$GPU E4 queue finished" >> "$PROG"
