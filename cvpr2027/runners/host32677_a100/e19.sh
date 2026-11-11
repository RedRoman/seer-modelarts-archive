#!/bin/bash
# e19.sh GPU JOB [JOB...]   JOB = "tag:flags(+sep)"
# Two things the architecture wave displaced, both more consequential:
#   1. a third seed for the prior-vs-none comparison, whose PSNR effect at
#      convergence is +0.60 dB and not significant on two seeds
#   2. cross-domain arms at convergence -- every cross-domain conclusion so far
#      (H5, the motion gradient, "spatial-semantic not temporal") was read off
#      2000-2500 iteration models, the same protocol that produced four
#      architecture conclusions now known to be unreliable
set -u
GPU=$1; shift
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e19
PROG=$RUNS/e19_progress.log
mkdir -p "$RUNS"; cd $CODE
for JOB in "$@"; do
  TAG=${JOB%%:*}; FLAGS=$(echo "${JOB#*:}" | tr '+' ' ')
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  while :; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    T=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    [ -z "$U" ] && { U=81920; T=81920; }
    [ $((T - U)) -ge 14000 ] && break
    echo "[$(date '+%F %T')] GPU$GPU wait $TAG" >> "$PROG"; sleep 300
  done
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] GPU$GPU start $TAG ($FLAGS)" >> "$PROG"
  CUDA_VISIBLE_DEVICES=$GPU SEER_ROOT=/cache/zihaohe/seer_cvpr \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $SCRIPT \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device cuda $FLAGS \
      --iters 20000 --val_every 1000 --save_every 5000 --num_workers 4 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] GPU$GPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] GPU$GPU FAILED $TAG" >> "$PROG"
  fi
done
