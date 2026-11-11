#!/bin/bash
# e18_cuda.sh GPU JOB [JOB...]   JOB = "name:flags..."
# Converged-protocol architecture wave. Every arm is the 22.4M branch at 20000
# iterations, the budget E17 showed is needed before architectures are
# distinguishable: at 2000 iterations one configuration's seeds spanned 0.91 dB
# while the effects under test were 0.11-0.68 dB.
set -u
GPU=$1; shift
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e18
PROG=$RUNS/e18_progress.log
NEED=${NEED:-14000}
mkdir -p "$RUNS"; cd $CODE
for JOB in "$@"; do
  TAG=${JOB%%:*}; FLAGS=$(echo "${JOB#*:}" | tr '+' ' ')
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  while :; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU 2>/dev/null)
    [ -z "$USED" ] && { USED=81920; TOT=81920; }
    [ $((TOT - USED)) -ge "$NEED" ] && break
    echo "[$(date '+%F %T')] GPU$GPU wait $TAG" >> "$PROG"; sleep 300
  done
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] GPU$GPU start $TAG ($FLAGS)" >> "$PROG"
  CUDA_VISIBLE_DEVICES=$GPU SEER_ROOT=/cache/zihaohe/seer_cvpr \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY tools/seer_train_cvpr.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device cuda --arch v2 \
      --branch small --vjepa pretrained $FLAGS \
      --iters "${ITERS:-20000}" --val_every 1000 --save_every 5000 --num_workers 4 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] GPU$GPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] GPU$GPU FAILED $TAG" >> "$PROG"
  fi
done
