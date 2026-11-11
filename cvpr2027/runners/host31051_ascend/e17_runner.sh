#!/bin/bash
# e17_runner.sh NPU JOB [JOB...]   JOB = "arm:branch:seed"
# E17 / H14: every architecture comparison so far was read off undertrained
# models. At 2000 iterations the three seeds of one configuration span 0.91 dB
# (23.82 / 23.44 / 22.91) while the effects being judged were 0.11-0.68 dB, and
# the validation curves of different architectures cross between 1000 and 2000.
# The protocol could not resolve the quantity it was being used to measure.
#
# This trains to convergence instead, so architectures can be compared on final
# quality rather than on how fast they happen to be learning at step 2000.
set -u
NPU=$1; shift
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e17
PROG=/temp/zihaohe/seer_cvpr/results/e17_progress.log
mkdir -p "$RUNS" "$(dirname "$PROG")"; cd $CODE
for JOB in "$@"; do
  ARM=${JOB%%:*}; R=${JOB#*:}; BR=${R%%:*}; SEED=${R##*:}
  TAG="e17_${BR}_${ARM}_s${SEED}"
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] NPU$NPU start $TAG" >> "$PROG"
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/seer_train_cvpr.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device npu --arch v2 \
      --vjepa "$ARM" --branch "$BR" --seed "$SEED" \
      --iters "${ITERS:-20000}" --val_every 1000 --save_every 5000 --num_workers 4 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] NPU$NPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] NPU$NPU FAILED $TAG" >> "$PROG"
  fi
done
