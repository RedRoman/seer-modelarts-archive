#!/bin/bash
# e20.sh NPU JOB [JOB...]   JOB = "heldout:arm:seed"
# Cross-domain at convergence. Every cross-domain conclusion in the paper so far
# -- H5's verdict, the motion gradient, "spatial-semantic rather than temporal"
# -- was read off 2000-2500 iteration models. That is the protocol which also
# produced four architecture conclusions that did not survive retesting at
# 20000, so the cross-domain claims cannot be trusted until they are rerun.
# 20000 iterations, 22.4M branch, leave-one-domain-out as in E4.
set -u
NPU=$1; shift
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e20
PROG=/temp/zihaohe/seer_cvpr/results/e20_progress.log
mkdir -p "$RUNS" "$(dirname "$PROG")"; cd $CODE
for JOB in "$@"; do
  HO=${JOB%%:*}; R=${JOB#*:}; ARM=${R%%:*}; SEED=${R##*:}
  TAG="e20_${HO}_${ARM}_s${SEED}"
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] NPU$NPU start $TAG" >> "$PROG"
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/seer_train_e4.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device npu \
      --heldout "$HO" --branch small --vjepa "$ARM" --seed "$SEED" \
      --iters 20000 --val_every 1000 --save_every 5000 --num_workers 4 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] NPU$NPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] NPU$NPU FAILED $TAG" >> "$PROG"
  fi
done
