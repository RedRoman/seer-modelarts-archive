#!/bin/bash
# e16_runner.sh NPU JOB [JOB...]   JOB = "edcmode:seed"
#   edcmode  on | on_nodetach | off
# E16 / H13: the exposure decoupling was diagnosed, built and probed, but never
# switched on in a full model -- every run so far was --edc off. GT-mean
# normalisation is worth 4-7 dB cross-domain to every method tested, so this
# measures how much of that headroom an estimated per-scene gain recovers.
set -u
NPU=$1; shift
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e16
PROG=/temp/zihaohe/seer_cvpr/results/e16_progress.log
mkdir -p "$RUNS" "$(dirname "$PROG")"; cd $CODE
for JOB in "$@"; do
  MODE=${JOB%%:*}; SEED=${JOB##*:}
  TAG="e16_${MODE}_s${SEED}"
  [ -f "$RUNS/$TAG/final.pt" ] && { echo "[$(date '+%F %T')] skip $TAG" >> "$PROG"; continue; }
  case "$MODE" in
    on)          EXTRA="--edc on" ;;
    on_nodetach) EXTRA="--edc on --no_edc_detach" ;;
    off)         EXTRA="--edc off" ;;
    *) echo "bad mode $MODE" >> "$PROG"; continue ;;
  esac
  mkdir -p "$RUNS/$TAG"
  echo "[$(date '+%F %T')] NPU$NPU start $TAG ($EXTRA)" >> "$PROG"
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/seer_train_cvpr.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device npu --arch v2 \
      --vjepa pretrained --seed "$SEED" $EXTRA \
      --iters "${ITERS:-2000}" --val_every 500 --save_every 2000 \
      > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null
  if [ -f "$RUNS/$TAG/final.pt" ]; then
    echo "[$(date '+%F %T')] NPU$NPU DONE $TAG" >> "$PROG"
  else
    echo "[$(date '+%F %T')] NPU$NPU FAILED $TAG" >> "$PROG"
  fi
done
