#!/bin/bash
# e1_runner.sh NPU_ID JOB [JOB...]   JOB = "tag:arm:seed"
# Runs E1 jobs sequentially on one NPU. Waits until that NPU has enough free HBM
# (needs ~31GB; other tenants share these cards), re-checking every 5 min.
# ARCH (v1|v2) and ITERS come from the environment.
set -u
NPU=$1; shift
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
NF=/cache/zihaohe/seer_cvpr/nf.py
RUNS=/cache/zihaohe/seer_cvpr/runs
SHARED=/temp/zihaohe/seer_cvpr/results/e1
NEED_FREE=${NEED_FREE:-33000}      # MiB that must be free before starting
mkdir -p $RUNS $SHARED; cd $CODE
for JOB in "$@"; do
  TAG=${JOB%%:*}; REST=${JOB#*:}; ARM=${REST%%:*}; SEED=${REST##*:}
  [ -f $RUNS/$TAG/final.pt ] && { echo "[$(date '+%F %T')] skip $TAG (done)" >> $SHARED/e1_progress.log; continue; }
  while :; do
    USED=$(python3 $NF $NPU 2>/dev/null); [ -z "$USED" ] && USED=65536
    FREE=$((65536 - USED))
    [ "$FREE" -ge "$NEED_FREE" ] && break
    echo "[$(date '+%F %T')] NPU$NPU wait: used ${USED}MiB free ${FREE}MiB < ${NEED_FREE}" >> $SHARED/e1_progress.log
    sleep 300
  done
  mkdir -p $RUNS/$TAG
  echo "[$(date '+%F %T')] NPU$NPU start $TAG arch=${ARCH:-v2} arm=$ARM seed=$SEED (free ${FREE}MiB)" >> $SHARED/e1_progress.log
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/seer_train_cvpr.py \
      --tag $TAG --out_dir $RUNS/$TAG --arch ${ARCH:-v2} --vjepa $ARM --seed $SEED \
      --iters ${ITERS:-4000} --val_every 500 --save_every 2000 \
      > $RUNS/$TAG/stdout.log 2>&1 < /dev/null
  if [ -f $RUNS/$TAG/final.pt ]; then
    cp $RUNS/$TAG/train_*.log $SHARED/ 2>/dev/null
    echo "[$(date '+%F %T')] NPU$NPU DONE $TAG" >> $SHARED/e1_progress.log
  else
    echo "[$(date '+%F %T')] NPU$NPU FAILED $TAG" >> $SHARED/e1_progress.log
  fi
done
echo "[$(date '+%F %T')] NPU$NPU queue finished" >> $SHARED/e1_progress.log
