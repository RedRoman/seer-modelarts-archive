#!/bin/bash
# queue_runner.sh -- claim-based job runner shared by all hosts.
# Queue file lines: TAG ITERS [extra flags...]   (seed = suffix after _s in TAG)
# A job is claimed atomically with mkdir on the shared disk; the runner waits
# for its GPU to be idle, trains, evaluates final.pt, then claims the next job.
set -u
Q=/temp/zihaohe/workspace/vjepa_llve/tools/seer_iclr_queue.txt
SHARED=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
CLAIMS=$SHARED/claims
RUNS=/cache/zihaohe/runs/seer_iclr
PY=/cache/zihaohe/conda_envs/vjepa2-pip/bin/python
LAUNCH_DEADLINE=$(date -d "2026-09-26 08:00" +%s)
GPU=${GPU:-0}
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $CLAIMS $RUNS
while [ ! -x $PY ]; do sleep 60; done
cd /temp/zihaohe/workspace/vjepa_llve

wait_idle() {
  local n=0
  while true; do
    u=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$u" -lt 8000 ]; then n=$((n+1)); [ $n -ge 3 ] && return; else n=0; fi
    sleep 120
  done
}

while true; do
  wait_idle
  [ "$(date +%s)" -gt "$LAUNCH_DEADLINE" ] && { echo "[$(date)] past launch deadline, exiting"; exit 0; }
  JOB=""
  while read -r TAG ITERS FLAGS; do
    [ -z "${TAG:-}" ] && continue
    [[ $TAG == \#* ]] && continue
    if mkdir $CLAIMS/$TAG 2>/dev/null; then JOB="$TAG"; break; fi
  done < $Q
  if [ -z "$JOB" ]; then echo "[$(date)] queue empty, sleeping"; sleep 600; continue; fi
  SEED=${TAG##*_s}
  echo "$(hostname) $(date)" > $CLAIMS/$TAG/host
  mkdir -p $RUNS/$TAG
  echo "[$(date)] train $TAG iters=$ITERS seed=$SEED flags=$FLAGS"
  $PY tools/v35_stage4_ctrl.py $FLAGS --init_ckpt /cache/zihaohe/checkpoint/runs/v35_stage3/final.pt \
      --out_dir $RUNS/$TAG --tag $TAG --iters $ITERS --audit_every 250 --seed $SEED \
      > $RUNS/$TAG/stdout.log 2>&1 < /dev/null
  cp $RUNS/$TAG/train_*.log $SHARED/ 2>/dev/null
  if [ -f $RUNS/$TAG/final.pt ]; then
    $PY tools/seer_eval_unified.py --ckpt $RUNS/$TAG/final.pt --out $RUNS/$TAG/eval_unified.json \
        > $RUNS/$TAG/eval.log 2>&1 < /dev/null
    cp $RUNS/$TAG/eval_unified.json $SHARED/${TAG}_eval_unified.json
    echo "[$(date)] done $TAG"
  else
    echo "[$(date)] FAILED $TAG (no final.pt)"; echo failed > $CLAIMS/$TAG/status
  fi
done
