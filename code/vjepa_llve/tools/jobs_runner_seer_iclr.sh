#!/bin/bash
# jobs_runner.sh "TAG ITERS [FLAGS...]" ...  -- run a fixed, host-assigned job list (no shared claims:
# mkdir on the SFS/NFS share proved not to be exclusive). Waits for the GPU to be idle before each job.
set -u
RUNS=/cache/zihaohe/runs/seer_iclr; SHARED=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
PY=/cache/zihaohe/conda_envs/vjepa2-pip/bin/python
LAUNCH_DEADLINE=$(date -d "2026-09-26 12:30" +%s)
GPU=${GPU:-0}
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $RUNS
cd /temp/zihaohe/workspace/vjepa_llve
wait_idle() {
  local n=0
  while true; do
    u=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$u" -lt 8000 ]; then n=$((n+1)); [ $n -ge 3 ] && return; else n=0; fi
    sleep 120
  done
}
for JOB in "$@"; do
  read -r TAG ITERS FLAGS <<< "$JOB"
  wait_idle
  [ "$(date +%s)" -gt "$LAUNCH_DEADLINE" ] && { echo "[$(date)] past launch deadline, skip $TAG"; continue; }
  SEED=${TAG##*_s}
  mkdir -p $RUNS/$TAG
  echo "$(hostname) $(date) $TAG" >> $SHARED/jobs_started.txt
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
    echo "[$(date)] FAILED $TAG (no final.pt)"
  fi
done
