#!/bin/bash
# chain.sh TAG [--no_vjepa] [NEXT_TAG]  -- wait for training TAG, eval final.pt, optionally launch NEXT_TAG (same arm, seed from tag suffix)
set -u
TAG=$1; FLAG=${2:-}; NEXT=${3:-}
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-} PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com
PY=/cache/zihaohe/conda_envs/vjepa2-pip/bin/python
RUNS=/cache/zihaohe/runs/seer_iclr
SHARED=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
mkdir -p $SHARED
cd /temp/zihaohe/workspace/vjepa_llve
while pgrep -f "tag $TAG " >/dev/null || pgrep -f "tag $TAG\$" >/dev/null; do sleep 300; done
echo "[$(date)] $TAG training finished"
cp $RUNS/$TAG/train_*.log $SHARED/ 2>/dev/null
if [ -n "$NEXT" ]; then
  SEED=${NEXT##*_s}
  mkdir -p $RUNS/$NEXT
  nohup $PY tools/v35_stage4_ctrl.py $FLAG --init_ckpt /cache/zihaohe/checkpoint/runs/v35_stage3/final.pt \
    --out_dir $RUNS/$NEXT --tag $NEXT --iters 1500 --audit_every 500 --save_at 750 --seed $SEED \
    > $RUNS/$NEXT/stdout.log 2>&1 &
  echo "[$(date)] launched $NEXT (seed $SEED)"
fi
if [ -f $RUNS/$TAG/final.pt ]; then
  $PY tools/seer_eval_unified.py --ckpt $RUNS/$TAG/final.pt --out $RUNS/$TAG/eval_unified.json $FLAG > $RUNS/$TAG/eval.log 2>&1
  cp $RUNS/$TAG/eval_unified.json $SHARED/${TAG}_eval_unified.json
  echo "[$(date)] $TAG eval done"
else
  echo "[$(date)] $TAG has no final.pt -- training failed?"
fi
if [ -n "$NEXT" ]; then
  while pgrep -f "tag $NEXT " >/dev/null || pgrep -f "tag $NEXT\$" >/dev/null; do sleep 300; done
  cp $RUNS/$NEXT/train_*.log $SHARED/ 2>/dev/null
  $PY tools/seer_eval_unified.py --ckpt $RUNS/$NEXT/final.pt --out $RUNS/$NEXT/eval_unified.json $FLAG > $RUNS/$NEXT/eval.log 2>&1
  cp $RUNS/$NEXT/eval_unified.json $SHARED/${NEXT}_eval_unified.json
  echo "[$(date)] $NEXT eval done"
fi
