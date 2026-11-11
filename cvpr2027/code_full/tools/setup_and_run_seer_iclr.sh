#!/bin/bash
# setup_and_run.sh TAG [--no_vjepa] -- fresh preemptible host: setup env, train TAG (seed from suffix), eval final.pt
set -u
TAG=$1; FLAG=${2:-}
bash /temp/zihaohe/setup_host.sh
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-} PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com
PY=/cache/zihaohe/conda_envs/vjepa2-pip/bin/python
RUNS=/cache/zihaohe/runs/seer_iclr; SHARED=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
mkdir -p $RUNS/$TAG $SHARED
cd /temp/zihaohe/workspace/vjepa_llve
SEED=${TAG##*_s}
echo "[$(date)] start $TAG seed=$SEED flag=$FLAG"
$PY tools/v35_stage4_ctrl.py $FLAG --init_ckpt /cache/zihaohe/checkpoint/runs/v35_stage3/final.pt \
  --out_dir $RUNS/$TAG --tag $TAG --iters 1500 --audit_every 500 --save_at 750 --seed $SEED > $RUNS/$TAG/stdout.log 2>&1
cp $RUNS/$TAG/train_*.log $SHARED/ 2>/dev/null
$PY tools/seer_eval_unified.py --ckpt $RUNS/$TAG/final.pt --out $RUNS/$TAG/eval_unified.json $FLAG > $RUNS/$TAG/eval.log 2>&1
cp $RUNS/$TAG/eval_unified.json $SHARED/${TAG}_eval_unified.json
echo "[$(date)] $TAG done"
