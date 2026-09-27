#!/bin/bash
# adapt_runner.sh TAG INIT_CKPT [--no_vjepa]
# Few-shot BVI adaptation (4 BVI *train* scenes, 200 steps, seed 0) from a controlled-study
# checkpoint, then evaluate on L2 BVI-RLV test and L0 (forgetting). Waits for an idle GPU.
set -u
TAG=$1; INIT=$2; FLAG=${3:-}
RUNS=/cache/zihaohe/runs/seer_iclr; SHARED=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
PY=/cache/zihaohe/conda_envs/vjepa2-pip/bin/python
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com
cd /temp/zihaohe/workspace/vjepa_llve
n=0; while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$u" -lt 8000 ]; then n=$((n+1)); [ $n -ge 2 ] && break; else n=0; fi; sleep 60; done
mkdir -p $RUNS/$TAG
echo "$(hostname) $(date) $TAG" >> $SHARED/jobs_started.txt
$PY tools/v35_stage4_ctrl.py $FLAG --init_ckpt $INIT --keep_gate_router --train_data bvi_fewshot \
    --out_dir $RUNS/$TAG --tag $TAG --iters 200 --audit_every 100 --save_at 100 --seed 0 \
    > $RUNS/$TAG/stdout.log 2>&1 < /dev/null
cp $RUNS/$TAG/train_*.log $SHARED/ 2>/dev/null
for CK in final step_100; do
  $PY tools/seer_eval_unified.py --ckpt $RUNS/$TAG/$CK.pt --out $RUNS/$TAG/eval_$CK.json \
      --splits L2_BVI-RLV L0_SDSD-indoor > $RUNS/$TAG/eval_$CK.log 2>&1 < /dev/null
  cp $RUNS/$TAG/eval_$CK.json $SHARED/${TAG}_${CK}_eval_unified.json
done
echo "[$(date)] $TAG done"
