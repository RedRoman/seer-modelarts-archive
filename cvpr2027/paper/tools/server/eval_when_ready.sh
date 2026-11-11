#!/bin/bash
# eval_when_ready.sh TAG CKPT_BASENAME -- evaluate RUNS/TAG/CKPT once it exists (runs alongside training)
set -u
TAG=$1; CK=$2
RUNS=/cache/zihaohe/runs/seer_iclr; SHARED=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
PY=/cache/zihaohe/conda_envs/vjepa2-pip/bin/python
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com
cd /temp/zihaohe/workspace/vjepa_llve
while [ ! -f $RUNS/$TAG/$CK ]; do sleep 300; done
sleep 120   # let torch.save finish writing
N=${CK%.pt}
$PY tools/seer_eval_unified.py --ckpt $RUNS/$TAG/$CK --out $RUNS/$TAG/eval_${N}.json > $RUNS/$TAG/eval_${N}.log 2>&1
cp $RUNS/$TAG/eval_${N}.json $SHARED/${TAG}_${N}_eval_unified.json
echo "[$(date)] $TAG $CK eval done"
