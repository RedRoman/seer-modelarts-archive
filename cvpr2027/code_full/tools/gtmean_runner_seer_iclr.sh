#!/bin/bash
# gtmean_runner.sh TAG CKPT -- GT-mean diagnostic for one checkpoint (shares the GPU with training)
set -u
TAG=$1; CK=$2
SHARED=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals; OUT=/cache/zihaohe/runs/seer_iclr/gtmean; mkdir -p $OUT
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /temp/zihaohe/workspace/vjepa_llve
while [ ! -f $CK ]; do sleep 120; done
/cache/zihaohe/conda_envs/vjepa2-pip/bin/python tools/seer_eval_gtmean.py --ckpt $CK --out $OUT/${TAG}_gtmean.json > $OUT/${TAG}_gtmean.log 2>&1
cp $OUT/${TAG}_gtmean.json $SHARED/${TAG}_gtmean_eval.json
echo "[$(date)] $TAG gtmean done"
