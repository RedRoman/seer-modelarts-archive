#!/bin/bash
# usage: eval_rf.sh <tag: union|indoor> <ckpt>
TAG=$1; CK=$2
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
PY=/cache/zihaohe/conda_envs/vjepa2-pip/bin/python
O=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
L=/cache/zihaohe/runs/seer_iclr/eval_${TAG}.log
cd /temp/zihaohe/workspace/baselines
echo "start $(date) ckpt=$CK" > $L
$PY run_retinexformer_seer_protocol.py --ckpt $CK --out $O/Retinexformer_${TAG}_seersplit_eval_v2.json >> $L 2>&1; echo "six-split rc=$? $(date)" >> $L
$PY run_retinexformer_seersplit_L0.py --ckpt $CK --out $O/Retinexformer_${TAG}_seersplit_L0seer_eval_v2.json >> $L 2>&1; echo "L0seer rc=$? $(date)" >> $L
echo EVAL_DONE >> $L
