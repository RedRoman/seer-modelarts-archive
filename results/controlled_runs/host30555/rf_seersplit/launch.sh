#!/bin/bash
# Retinexformer retrain on SDSD indoor+outdoor union, Seer split. Local code copy so outputs stay on /cache.
R=/cache/zihaohe/runs/seer_iclr/rf_seersplit
ENV=/cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH=$ENV/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$R/Retinexformer
MIRROR=/temp/zihaohe/cache/checkpoint/runs/retinexformer_seersplit_mirror
EXP=$R/Retinexformer/experiments/RetinexFormer_SDSD_union_seersplit
mkdir -p $MIRROR
cd $R/Retinexformer
$ENV/bin/python basicsr/train.py --opt Options/RetinexFormer_SDSD_union_seersplit.yml >> $R/train.log 2>&1 &
PID=$!
echo "train pid $PID start $(date)" >> $R/train.log
# preemption fallback: mirror small checkpoints to shared disk (plain cp, no rename)
while kill -0 $PID 2>/dev/null; do
  sleep 600
  for d in models training_states; do
    [ -d $EXP/$d ] && mkdir -p $MIRROR/$d && cp -u $EXP/$d/* $MIRROR/$d/ 2>/dev/null
  done
done
wait $PID; RC=$?
for d in models training_states; do [ -d $EXP/$d ] && mkdir -p $MIRROR/$d && cp -u $EXP/$d/* $MIRROR/$d/ 2>/dev/null; done
cp -u $EXP/*.log $EXP/metric.csv $R/train.log $MIRROR/ 2>/dev/null
echo "TRAIN EXIT $RC $(date)" >> $R/train.log
