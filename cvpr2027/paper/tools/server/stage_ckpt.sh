#!/bin/bash
# stage_ckpt.sh TAG -- once RUNS/TAG/final.pt exists and its eval JSON is on the shared disk, copy final.pt to /temp
set -u
TAG=$1
RUNS=/cache/zihaohe/runs/seer_iclr; SHARED=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
DST=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_ckpts; mkdir -p $DST
while [ ! -f $SHARED/${TAG}_eval_unified.json ]; do sleep 300; done
cp $RUNS/$TAG/final.pt $DST/${TAG}_final.pt.tmp && mv $DST/${TAG}_final.pt.tmp $DST/${TAG}_final.pt 2>/dev/null || cp $RUNS/$TAG/final.pt $DST/${TAG}_final.pt
echo "[$(date)] staged $DST/${TAG}_final.pt"
