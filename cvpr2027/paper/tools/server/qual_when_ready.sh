#!/bin/bash
# qual_when_ready.sh -- when both s0 checkpoints are staged on /temp, dump the qualitative frames
set -u
C=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_ckpts
D=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals/qualitative_final
while [ ! -f $C/ctrl_vjepa_s0_final.pt ] || [ ! -f $C/ctrl_novjepa_s0_final.pt ]; do sleep 300; done
sleep 60
mkdir -p $D && cd /temp/zihaohe/workspace/vjepa_llve
export LD_LIBRARY_PATH=/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/cache/zihaohe/conda_envs/vjepa2-pip/bin/python tools/dump_qualitative.py --out $D \
  --seer "Seer=$C/ctrl_vjepa_s0_final.pt" --seer "Seer w/o V-JEPA=$C/ctrl_novjepa_s0_final.pt" > $D/dump.log 2>&1
echo "[$(date)] qualitative dump done: $D"
