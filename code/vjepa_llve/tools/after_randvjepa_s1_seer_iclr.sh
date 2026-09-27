#!/bin/bash
# Wait for abl_randvjepa_s1 (train + eval) on this host, then run the novjepa_s1 adaptation.
C=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_ckpts
S=/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals
while [ ! -f $C/ctrl_novjepa_s1_final.pt ]; do sleep 60; done
while [ ! -f $S/abl_randvjepa_s1_eval_unified.json ]; do sleep 120; done
bash /temp/zihaohe/workspace/vjepa_llve/tools/adapt_runner_seer_iclr.sh adapt_novjepa_s1 $C/ctrl_novjepa_s1_final.pt --no_vjepa
