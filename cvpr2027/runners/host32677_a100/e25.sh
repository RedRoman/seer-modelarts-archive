#!/bin/bash
# E25 / H16' (Amendment 16). Starts when E22 vacates GPUs 2 and 3 rather than
# adding a fifth and sixth job to four cards, which is what pushed the shared
# step time from 6.3 to 12 s and delayed E24 by an hour.
#
# One variable against E21: the gate is identical, the architecture is
# identical, and a fraction of training steps condition the branch on the
# previous clip's tokens so that closing the gate can lower the loss at all.
# Verified to fire before launch: a 120-step smoke read corrupt=239 of 800
# forward passes, 29.9% against the configured 30%.
set -u
RUNS=/cache/zihaohe/seer_cvpr/runs_e19
for f in "$RUNS/e22_tdelta_s0/final.pt" "$RUNS/e22_tdelta_s1/final.pt"; do
  while [ ! -s "$f" ]; do sleep 180; done
done
sleep 120
export SCRIPT=tools/seer_train_cvpr.py
cd /cache/zihaohe
FL='--arch+v2+--branch+small+--vjepa+pretrained+--xattn+gated+--token_corrupt+0.3'
setsid ./e19.sh 2 "e25_cgate_s0:${FL}+--seed+0" > /cache/zihaohe/e25_s0.log 2>&1 < /dev/null &
setsid ./e19.sh 3 "e25_cgate_s1:${FL}+--seed+1" > /cache/zihaohe/e25_s1.log 2>&1 < /dev/null &
echo "[$(date '+%F %T')] E25 launched on GPU 2 and 3"
