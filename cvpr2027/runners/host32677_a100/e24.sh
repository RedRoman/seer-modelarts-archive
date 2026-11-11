#!/bin/bash
# E24: the registered fix to E22's loss (Amendment 14).
#
# E22's term is normalised by the target's delta magnitude, so a prediction
# with no temporal variation scores exactly 1.0 and nothing below that is
# reachable without reconstructing the motion. It took the cheap route:
# td_probe.py reads magnitude 0.202 at step 5000 against the converged
# baseline's 1.063, with alignment falling from +0.034 to +0.014. E24 keeps
# everything else fixed and changes only the form of that one term to one that
# is scale-invariant in the prediction, so shrinking cannot satisfy it.
#
# Weight stays at 0.3, matched to E22, because the two losses are on the same
# scale at the baseline's state (1.44 vs 1.03) and the form is the only
# variable this run is allowed to move.
#
# Starts when E21 vacates GPUs 0 and 1 rather than crowding four jobs onto
# two cards, which is what took the shared-card step time from 6.3 to 12.2 s.
set -u
RUNS=/cache/zihaohe/seer_cvpr/runs_e19
for f in "$RUNS/e21_gated_s0/final.pt" "$RUNS/e21_gated_s1/final.pt"; do
  while [ ! -s "$f" ]; do sleep 180; done
done
sleep 120
export SCRIPT=tools/seer_train_cvpr.py
cd /cache/zihaohe
FL='--arch+v2+--branch+small+--vjepa+pretrained+--tdelta+0.3+--tdelta_form+cosine'
nohup ./e19.sh 0 "e24_tdcos_s0:${FL}+--seed+0" > /cache/zihaohe/e24_s0.log 2>&1 &
nohup ./e19.sh 1 "e24_tdcos_s1:${FL}+--seed+1" > /cache/zihaohe/e24_s1.log 2>&1 &
echo "[$(date '+%F %T')] E24 launched on GPU 0 and 1"
