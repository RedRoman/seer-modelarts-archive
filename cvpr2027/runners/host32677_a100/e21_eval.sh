#!/bin/bash
# H16: the trust gate, judged on both halves of its pre-registered rule.
#   (i)  does it help, against the converged ungated baseline
#   (ii) does it close when the tokens are made useless -- inference only,
#        independent of (i), so a gate that learns the right thing is
#        distinguishable from one that merely scores well
# The audit runs first because a gate still sitting at its initialisation would
# make both numbers meaningless, and this project has shipped a silently
# unloaded module before.
set -u
export SEER_ROOT=/cache/zihaohe/seer_cvpr
CODE=/cache/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_cu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e19
OUT=/cache/zihaohe/seer_cvpr/results/e21_eval_test
mkdir -p "$OUT"; cd $CODE
for s in 0 1; do
  while [ ! -s "$RUNS/e21_gated_s$s/final.pt" ]; do sleep 120; done
done
sleep 90
for s in 0 1; do
  CK=$RUNS/e21_gated_s$s/final.pt
  echo "=== gate audit seed $s"
  CUDA_VISIBLE_DEVICES=0 $PY tools/gate_audit.py --ckpt "$CK" --clips 8 \
      --out "$OUT/gate_audit_s$s.json"
  echo "=== eval seed $s, real tokens"
  CUDA_VISIBLE_DEVICES=0 $PY tools/eval_e4.py --ckpt "$CK" --heldout sdsd \
      --split test --out "$OUT/e21_gated_s$s.json" --device cuda
  echo "=== eval seed $s, random tokens (criterion ii)"
  CUDA_VISIBLE_DEVICES=1 $PY tools/eval_e4.py --ckpt "$CK" --heldout sdsd \
      --split test --out "$OUT/e21_gated_randtok_s$s.json" --device cuda \
      --force_arm random
done
echo "=== done"
