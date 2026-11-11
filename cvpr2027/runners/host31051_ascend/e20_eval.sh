#!/bin/bash
# e20_eval.sh NPU HELDOUT ARM [HELDOUT ARM ...]
# Score the E20 cross-domain runs on the held-out domain test split. This is
# the retest of H5 at convergence; every earlier cross-domain number came from
# a 2000-2500 iteration model, the same protocol that produced four
# architecture verdicts which reversed at 20000.
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
NPU=$1; shift
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e20
OUT=/cache/zihaohe/seer_cvpr/results/e20_eval_test
mkdir -p "$OUT"; cd $CODE
while [ $# -ge 2 ]; do
  HO=$1; ARM=$2; shift 2
  TAG="e20_${HO}_${ARM}_s0"
  [ -f "$OUT/$TAG.json" ] && { echo "skip $TAG"; continue; }
  echo "[$(date "+%F %T")] NPU$NPU eval $TAG"
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/eval_e4.py \
    --ckpt "$RUNS/$TAG/final.pt" --heldout "$HO" --split test \
    --out "$OUT/$TAG.json" --device npu
done
