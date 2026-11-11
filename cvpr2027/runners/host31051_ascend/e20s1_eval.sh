#!/bin/bash
# H5's second seed, scored the moment each cross-domain pair finishes.
# Seed 0 satisfied the rule's per-domain condition on SDSD and BVI via LPIPS
# with PSNR not significantly degraded; the all-seeds-agree half of ground
# rule 4 needs this seed before any of that can be claimed. RAFT is on the
# path this time, so these JSONs carry the temporal metrics seed 0's lack.
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e20
OUT=/cache/zihaohe/seer_cvpr/results/e20_eval_test
NPU=$1; shift
mkdir -p "$OUT"; cd $CODE
while [ $# -ge 2 ]; do
  HO=$1; ARM=$2; shift 2
  TAG="e20_${HO}_${ARM}_s1"
  while [ ! -s "$RUNS/$TAG/final.pt" ]; do sleep 300; done
  sleep 120
  echo "[$(date '+%F %T')] NPU$NPU eval $TAG"
  ASCEND_RT_VISIBLE_DEVICES=$NPU $PY tools/eval_e4.py \
    --ckpt "$RUNS/$TAG/final.pt" --heldout "$HO" --split test \
    --out "$OUT/$TAG.json" --device npu
done
echo "[$(date '+%F %T')] === done"
