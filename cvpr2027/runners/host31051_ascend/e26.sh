#!/bin/bash
# H18 (Amendment 19): does the low-rank sufficiency survive training, as
# opposed to being an inference-time property of a model trained on the full
# field? Rank 64 is the smallest fixed rank the screen supports -- global
# rank 4 and 16 cost 4.02 and 3.59 dB, within noise of destroying the prior.
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
CODE=/temp/zihaohe/seer_cvpr/code/vjepa_llve
PY=/cache/zihaohe/seer_cvpr/venv_npu/bin/python
RUNS=/cache/zihaohe/seer_cvpr/runs_e26
BASIS=/cache/zihaohe/seer_cvpr/results/token_basis_train.json
mkdir -p "$RUNS"; cd $CODE
for S in 0 1; do
  TAG="e26_grank64_s$S"
  [ -f "$RUNS/$TAG/final.pt" ] && continue
  mkdir -p "$RUNS/$TAG"
  ASCEND_RT_VISIBLE_DEVICES=$S $PY tools/seer_train_cvpr.py \
      --tag "$TAG" --out_dir "$RUNS/$TAG" --device npu \
      --arch v2 --branch small --vjepa pretrained \
      --token_proj global_rank --token_proj_k 64 --token_basis "$BASIS" \
      --seed $S --iters 20000 --val_every 1000 --save_every 5000 \
      --num_workers 4 > "$RUNS/$TAG/stdout.log" 2>&1 < /dev/null &
done
wait
echo "=== e26 done"
