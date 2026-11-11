#!/bin/bash
# Critical control: pure RGB + multi-seed E2b + REPA-only.
set -uo pipefail
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH="/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd /home/ma-user/workspace/zihaohe/vjepa_llve

CKPT_DIR=/cache/zihaohe/checkpoint/v11_probes
echo "[CTRL-Q] start at $(date)"

# 1) Pure RGB baseline (no V-JEPA at all)
echo "==== [pure_rgb] (lambda_repa=0, no V-JEPA) $(date) ===="
rm -rf $CKPT_DIR/c2_pure_rgb
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_pure_rgb \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 0 --rgb_weight 1.0 --early_stop_iter -1 \
    --seed 42 \
    2>&1 | tee $CKPT_DIR/c2_pure_rgb.log
echo "[pure_rgb] done $(date)"

# 2) Multi-seed E2b (seed=0)
echo "==== [E2b seed=0] $(date) ===="
rm -rf $CKPT_DIR/c2_repa_haste_seed0
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_repa_haste_seed0 \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 0.5 --rgb_weight 1.0 --ref_layer 3 --early_stop_iter 900 \
    --seed 0 \
    2>&1 | tee $CKPT_DIR/c2_repa_haste_seed0.log
echo "[seed=0] done $(date)"

echo "==== [E2b seed=100] $(date) ===="
rm -rf $CKPT_DIR/c2_repa_haste_seed100
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_repa_haste_seed100 \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 0.5 --rgb_weight 1.0 --ref_layer 3 --early_stop_iter 900 \
    --seed 100 \
    2>&1 | tee $CKPT_DIR/c2_repa_haste_seed100.log
echo "[seed=100] done $(date)"

# 3) REPA-only (no RGB Charbonnier)
echo "==== [repa_only] (rgb_weight=0, lambda_repa=1.0) $(date) ===="
rm -rf $CKPT_DIR/c2_repa_only
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_repa_only \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 1.0 --rgb_weight 0 --ref_layer 3 --early_stop_iter -1 \
    --seed 42 \
    2>&1 | tee $CKPT_DIR/c2_repa_only.log
echo "[repa_only] done $(date)"

echo "[CTRL-Q complete] $(date)"
echo "=== Results ==="
for d in $CKPT_DIR/c2_pure_rgb $CKPT_DIR/c2_repa_haste_seed0 $CKPT_DIR/c2_repa_haste_seed100 $CKPT_DIR/c2_repa_only; do
    [ -f $d/REPA_EVAL.md ] && echo "$(basename $d): $(grep 'no_prior PSNR' $d/REPA_EVAL.md | head -1)"
done
