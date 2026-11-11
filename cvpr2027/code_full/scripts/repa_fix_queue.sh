#!/bin/bash
# Rerun broken N5/N6/N8 + all 6 R-more variants with safe_cos fix.
set -uo pipefail
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH="/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd /home/ma-user/workspace/zihaohe/vjepa_llve

CKPT_DIR=/cache/zihaohe/checkpoint/v11_probes

echo "[FIX-Q] start at $(date)"

# Rerun N5/N6/N8 (broken) — same script, fixed in-place
for V in N5_bidirectional N6_local_window N8_hard_token; do
    NAME=$(echo $V | tr '[:upper:]' '[:lower:]')
    echo "==== rerun [$V] $(date) ===="
    rm -rf $CKPT_DIR/c2_repa_arch_$NAME
    CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_arch.py \
        --out_dir $CKPT_DIR/c2_repa_arch_$NAME --variant $V \
        --iters 1500 --crop 256 --log_every 50 \
        --lambda_repa 0.5 --ref_layer 3 --early_stop_iter 900 \
        2>&1 | tee $CKPT_DIR/c2_repa_arch_$NAME.log
    echo "[$V] done $(date)"
done

# All 6 R-more variants with safe_cos
for M in R_GT R_blend R_random R_warmup R_multidepth R_no_proj; do
    NAME=$(echo $M | tr '[:upper:]' '[:lower:]')
    echo "==== [$M] $(date) ===="
    rm -rf $CKPT_DIR/c2_repa_more_$NAME
    CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_more_v2.py \
        --out_dir $CKPT_DIR/c2_repa_more_$NAME --mode $M \
        --iters 1500 --crop 256 --log_every 50 \
        --lambda_repa 0.5 --ref_layer 3 --early_stop_iter 900 \
        2>&1 | tee $CKPT_DIR/c2_repa_more_$NAME.log
    echo "[$M] done $(date)"
done

echo "[FIX-Q complete] $(date)"
echo "=== ALL RESULTS ==="
for d in $CKPT_DIR/c2_repa_arch_n5_bidirectional $CKPT_DIR/c2_repa_arch_n6_local_window $CKPT_DIR/c2_repa_arch_n8_hard_token; do
    [ -f $d/REPA_ARCH_EVAL.md ] && grep "no_prior PSNR" $d/REPA_ARCH_EVAL.md
done
for d in $CKPT_DIR/c2_repa_more_r_gt $CKPT_DIR/c2_repa_more_r_blend $CKPT_DIR/c2_repa_more_r_random $CKPT_DIR/c2_repa_more_r_warmup $CKPT_DIR/c2_repa_more_r_multidepth $CKPT_DIR/c2_repa_more_r_no_proj; do
    [ -f $d/REPA_MORE_EVAL.md ] && grep "no_prior PSNR" $d/REPA_MORE_EVAL.md
done
