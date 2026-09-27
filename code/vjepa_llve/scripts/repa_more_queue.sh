#!/bin/bash
# Run 6 more REPA designs after repa_arch finishes.
set -uo pipefail
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH="/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd /home/ma-user/workspace/zihaohe/vjepa_llve

CKPT_DIR=/cache/zihaohe/checkpoint/v11_probes

echo "[MORE-Q] waiting for repa_arch..."
while tmux has-session -t repa_arch 2>/dev/null; do sleep 60; done
echo "[MORE-Q] start at $(date)"

for M in R_GT R_blend R_random R_warmup R_multidepth R_no_proj; do
    NAME=$(echo $M | tr '[:upper:]' '[:lower:]')
    echo "=========================================="
    echo "[$M] start $(date)"
    rm -rf $CKPT_DIR/c2_repa_more_$NAME
    CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_more.py \
        --out_dir $CKPT_DIR/c2_repa_more_$NAME \
        --mode $M \
        --iters 1500 --crop 256 --log_every 50 \
        --lambda_repa 0.5 --ref_layer 3 --early_stop_iter 900 \
        2>&1 | tee $CKPT_DIR/c2_repa_more_$NAME.log
    echo "[$M] done $(date)"
done

echo "=========================================="
echo "[MORE-Q complete] $(date)"
for M in r_gt r_blend r_random r_warmup r_multidepth r_no_proj; do
    f=$CKPT_DIR/c2_repa_more_$M/REPA_MORE_EVAL.md
    [ -f $f ] && grep "no_prior PSNR" $f
done
