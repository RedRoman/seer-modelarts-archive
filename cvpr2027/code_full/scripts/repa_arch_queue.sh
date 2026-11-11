#!/bin/bash
# Run 8 REPA arch variants N1-N8 sequentially after repa_followup completes.
set -uo pipefail
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH="/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd /home/ma-user/workspace/zihaohe/vjepa_llve

CKPT_DIR=/cache/zihaohe/checkpoint/v11_probes

echo "[ARCH-Q] waiting for repa_followup..."
while tmux has-session -t repa_followup 2>/dev/null; do sleep 60; done
echo "[ARCH-Q] start at $(date)"

for V in N1_attn_map N2_predictive N3_cka N4_rkd_distance N5_bidirectional N6_local_window N7_self_distill N8_hard_token; do
    NAME=$(echo $V | tr '[:upper:]' '[:lower:]')
    echo "=========================================="
    echo "[$V] start $(date)"
    rm -rf $CKPT_DIR/c2_repa_arch_$NAME
    CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_arch.py \
        --out_dir $CKPT_DIR/c2_repa_arch_$NAME \
        --variant $V \
        --iters 1500 --crop 256 --log_every 50 \
        --lambda_repa 0.5 --ref_layer 3 --early_stop_iter 900 \
        2>&1 | tee $CKPT_DIR/c2_repa_arch_$NAME.log
    echo "[$V] done $(date)"
done

echo "=========================================="
echo "[ARCH-Q complete] $(date)"
echo "Results:"
for V in n1_attn_map n2_predictive n3_cka n4_rkd_distance n5_bidirectional n6_local_window n7_self_distill n8_hard_token; do
    grep "no_prior PSNR" $CKPT_DIR/c2_repa_arch_$V/REPA_ARCH_EVAL.md 2>/dev/null
done
