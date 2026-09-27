#!/bin/bash
# Run REPA variants V-A → V-B → V-C → V-D after expr_queue finishes.
set -uo pipefail
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH="/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd /home/ma-user/workspace/zihaohe/vjepa_llve

CKPT_DIR=/cache/zihaohe/checkpoint/v11_probes
LOG_DIR=$CKPT_DIR

echo "[REPA-Q] waiting for expr_queue..."
while tmux has-session -t expr_queue 2>/dev/null; do sleep 60; done
echo "[REPA-Q] expr_queue done at $(date)"

run_variant () {
    local NAME=$1
    local VARIANT=$2
    local EXTRA="${3:-}"
    echo "=========================================="
    echo "[$NAME] variant=$VARIANT extra='$EXTRA' start=$(date)"
    rm -rf $CKPT_DIR/c2_repa_$NAME
    CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_variants.py \
        --out_dir $CKPT_DIR/c2_repa_$NAME \
        --variant $VARIANT \
        --iters 1500 --crop 256 --log_every 50 \
        --lambda_repa 0.5 --ref_layer 3 \
        $EXTRA 2>&1 | tee $LOG_DIR/c2_repa_$NAME.log
    echo "[$NAME] done $(date)"
}

run_variant VA VA_delta ""
run_variant VB VB_contrastive "--margin 0.1"
run_variant VC VC_temporal ""
run_variant VD VD_multilayer "--multilayer_ref 2,3"

echo "=========================================="
echo "[REPA-Q complete] $(date)"
echo "Results:"
for v in VA VB VC VD; do
    echo "  $v: $CKPT_DIR/c2_repa_$v/REPA_VARIANT_EVAL.md"
done
