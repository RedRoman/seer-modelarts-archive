#!/bin/bash
# Follow-up after repa_variants finishes:
# 1) re-run V-B with bug fix
# 2) REPA + HASTE @ ref_layer=2 (V-JEPA layer 17 mid)
# 3) REPA + HASTE @ lambda=1.0 (stronger alignment)
# 4) REPA + HASTE long training (3000 iters)
set -uo pipefail
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH="/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd /home/ma-user/workspace/zihaohe/vjepa_llve

CKPT_DIR=/cache/zihaohe/checkpoint/v11_probes

echo "[FOLLOWUP] waiting for repa_variants..."
while tmux has-session -t repa_variants 2>/dev/null; do sleep 60; done
echo "[FOLLOWUP] start at $(date)"

# 1) V-B re-run (load_clip bug fixed)
echo "[VB-fixed] start $(date)"
rm -rf $CKPT_DIR/c2_repa_VB
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_variants.py \
    --out_dir $CKPT_DIR/c2_repa_VB --variant VB_contrastive \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 0.5 --ref_layer 3 --margin 0.1 \
    2>&1 | tee $CKPT_DIR/c2_repa_VB.log
echo "[VB-fixed] done $(date)"

# 2) REPA + HASTE at deeper-mid V-JEPA layer (idx 2 = layer 17)
echo "[REPA-mid-layer] start $(date)"
rm -rf $CKPT_DIR/c2_repa_haste_layer17
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_repa_haste_layer17 \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 0.5 --ref_layer 2 --early_stop_iter 900 \
    2>&1 | tee $CKPT_DIR/c2_repa_haste_layer17.log
echo "[REPA-mid-layer] done $(date)"

# 3) REPA + HASTE with stronger lambda
echo "[REPA-lambda1.0] start $(date)"
rm -rf $CKPT_DIR/c2_repa_haste_lambda1
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_repa_haste_lambda1 \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 1.0 --ref_layer 3 --early_stop_iter 900 \
    2>&1 | tee $CKPT_DIR/c2_repa_haste_lambda1.log
echo "[REPA-lambda1.0] done $(date)"

# 4) REPA + HASTE long training (best config so far, 3000 iter)
echo "[REPA-long] start $(date)"
rm -rf $CKPT_DIR/c2_repa_haste_3000
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_repa_haste_3000 \
    --iters 3000 --crop 256 --log_every 100 \
    --lambda_repa 0.5 --ref_layer 3 --early_stop_iter 1800 \
    2>&1 | tee $CKPT_DIR/c2_repa_haste_3000.log
echo "[REPA-long] done $(date)"

echo "[FOLLOWUP COMPLETE] $(date)"
echo "Results:"
for d in c2_repa_VB c2_repa_haste_layer17 c2_repa_haste_lambda1 c2_repa_haste_3000; do
    echo "  $d:"
    grep "no_prior PSNR" $CKPT_DIR/$d/REPA_VARIANT_EVAL.md $CKPT_DIR/$d/REPA_EVAL.md 2>/dev/null | head -1
done
