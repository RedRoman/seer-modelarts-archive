#!/bin/bash
# Tier A+B sweep: lambda × ref_layer × early_stop + BVI-RLV eval + 50M decoder.
set -uo pipefail
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH="/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd /home/ma-user/workspace/zihaohe/vjepa_llve

CKPT_DIR=/cache/zihaohe/checkpoint/v11_probes
echo "[SWEEP-Q] start at $(date)"

# === A.1 BVI-RLV eval of E2b leader ===
echo "==== BVI-RLV eval of E2b leader $(date) ===="
mkdir -p $CKPT_DIR/c2_repa_haste_bvi_eval
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_eval_bvi.py \
    --ckpt $CKPT_DIR/c2_repa_haste/ckpt.pt \
    --out_dir $CKPT_DIR/c2_repa_haste_bvi_eval \
    2>&1 | tee $CKPT_DIR/c2_repa_haste_bvi_eval.log || true
echo "[BVI eval] done $(date)"

# === B.9 lambda sweep ===
for L in 0.1 0.3 0.7 1.5; do
    NAME="lam${L/./}"
    echo "==== [lambda=$L] $(date) ===="
    rm -rf $CKPT_DIR/c2_repa_sweep_$NAME
    CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
        --out_dir $CKPT_DIR/c2_repa_sweep_$NAME \
        --iters 1500 --crop 256 --log_every 50 \
        --lambda_repa $L --ref_layer 3 --early_stop_iter 900 \
        2>&1 | tee $CKPT_DIR/c2_repa_sweep_$NAME.log
    echo "[lam=$L] done $(date)"
done

# === B.10 ref_layer sweep (besides 3 already done) ===
for RL in 0 1 2; do
    NAME="ref$RL"
    echo "==== [ref_layer=$RL] $(date) ===="
    rm -rf $CKPT_DIR/c2_repa_sweep_$NAME
    CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
        --out_dir $CKPT_DIR/c2_repa_sweep_$NAME \
        --iters 1500 --crop 256 --log_every 50 \
        --lambda_repa 0.5 --ref_layer $RL --early_stop_iter 900 \
        2>&1 | tee $CKPT_DIR/c2_repa_sweep_$NAME.log
    echo "[ref=$RL] done $(date)"
done

# === B.11 early_stop sweep ===
for ES in 600 750 1200; do
    NAME="es$ES"
    echo "==== [early_stop=$ES] $(date) ===="
    rm -rf $CKPT_DIR/c2_repa_sweep_$NAME
    CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
        --out_dir $CKPT_DIR/c2_repa_sweep_$NAME \
        --iters 1500 --crop 256 --log_every 50 \
        --lambda_repa 0.5 --ref_layer 3 --early_stop_iter $ES \
        2>&1 | tee $CKPT_DIR/c2_repa_sweep_$NAME.log
    echo "[es=$ES] done $(date)"
done

echo "[SWEEP-Q complete] $(date)"
echo "=== ALL RESULTS ==="
for d in $CKPT_DIR/c2_repa_sweep_*; do
    [ -f $d/REPA_EVAL.md ] && echo "$(basename $d): $(grep 'no_prior PSNR' $d/REPA_EVAL.md | head -1)"
done
echo "BVI-RLV E2b: $(grep 'mean PSNR' $CKPT_DIR/c2_repa_haste_bvi_eval/BVI_RLV_EVAL.md 2>/dev/null | head -1)"
