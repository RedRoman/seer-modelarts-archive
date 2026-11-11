#!/bin/bash
# Master experiment queue. Wait for c2_stage2_nohb, then run E4 → E1 → E2 sequentially.
set -uo pipefail

source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate /cache/zihaohe/conda_envs/vjepa2-pip
export LD_LIBRARY_PATH="/cache/zihaohe/conda_envs/vjepa2-pip/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd /home/ma-user/workspace/zihaohe/vjepa_llve

CKPT_DIR=/cache/zihaohe/checkpoint/v11_probes
LOG_DIR=$CKPT_DIR

echo "=========================================="
echo "[QUEUE] Waiting for c2_stage2_nohb to finish..."
while tmux has-session -t c2_stage2_nohb 2>/dev/null; do
    sleep 60
done
echo "[QUEUE] c2_stage2_nohb done at $(date)"

# ===== E4: HASTE early-stop =====
echo "=========================================="
echo "[E4 HASTE] Stage 2 with --rank_stop_iter 600 (stop ranking after 60% iters)"
echo "[E4 start] $(date)"
rm -rf $CKPT_DIR/c2_stage2_haste
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_stage2_train.py \
    --out_dir $CKPT_DIR/c2_stage2_haste \
    --stage1_ckpt $CKPT_DIR/c2_stage1/ckpt.pt \
    --iters 1000 --crop 256 --log_every 25 \
    --rank_stop_iter 600 \
    2>&1 | tee $LOG_DIR/c2_stage2_haste.log

CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_eval.py \
    --ckpt $CKPT_DIR/c2_stage2_haste/ckpt.pt \
    --out_dir $CKPT_DIR/c2_stage2_haste/eval \
    2>&1 | tee $LOG_DIR/c2_stage2_haste_eval.log
echo "[E4 done] $(date)"

# ===== E1: iREPA conv adapter (Stage 1 + Stage 2) =====
echo "=========================================="
echo "[E1 iREPA] Stage 1 (1500 iters, 3x3 conv projector + spatial norm)"
echo "[E1.1 start] $(date)"
rm -rf $CKPT_DIR/c2_stage1_irepa $CKPT_DIR/c2_stage2_irepa
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_stage1_irepa.py \
    --out_dir $CKPT_DIR/c2_stage1_irepa \
    --iters 1500 --crop 256 --log_every 50 \
    2>&1 | tee $LOG_DIR/c2_stage1_irepa.log

echo "[E1.2 Stage 2 iREPA] $(date)"
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_stage2_irepa.py \
    --out_dir $CKPT_DIR/c2_stage2_irepa \
    --stage1_ckpt $CKPT_DIR/c2_stage1_irepa/ckpt.pt \
    --iters 1000 --crop 256 --log_every 25 \
    2>&1 | tee $LOG_DIR/c2_stage2_irepa.log

CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_eval_irepa.py \
    --ckpt $CKPT_DIR/c2_stage2_irepa/ckpt.pt \
    --out_dir $CKPT_DIR/c2_stage2_irepa/eval \
    2>&1 | tee $LOG_DIR/c2_stage2_irepa_eval.log
echo "[E1 done] $(date)"

# ===== E2: REPA standalone (alignment loss only, no V-JEPA at inference) =====
echo "=========================================="
echo "[E2 REPA standalone] Decoder + alignment loss; V-JEPA dropped at inference"
echo "[E2 start] $(date)"
rm -rf $CKPT_DIR/c2_repa_standalone
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_repa_standalone \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 0.5 --ref_layer 3 \
    2>&1 | tee $LOG_DIR/c2_repa_standalone.log
echo "[E2 done] $(date)"

# ===== E2b: REPA with HASTE early-stop =====
echo "=========================================="
echo "[E2b REPA + HASTE early-stop at 60%] $(date)"
rm -rf $CKPT_DIR/c2_repa_haste
CUDA_VISIBLE_DEVICES=0 python -u tools/v11_c2_repa_train.py \
    --out_dir $CKPT_DIR/c2_repa_haste \
    --iters 1500 --crop 256 --log_every 50 \
    --lambda_repa 0.5 --ref_layer 3 \
    --early_stop_iter 900 \
    2>&1 | tee $LOG_DIR/c2_repa_haste.log
echo "[E2b done] $(date)"

echo "=========================================="
echo "[QUEUE COMPLETE] $(date)"
echo "Results:"
echo "  c2_stage2_nohb (DBB5.B):  $CKPT_DIR/c2_stage2_nohb/eval/C2_EVAL_SUMMARY.md"
echo "  c2_stage2_haste (E4):     $CKPT_DIR/c2_stage2_haste/eval/C2_EVAL_SUMMARY.md"
echo "  c2_stage2_irepa (E1):     $CKPT_DIR/c2_stage2_irepa/eval/C2_EVAL_SUMMARY.md"
echo "  c2_repa_standalone (E2):  $CKPT_DIR/c2_repa_standalone/REPA_EVAL.md"
echo "  c2_repa_haste (E2b):      $CKPT_DIR/c2_repa_haste/REPA_EVAL.md"
