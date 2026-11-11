#!/bin/bash
# H20 (Amendment 19): is the per-clip rank-4 projection an architectural
# improvement, or only an inference-time artefact? At inference it is +0.426 dB
# [+0.033, +0.853] over the full token field on three seeds, all agreeing. This
# trains with it. One variable against the converged baseline.
set -u
export SCRIPT=tools/seer_train_cvpr.py
cd /cache/zihaohe
FL="--arch+v2+--branch+small+--vjepa+pretrained+--token_proj+clip_rank+--token_proj_k+4"
./e19.sh 0 "e28_cliprank4_s0:${FL}+--seed+0" &
./e19.sh 1 "e28_cliprank4_s1:${FL}+--seed+1" &
wait
