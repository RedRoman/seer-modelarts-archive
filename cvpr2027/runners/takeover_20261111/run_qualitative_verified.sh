#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/cann-8.5.2/set_env.sh
set -u
export SEER_ROOT=/temp/zihaohe/seer_cvpr
export RAFT_WEIGHTS=/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/temp/zihaohe/seer_cvpr/code/baselines:/temp/zihaohe/seer_cvpr/code/baselines/common:${PYTHONPATH:-}
TASK_ROOT=/cache/zihaohe/seer_takeover_20261111
test ! -e "$TASK_ROOT/qualitative_verified"
cd "$TASK_ROOT/code_full"
date -u '+START %Y-%m-%dT%H:%M:%SZ'
sha256sum tools/eval_e4.py "$TASK_ROOT/incoming/dump_qual_cvpr_verified.py"
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -B "$TASK_ROOT/incoming/dump_qual_cvpr_verified.py" \
  --ours /cache/zihaohe/seer_cvpr/runs_e17/e17_small_pretrained_s0/final_cpu.pt \
  --none /cache/zihaohe/seer_cvpr/runs_e17/e17_small_none_s0/final_cpu.pt \
  --retinexformer /temp/zihaohe/seer_cvpr/weights_small/checkpoint/runs/retinexformer_seer_protocol/experiments/RetinexFormer_SDSD_union_seer_protocol/models/net_g_160000.pth \
  --margins "$TASK_ROOT/incoming/qualitative_selection_margins.json" \
  --crop 256 --device npu:0 --out "$TASK_ROOT/qualitative_verified"
date -u '+END %Y-%m-%dT%H:%M:%SZ'
