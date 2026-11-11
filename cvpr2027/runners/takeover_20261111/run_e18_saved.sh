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
cd "$TASK_ROOT/code_full"
for CASE in hybrid_s0 hybrid_s1 layers_all_s2; do
  test ! -e "$TASK_ROOT/results/e18_${CASE}_saved.json"
done
date -u '+START %Y-%m-%dT%H:%M:%SZ'
sha256sum tools/eval_e4.py tools/checkpoint_config.py models/pixel_branch_v2.py models/token_projection.py
/cache/zihaohe/seer_cvpr/venv_npu/bin/python -B - <<'PY'
import datetime, hashlib, json, os, pathlib, torch
root = pathlib.Path('/cache/zihaohe/seer_takeover_20261111')
checkpoints = {}
for name in ['e18_hybrid_s0', 'e18_hybrid_s1', 'e18_layers_all_npu_s0']:
    path = pathlib.Path('/cache/zihaohe/seer_cvpr/runs_e18') / name / 'final_cpu.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    checkpoints[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'step': ckpt['step'], 'saved_args': ckpt['args']}
source = ['tools/eval_e4.py', 'tools/checkpoint_config.py', 'models/pixel_branch_v2.py', 'models/token_projection.py', 'models/vjepa_backbone.py']
data = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'Existing final-step E18 checkpoints, saved reconstruction, all 39 TEST sequences; no selection or tuning', 'checkpoints': checkpoints, 'source_sha256': {name: hashlib.sha256((root/'code_full'/name).read_bytes()).hexdigest() for name in source}, 'visible_devices': os.environ['ASCEND_RT_VISIBLE_DEVICES'], 'device': 'npu:0', 'reconstruction': 'saved', 'heldout': 'sdsd', 'split': 'test', 'crop': 256, 'torch_version': torch.__version__}
(root/'results/e18_saved_provenance.json').write_text(json.dumps(data, indent=2)+'\n')
PY
for CASE in hybrid_s0 hybrid_s1 layers_all_s2; do
  case "$CASE" in
    hybrid_s0) CKPT=/cache/zihaohe/seer_cvpr/runs_e18/e18_hybrid_s0/final_cpu.pt ;;
    hybrid_s1) CKPT=/cache/zihaohe/seer_cvpr/runs_e18/e18_hybrid_s1/final_cpu.pt ;;
    layers_all_s2) CKPT=/cache/zihaohe/seer_cvpr/runs_e18/e18_layers_all_npu_s0/final_cpu.pt ;;
  esac
  date -u "+CASE $CASE START %Y-%m-%dT%H:%M:%SZ"
  /cache/zihaohe/seer_cvpr/venv_npu/bin/python -B tools/eval_e4.py \
    --ckpt "$CKPT" --heldout sdsd --split test --crop 256 --device npu \
    --reconstruction saved --out "$TASK_ROOT/results/e18_${CASE}_saved.json"
done
date -u '+END %Y-%m-%dT%H:%M:%SZ'
