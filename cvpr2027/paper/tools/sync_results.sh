#!/bin/sh
# Pull every eval JSON/log produced for the ICLR re-run from the shared SFS disk.
# Source of truth: /temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals/ (mounted on every cloud host).
# Usage: tools/sync_results.sh [port]   (default 30674; any running ModelArts host works)
set -eu
cd "$(dirname "$0")/.."
PORT=${1:-30674}
mkdir -p results/seer_iclr_evals
ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i /Users/hezihao/Desktop/KeyPair-liusonghua.pem -p "$PORT" ma-user@dev-modelarts-cnnorth9.huaweicloud.com \
  "cd /temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals && tar cf - --exclude=claims . 2>/dev/null" \
  | tar xf - -C results/seer_iclr_evals
ls results/seer_iclr_evals
