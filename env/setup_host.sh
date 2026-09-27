#!/bin/bash
# Idempotent wanvj host setup. Safe to re-run.
# Usage: bash setup_host.sh
set -u

TAR=/temp/zihaohe/cache/conda_envs.tar
ENV_PREFIX=/cache/zihaohe/conda_envs/vjepa2-pip
HUB=/home/ma-user/.cache/torch/hub

echo "=== [1/4] path symlinks under /cache/zihaohe ==="
mkdir -p /cache/zihaohe
ln -sfn /temp/zihaohe/cache/checkpoint      /cache/zihaohe/checkpoint
ln -sfn /temp/zihaohe/cache/datasets        /cache/zihaohe/datasets
ln -sfn /temp/zihaohe/cache/visualizations  /cache/zihaohe/visualizations
ln -sfn /temp/zihaohe/cache/modelzoo        /cache/zihaohe/modelzoo
ls -l /cache/zihaohe/ | sed 's/^/    /'
for p in /cache/zihaohe/checkpoint/runs/v9_0b/config.yaml \
         /cache/zihaohe/modelzoo/vjepa2-main \
         /cache/zihaohe/checkpoint/wan21_t2v_1_3b; do
    [ -e "$p" ] && echo "    OK  $p" || echo "    MISSING  $p"
done

echo "=== [2/4] conda env vjepa2-pip ==="
if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "    already present, skipping extract"
else
    rm -rf /cache/zihaohe/conda_envs_tmp
    mkdir -p /cache/zihaohe/conda_envs_tmp
    echo "    extracting (this scans a 74G tar, ~10-20 min)..."
    ( cd /cache/zihaohe/conda_envs_tmp && tar xf "$TAR" conda_envs/vjepa2-pip ) || { echo "    TAR FAILED"; exit 1; }
    mkdir -p /cache/zihaohe/conda_envs
    mv /cache/zihaohe/conda_envs_tmp/conda_envs/vjepa2-pip /cache/zihaohe/conda_envs/
    rm -rf /cache/zihaohe/conda_envs_tmp
fi
du -sh "$ENV_PREFIX" | sed 's/^/    /'

echo "=== [3/4] /Users/rgupta workspace + torch hub cache ==="
# workspace root is shared with other tenants -- only touch our own entry
sudo mkdir -p /home/ma-user/workspace
if [ ! -e /home/ma-user/workspace/zihaohe ]; then
    sudo ln -sfn /temp/zihaohe/workspace /home/ma-user/workspace/zihaohe
fi
# torch hub: real dir, symlink each weight file in (pyiqa/lpips resolve by filename)
sudo mkdir -p "$HUB/checkpoints"
sudo chown -R ma-user "$HUB" 2>/dev/null || true
rm -rf "$HUB/checkpoints"
mkdir -p "$HUB/checkpoints"
n=0
for f in /temp/zihaohe/cache/hf_cache/torch/hub/checkpoints/*.pth; do
    [ -e "$f" ] || continue
    ln -sfn "$f" "$HUB/checkpoints/$(basename "$f")"
    n=$((n+1))
done
echo "    linked $n hub weights"
ls "$HUB/checkpoints/LPIPS_v0.1_alex-df73285e.pth" >/dev/null 2>&1 \
    && echo "    LPIPS OK" || echo "    LPIPS MISSING"
ls -lh /home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt 2>/dev/null \
    | awk '{print "    VJEPA ckpt OK", $5}'

echo "=== [4/4] import smoke test ==="
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
"$ENV_PREFIX/bin/python" - <<'PY' 2>&1 | grep -Ev "FutureWarning|warnings.warn" | sed 's/^/    /'
import torch, diffusers
from diffusers import AutoencoderKLWan, WanTransformer3DModel
print("torch", torch.__version__, "| ngpu", torch.cuda.device_count(),
      "| diffusers", diffusers.__version__, "| Wan classes OK")
import pyiqa, lpips, decord, cv2, timm
print("pyiqa/lpips/decord/cv2/timm OK")
PY
echo "=== SETUP DONE on $(hostname) ==="
