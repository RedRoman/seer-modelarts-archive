#!/usr/bin/env bash
# v5: V-JEPA-conditioned latent flow matching for LLVE.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
python train_diffusion.py --config configs/train_v5_diffusion.yaml "$@"
