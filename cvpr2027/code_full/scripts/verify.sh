#!/usr/bin/env bash
# Environment self-check before training.
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/verify_setup.py "$@"
