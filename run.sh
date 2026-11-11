#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

seed="${1:-42}"
PYTHONHASHSEED="$seed" python train.py --seed "$seed"
PYTHONHASHSEED="$seed" python evaluate.py --seed "$seed"
