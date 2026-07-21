#!/usr/bin/env bash
set -euo pipefail

python scripts/run_two_view_reconstruction.py \
    --config configs/two_view/real_pair.yaml
