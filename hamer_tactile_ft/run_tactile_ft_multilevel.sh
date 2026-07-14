#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

exec "$SCRIPT_DIR/run_tactile_ft.sh" \
    --exp_name mixed_dense_v2_multilevel \
    --tactile_head_type dense_v2_multilevel \
    --backbone_feature_layers 16,24,32 \
    "$@"
