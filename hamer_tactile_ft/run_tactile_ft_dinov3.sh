#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DINO_WEIGHTS="${DINO_WEIGHTS:-/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"

exec "$SCRIPT_DIR/run_tactile_ft.sh" \
    --exp_name mixed_dense_v2_dinov3_hplus \
    --visual_backbone dinov3_hplus \
    --tactile_head_type dense_v2 \
    --dino_weights "$DINO_WEIGHTS" \
    "$@"
