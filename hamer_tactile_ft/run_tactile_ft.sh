#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
DINO_WEIGHTS="${DINO_WEIGHTS:-$WORKSPACE_DIR/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-32}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"

if [[ "$PERSISTENT_WORKERS" == "0" ]]; then
    PERSISTENT_WORKERS_ARG="--no-persistent_workers"
else
    PERSISTENT_WORKERS_ARG="--persistent_workers"
fi

cd "$SCRIPT_DIR"

if [[ ! -f "$DINO_WEIGHTS" ]]; then
    echo "Error: DINOv3 weights not found at $DINO_WEIGHTS" >&2
    exit 1
fi

echo "Starting tactile regression training..."

# These are fallback defaults for direct invocation. Arguments supplied by a
# preset or by the user are appended last and therefore take precedence.
exec python train.py \
    --dino_weights "$DINO_WEIGHTS" \
    --visual_backbone dinov3_hplus \
    --tactile_head_type dense_v2_dino_rezero \
    --backbone_feature_layers 8,16,24,32 \
    --pool_layout fullgrid32 \
    --gpus "0,1,2,3,4,5,6,7" \
    --lr 5e-5 \
    --batch_size 128 \
    --epochs 60 \
    --use_wandb \
    --datasets opentouch,touchanything \
    --exp_name mixed_dense_v2_dinov3_rezero_fullgrid32 \
    --index_cache_dir "$SCRIPT_DIR/index_cache" \
    --index_workers 32 \
    --index_chunksize 512 \
    --num_workers "$NUM_WORKERS" \
    --val_num_workers "$VAL_NUM_WORKERS" \
    "$PERSISTENT_WORKERS_ARG" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --gradient_clip_val 1.0 \
    --trainer_precision "$TRAINER_PRECISION" \
    --seed 521 \
    --lr_warmup_epochs 3 \
    --tactile_loss_scale 10.0 \
    --check_val_every_n_epoch 1 \
    --active_pressure_thr 0.05 \
    --active_pressure_peak 0.10 \
    --active_pressure_high 0.30 \
    --background_pressure_thr 0.02 \
    --background_pred_margin 0.02 \
    --active_pressure_weight 1.0 \
    --active_pressure_gamma 1.0 \
    --background_loss_weight 1.0 \
    --logit_bce_weight 0.1 \
    --loss_ramp_epochs 5 \
    --frame_low_volume_thr 30.0 \
    --frame_high_volume_thr 150.0 \
    --opentouch_high_pressure_thr 0.9 \
    --opentouch_high_pressure_weight 0.3 \
    "$@"
