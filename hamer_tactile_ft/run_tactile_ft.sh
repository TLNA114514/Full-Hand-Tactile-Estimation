#!/bin/bash
# run_tactile_ft.sh
# Script to run the tactile regression fine-tuning process.

# Get absolute path of this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Define paths
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT="$WORKSPACE_DIR/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"

# Change to the script's directory
cd "$SCRIPT_DIR"

# Ensure the checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Fine-tuned checkpoint not found at $CHECKPOINT"
    echo "Please train the base OpenTouch model first or provide a valid checkpoint path."
    exit 1
fi

echo "Starting tactile regression training using fine-tuned checkpoint..."
python train.py \
    --checkpoint "$CHECKPOINT" \
    --gpus "4" \
    --lr 1e-4 \
    --batch_size 32 \
    --epochs 90 \
    --use_wandb \
    --datasets opentouch,touchanything,egotactile \
    --exp_name mixed_ot_ta_ego \
    --index_cache_dir "$SCRIPT_DIR/index_cache" \
    --index_workers 32 \
    --index_chunksize 512 \
    --num_workers 32 \
    --active_pressure_thr 0.05 \
    --active_pressure_peak 0.10 \
    --active_pressure_high 0.30 \
    --background_pressure_thr 0.02 \
    --background_pred_margin 0.02 \
    --active_pressure_weight 1.0 \
    --active_pressure_gamma 1.0 \
    --background_loss_weight 1.0 \
    --volume_iou_loss_weight 0.0 \
    --opentouch_high_pressure_thr 0.9 \
    --opentouch_high_pressure_weight 0.3 \
    --loss_ramp_epochs 10 \
    "$@"
