#!/bin/bash
# run_tactile_ft.sh
# Script to run the tactile regression fine-tuning process.

# Get absolute path of this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Define paths
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT="$WORKSPACE_DIR/opentouch_hamer_ft/checkpoints/regression_only/best_ft_model.ckpt"

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
    --batch_size 16 \
    --epochs 30 \
    --exp_name "tactile_ft" \
    --use_wandb \
    "$@"
