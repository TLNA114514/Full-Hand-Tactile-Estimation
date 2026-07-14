#!/bin/bash
# run_tactile_ft.sh
# Script to run the tactile regression fine-tuning process.

# Get absolute path of this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Define paths
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT="$WORKSPACE_DIR/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-32}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

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
    --gpus "0,1,2,3,4,5,6,7" \
    --lr 5e-5 \
    --batch_size 64 \
    --epochs 60 \
    --use_wandb \
    --datasets opentouch,touchanything \
    --exp_name mixed_dense_v2_repro \
    --checkpoint_monitor val/eval_rmse \
    --checkpoint_mode min \
    --index_cache_dir "$SCRIPT_DIR/index_cache" \
    --index_workers 32 \
    --index_chunksize 512 \
    --num_workers "$NUM_WORKERS" \
    --val_num_workers "$VAL_NUM_WORKERS" \
    --persistent_workers \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --tactile_only_forward \
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
    --loss_ramp_epochs 10 \
    --frame_low_volume_thr 30.0 \
    --frame_high_volume_thr 150.0 \
    --opentouch_high_pressure_thr 0.9 \
    --opentouch_high_pressure_weight 0.3 \
    "$@"
