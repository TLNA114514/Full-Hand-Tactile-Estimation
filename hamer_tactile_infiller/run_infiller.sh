#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT="$WORKSPACE_DIR/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
MANIFEST_DIR="$SCRIPT_DIR/manifests"

cd "$SCRIPT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found at $CHECKPOINT"
    echo "Please provide a valid checkpoint with --checkpoint or update CHECKPOINT in this script."
    exit 1
fi

echo "Starting sequence-level tactile infiller training..."
echo "Checkpoint: $CHECKPOINT"
echo "Manifest dir: $MANIFEST_DIR"

python train.py \
    --checkpoint "$CHECKPOINT" \
    --gpus "4" \
    --lr 1e-4 \
    --batch_size 16 \
    --epochs 90 \
    --num_workers 12 \
    --val_num_workers 8 \
    --persistent_workers \
    --prefetch_factor 4 \
    --check_val_every_n_epoch 1 \
    --use_wandb \
    --checkpoint_monitor val/eval_missing_bbox_mae \
    --checkpoint_mode min \
    --datasets opentouch,touchanything,egotactile \
    --manifest_dir "$MANIFEST_DIR" \
    --manifest_workers 128 \
    --egotactile_split_source extracted \
    --seq_len 16 \
    --seq_stride 8 \
    --eval_seq_stride 16 \
    --sample_frame_rate 1 \
    --min_observed_bbox 1 \
    --allow_missing_bbox \
    --mask_prob 0.5 \
    --target_policy has_tactile \
    --missing_bbox_weight 1.0 \
    --observed_bbox_weight 0.5 \
    --pressure_key_priority "continuous_subdiv>gaussian_pressure>original_hdf5_data" \
    --temporal_smooth_weight 0.05 \
    --active_pressure_thr 0.05 \
    --active_pressure_weight 1.5 \
    --active_pressure_gamma 0.7 \
    --active_pressure_max_weight 2.5 \
    --active_pressure_weight_warmup_epochs 10 \
    --pressure_loss_warmup_epochs 0 \
    --pressure_loss_warmup_start 1.0 \
    --opentouch_high_pressure_thr 0.9 \
    --opentouch_high_pressure_weight 0.3 \
    --exp_name mixed_ot_ta_ego_infiller \
    "$@"
