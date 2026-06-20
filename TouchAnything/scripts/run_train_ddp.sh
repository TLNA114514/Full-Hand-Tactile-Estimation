#!/bin/bash

# DDP multi-GPU training launcher
# Distributed training is started via torchrun
# GPU selection priority:
# 1. Command-line --gpus
# 2. Environment variable GPU_IDS
# 3. Default GPU_IDS defined in this script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/touchanything_with_glove_aug_wilor.yaml}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"

usage() {
    cat <<EOF
Usage:
  bash scripts/run_train_ddp.sh [--config <config.yaml>] [--gpus <gpu_ids>]

Examples:
  bash scripts/run_train_ddp.sh
  bash scripts/run_train_ddp.sh --gpus 0,1,2,3
  bash scripts/run_train_ddp.sh --config configs/hamer_pose_training_330_full.yaml --gpus 4,5

Notes:
  --gpus accepts physical GPU IDs separated by commas
  If --gpus is not provided, the script first checks the GPU_IDS environment variable
  If GPU_IDS is also unset, the script falls back to the default GPU_IDS defined here
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpus|--gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file does not exist: $CONFIG"
    exit 1
fi

if [ -z "$GPU_IDS" ]; then
    echo "Error: GPU_IDS is empty. Please provide it via --gpus or the GPU_IDS environment variable."
    exit 1
fi

# Count GPUs
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | sed '/^\s*$/d' | wc -l)

# Set visible GPUs
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "=================================================="
echo "Starting DDP Training"
echo "Config: $CONFIG"
echo "Physical GPUs: $GPU_IDS"
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

# Launch distributed training with torchrun
cd "$PROJECT_ROOT"
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    scripts/core/train.py \
    --config "$CONFIG"

echo "Training finished!"
