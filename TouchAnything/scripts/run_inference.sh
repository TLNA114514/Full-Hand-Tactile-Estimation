#!/bin/bash
# Tactile pressure-map inference script for automated batch inference
# Supports systematic evaluation across view configurations and dataset splits
# Supports lite runs with one sample per task

# Usage notes:
# 1. Full run: bash scripts/run_inference_mano.sh
# 2. Lite run: bash scripts/run_inference_mano.sh --lite
# 3. Custom arguments: bash scripts/run_inference_mano.sh --lite --gpu 0,1

# ========== Parse command-line arguments ==========
LITE_MODE=false
CUSTOM_GPU="2"
CUSTOM_BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --lite)
            LITE_MODE=true
            shift
            ;;
        --gpu)
            CUSTOM_GPU="$2"
            shift 2
            ;;
        --batch_size)
            CUSTOM_BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage:"
            echo "  bash scripts/run_inference_mano.sh [Options]"
            echo ""
            echo "Options:"
            echo "  --lite              Lite run (one sample per task)"
            echo "  --gpu GPU_IDS       Specify GPUs (for example: --gpu 0,1)"
            echo "  --batch_size SIZE   Specify batch size"
            echo "  -h, --help          Show help information"
            echo ""
            echo "Examples:"
            echo "  bash scripts/run_inference_mano.sh --lite"
            echo "  bash scripts/run_inference_mano.sh --lite --gpu 0,1 --batch_size 32"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use -h or --help for help"
            exit 1
            ;;
    esac
done

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate touch

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the project root
cd "$PROJECT_ROOT"

# Add the project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ========== Configuration parameters ==========
CHECKPOINT="/data_all/intern10/tmp/TouchAnything-Dev/checkpoints/20260421_094954_dinov2_vitb14_tactile_prediction/best_model.pth"
CONFIG="/data_all/intern10/TouchAnything/configs/touchanything_with_glove_aug_wilor.yaml"
OUTPUT_BASE="outputs/inference/5-12"

# GPU configuration, command-line arguments take priority
if [ -n "$CUSTOM_GPU" ]; then
    GPU_IDS="$CUSTOM_GPU"
else
    GPU_IDS="1"
fi

# Batch-size configuration, command-line arguments take priority
if [ -n "$CUSTOM_BATCH_SIZE" ]; then
    BATCH_SIZE="$CUSTOM_BATCH_SIZE"
else
    BATCH_SIZE=64
fi

NUM_WORKERS=2  # Set to the number of GPUs for multi-GPU parallel inference
FPS=30

# Set the number of trajectories based on the run mode
if [ "$LITE_MODE" = true ]; then
    NUM_TRAJ=1              # Lite mode: test one sample per task
    MODE_SUFFIX="_lite"
else
    NUM_TRAJ=1000           # Full mode: test all samples
    MODE_SUFFIX=""
fi

# ========== Optional feature switches ==========
SAVE_HDF5=false            # Whether to save HDF5 result files (true/false)
                           # true: Save full inference results to HDF5, uses substantial storage
                           # false: Save only videos and metrics, recommended

# Dataset splits
SPLITS=("test_unseen")

# View configurations; comment out views you do not need to test
VIEWS=("ego")
# VIEWS=("all")  # Use this line to test all views only

# ========== Automatically extract names ==========
# Extract the model name from the checkpoint path, using the checkpoint directory name
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
MODEL_NAME=$(basename "$CHECKPOINT_DIR")

# Extract the config name from the config path, removing the .yaml suffix
CONFIG_NAME=$(basename "$CONFIG" .yaml)

echo "=========================================="
echo "Batch Inference Configuration"
echo "=========================================="
echo "Run mode: $([ "$LITE_MODE" = true ] && echo "Lite mode (one sample per task)" || echo "Full mode (all samples)")"
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset splits: ${SPLITS[@]}"
echo "View configuration: ${VIEWS[@]}"
echo "GPU: $GPU_IDS"
echo "Batch size: $BATCH_SIZE"
echo "Worker processes: $NUM_WORKERS"
echo "Trajectories per split: $NUM_TRAJ"
echo "Save HDF5: $SAVE_HDF5"
echo "=========================================="
echo ""

# Lite-mode notice
if [ "$LITE_MODE" = true ]; then
    echo "⚡ Lite mode enabled"
    echo "   - Only the first sample of each task is tested"
    echo "   - Suitable for quick model validation"
    echo "   - For full evaluation, omit --lite"
    echo ""
fi

# ========== Batch inference loop ==========
for view in "${VIEWS[@]}"; do
    echo "---------- View configuration: $view ----------"
    
    for split in "${SPLITS[@]}"; do
        # Automatically generate output path: {config_name}/{model_name}/{view}/{data_split}
        # Lite mode adds the _lite suffix to the path
        OUTPUT_DIR="${OUTPUT_BASE}/${CONFIG_NAME}/${MODEL_NAME}/${view}/data_${split}${MODE_SUFFIX}"
        
        echo "Inference: $split split (view: $view)"
        echo "Output path: $OUTPUT_DIR"
        
        # Lite mode: generate a lite trajectory list
        LITE_TRAJ_FILE=""
        if [ "$LITE_MODE" = true ]; then
            # Extract split_file path from the config file
            SPLIT_FILE=$(grep "split_file:" "$CONFIG" | awk -F'"' '{print $2}')
            
            if [ -z "$SPLIT_FILE" ]; then
                echo "⚠️  Warning: Could not extract split_file from the config file; using the default path"
                SPLIT_FILE="datasets/TouchAnything_hdf5_clean/split.json"
            else
                echo "Extracted split_file from config: $SPLIT_FILE"
            fi
            
            # Generate lite trajectory list
            LITE_TRAJ_FILE="/tmp/lite_trajectories_${split}_$$.txt"
            echo "Generating lite trajectory list..."
            python scripts/utils/sample_lite_trajectories.py \
                --split_file "$SPLIT_FILE" \
                --split "$split" \
                --output "$LITE_TRAJ_FILE"
            
            if [ $? -ne 0 ]; then
                echo "✗ Failed to generate lite trajectory list"
                continue
            fi
            echo ""
        fi
        
        # Build command
        CMD="python scripts/core/inference_tactile_parallel.py \
            --checkpoint \"$CHECKPOINT\" \
            --config \"$CONFIG\" \
            --split \"$split\" \
            --num_traj $NUM_TRAJ \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --gpu_ids \"$GPU_IDS\" \
            --views \"$view\" \
            --output_dir \"$OUTPUT_DIR\" \
            --fps $FPS "
        
        # Lite mode: add trajectory-list argument
        if [ "$LITE_MODE" = true ] && [ -n "$LITE_TRAJ_FILE" ]; then
            CMD="$CMD --trajectory_list \"$LITE_TRAJ_FILE\""
        fi
        
        # If HDF5 is not saved, add --skip_hdf5
        if [ "$SAVE_HDF5" = false ]; then
            CMD="$CMD --skip_hdf5"
        fi
        
        # Execute command
        eval $CMD
        
        # Clean up temporary files
        if [ "$LITE_MODE" = true ] && [ -n "$LITE_TRAJ_FILE" ]; then
            rm -f "$LITE_TRAJ_FILE"
        fi
        
        if [ $? -eq 0 ]; then
            echo "✓ $split split inference complete (view: $view)"
        else
            echo "✗ $split split inference failed (view: $view)"
        fi
        echo ""
    done
    
    echo "---------- View $view complete ----------"
    echo ""
done

echo "=========================================="
echo "All inference tasks complete!"
echo "=========================================="
echo "Results saved under: $OUTPUT_BASE/$CONFIG_NAME/$MODEL_NAME/"
echo ""
# ========== Serial inference, original slower version ==========

# Usage notes:
# 1. Replace LATEST_CHECKPOINT with the actual checkpoint directory name
#    Examples: checkpoints/20260315_000238_touch_anything_tactile_prediction
# 2. If using example 2, replace SOME_FILE.hdf5 with the actual HDF5 filename
# 3. Ensure the config matches training, especially tactile_size
# 4. Output videos are saved to the directory specified by --output_dir

echo "Inference complete! Check the output directory for visualization videos."
