#!/bin/bash
set -e

export PYTHONFAULTHANDLER=1
# Prevent OpenMP/MKL from spawning threads in DataLoader worker processes after fork()
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Change directory to script folder
cd "$(dirname "$0")"

# Help message
show_help() {
    echo "Usage: ./run_ft.sh [options]"
    echo ""
    echo "Options:"
    echo "  --test        Run in verification mode (fast extraction of 1 clip and 1 epoch training on a tiny subset)"
    echo "  --gpus [IDs]  GPU indices to use, comma-separated (default: 4. Example: --gpus 4,5)"
    echo "  --epochs [N]  Number of training epochs (default: 10)"
    echo "  --lr [val]    Base learning rate per GPU (default: 1e-5)"
    echo "  --help        Show this help message"
}

# Defaults
GPUS="4"
EPOCHS="10"
LR="1e-5"
TEST_MODE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test) TEST_MODE=true; shift ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; show_help; exit 1 ;;
    esac
done

# Parse GPUs
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# Select isolated BBox JSON file name for test mode to prevent overwriting master data
BBOX_JSON_FILE="$(pwd)/opentouch_train_val_bboxes.json"
if [ "$TEST_MODE" = true ]; then
    BBOX_JSON_FILE="$(pwd)/opentouch_train_val_bboxes_test.json"
fi

echo "=========================================================="
echo "    Multi-GPU Parallel Hamer Fine-tuning on OpenTouch"
echo "=========================================================="
echo "GPU List      : $GPUS (Total: $NUM_GPUS active GPUs)"
echo "Test Mode     : $TEST_MODE"
echo "BBox Cache File: $BBOX_JSON_FILE"
if [ "$TEST_MODE" = false ]; then
    echo "Epochs        : $EPOCHS"
    echo "Base LR       : $LR (Linear Scaled to $((NUM_GPUS))x: $(python3 -c "print($LR * $NUM_GPUS)"))"
fi
echo "=========================================================="
echo ""

# Configure sample flags for test mode
SAMPLE_FLAG=""
QUICK_TEST_FLAG=""
if [ "$TEST_MODE" = true ]; then
    SAMPLE_FLAG="--sample_only"
    QUICK_TEST_FLAG="--quick_test"
    # Under test mode, force single process to prevent output truncation
    NUM_GPUS=1
    GPU_LIST=(${GPU_LIST[0]})
    GPUS=${GPU_LIST[0]}
fi

# Step 1: Parallel BBox extraction
echo ">>> Step 1: Pre-extracting hand BBoxes (Offline, Multi-GPU Parallel)..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${GPU_LIST[$i]}
    echo "  [Process $i/$NUM_GPUS] Starting extraction on GPU $GPU_ID (logging to extract_gpu_${GPU_ID}.log)..."
    # Launch in the background with absolute path and output redirection
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 "$(pwd)/extract_bboxes.py" \
        --gpu "$GPU_ID" \
        --gpu_idx "$i" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$BBOX_JSON_FILE" \
        $SAMPLE_FLAG > "extract_gpu_${GPU_ID}.log" 2>&1 &
done

# Wait for all background subprocesses to finish
wait

# Validation Check: Ensure at least one temp GPU JSON file exists before merging
# If not, it means the extraction crashed/exited immediately!
TEMP_FILES=(${BBOX_JSON_FILE}.gpu_*)
if [ ! -e "${TEMP_FILES[0]}" ]; then
    echo "❌ Error: BBox extraction failed! No temporary GPU JSON files (.gpu_*) were created."
    echo "This usually indicates that the background extraction processes crashed."
    echo "Displaying the logs for diagnostics:"
    for GPU_ID in "${GPU_LIST[@]}"; do
        LOG_FILE="extract_gpu_${GPU_ID}.log"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "=========================================================="
            echo "--- Log for GPU $GPU_ID ($LOG_FILE) ---"
            cat "$LOG_FILE"
            echo "=========================================================="
        fi
    done
    exit 1
fi

# Merge the extracted JSON caches
echo ">>> All GPU extraction tasks completed. Merging cache files..."
python3 "$(pwd)/extract_bboxes.py" --output_json "$BBOX_JSON_FILE" --merge

# Clean up successful run log files to keep workspace tidy
rm -f extract_gpu_*.log

# Step 2: Fine-tuning training
echo ""
echo ">>> Step 2: Running Hamer Fine-tuning (Multi-GPU DDP)..."
if [ "$TEST_MODE" = true ]; then
    python3 "$(pwd)/train.py" \
        --gpus "$GPUS" \
        --lr "$LR" \
        --epochs 1 \
        --bbox_json "$BBOX_JSON_FILE" \
        $QUICK_TEST_FLAG
else
    python3 "$(pwd)/train.py" \
        --gpus "$GPUS" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --bbox_json "$BBOX_JSON_FILE" \
        --batch_size 16 \
        --num_workers 4
fi

# Step 3: Cleanup if we are in test mode
if [ "$TEST_MODE" = true ]; then
    echo ""
    echo "=========================================================="
    echo ">>> Cleanup: Removing test mode temporary files..."
    rm -f "$BBOX_JSON_FILE"
    rm -rf checkpoints_test
    echo "Temporary test files (BBox JSON and checkpoints_test folder) successfully deleted."
fi

echo ""
echo "=========================================================="
echo "🎉 Multi-GPU Parallel Pipeline Completed Successfully!"
echo "=========================================================="
if [ "$TEST_MODE" = false ]; then
    echo "Best checkpoint is saved at: opentouch_hamer_ft/checkpoints/best_ft_model.ckpt"
    echo ""
    echo "To evaluate this fine-tuned checkpoint on the test set, run:"
    echo "python3 ../evaluation/eval_hamer.py \\"
    echo "  --checkpoint checkpoints/best_ft_model.ckpt \\"
    echo "  --split test \\"
    echo "  --gpu ${GPU_LIST[0]}"
fi
echo "=========================================================="
