#!/bin/bash
# Automatic dataset split generation script with test_seen/test_unseen support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ========== Configuration ==========
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets/TouchAnything_hdf5_opensource_new}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MODE="${MODE:-full}"  # full | mini
TEST_UNSEEN_RATIO="${TEST_UNSEEN_RATIO:-0.05}"  # Ratio of test tasks allocated to test_unseen (0.2 = 20% of test tasks)

# ========== Mode 1: full split generation (default) ==========
# Generates split.json with train/val/test_seen/test_unseen
if [ "$MODE" = "full" ]; then
    python "$SCRIPT_DIR/core/create_dataset_split.py" \
        --data_root "$DATA_ROOT" \
        --output "$DATA_ROOT/split.json" \
        --num_workers "$NUM_WORKERS" \
        --wilor_min_valid_ratio 0.3 \
        --test_unseen_ratio "$TEST_UNSEEN_RATIO" \

    exit 0
fi

# ========== Mode 2: mini mode (quick code validation) ==========
# Generates split_mini.json with a small set of selected tasks for quick verification
if [ "$MODE" = "mini" ]; then
    python "$SCRIPT_DIR/core/create_dataset_split.py" \
        --data_root "$DATA_ROOT" \
        --output "$DATA_ROOT/split_mini.json" \
        --num_workers "$NUM_WORKERS" \
        --wilor_min_valid_ratio 0.3 \
        --test_unseen_ratio "$TEST_UNSEEN_RATIO" \
        --mini_tasks "pick_up_bottle" "pick_up_paper_box" "squeeze_toothpaste" "pick_up_earphones" "wipe_tableware_with_tissue" \
        --mini_max_trajs_per_task 10
    exit 0
fi

echo "Error: MODE must be 'full' or 'mini' (current: $MODE)"
exit 1

# ========== Additional examples ==========

# Example 3: Specify test_unseen tasks manually
# python scripts/core/create_dataset_split.py \
#     --data_root datasets/TouchAnything_hdf5 \
#     --output datasets/split.json \
#     --test_unseen_tasks "pick_up_bottle" "open_toolbox" "twist_bottle_cap" \
#     --num_workers 12

# Example 4: Custom ratios + test_unseen_ratio
# python scripts/core/create_dataset_split.py \
#     --data_root datasets/TouchAnything_hdf5 \
#     --output datasets/split.json \
#     --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \
#     --test_unseen_ratio 0.6 \
#     --num_workers 12

# Example 5: Mini mode + more trajectories
# MODE=mini TEST_UNSEEN_RATIO=0.5 bash scripts/run_create_split.sh
# Or manually:
# python scripts/core/create_dataset_split.py \
#     --data_root $DATA_ROOT \
#     --output $DATA_ROOT/split_mini_large.json \
#     --num_workers $NUM_WORKERS \
#     --wilor_min_valid_ratio 0.3 \
#     --test_unseen_ratio 0.5 \
#     --mini_tasks "pick_up_bottle" "pick_up_paper_box" "squeeze_toothpaste" "pick_up_earphones" "wipe_tableware_with_tissue" "pick_up_mouse" "pick_up_charger" "twist_bottle_cap" \
#     --mini_max_trajs_per_task 15

# ========== Environment Variables ==========
# DATA_ROOT: Dataset root directory (default: datasets/TouchAnything_hdf5_opensource)
# NUM_WORKERS: Number of parallel workers (default: 16)
# MODE: Split mode - 'full' or 'mini' (default: full)
# TEST_UNSEEN_RATIO: Ratio of test tasks for test_unseen (default: 0.5)
#
# Usage:
#   MODE=mini bash scripts/run_create_split.sh              # Quick test with 5 tasks
#   MODE=full bash scripts/run_create_split.sh              # Full dataset split
#   TEST_UNSEEN_RATIO=0.6 MODE=full bash scripts/run_create_split.sh  # Custom unseen ratio
