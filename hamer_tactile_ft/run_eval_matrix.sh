#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Usage:
#   bash hamer_tactile_ft/run_eval_matrix.sh [EXP_NAME] [CKPT ...]
#
# Examples:
#   bash hamer_tactile_ft/run_eval_matrix.sh
#   bash hamer_tactile_ft/run_eval_matrix.sh mixed_dense_v2_repro rmse-best viou-best last

EXP_NAME="${1:-mixed_dense_v2_repro}"
if (( $# > 0 )); then
    shift
fi

if (( $# > 0 )); then
    CKPT_SELECTORS=("$@")
else
    CKPT_SELECTORS=(rmse-best)
fi

# Each entry is DATASET:SPLIT. Add or remove entries to change the eval matrix.
EVAL_TASKS=(
    "opentouch:test"
    "touchanything:test_seen"
    "touchanything:test_unseen"
)

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INDEX_WORKERS="${INDEX_WORKERS:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DINO_WEIGHTS="${DINO_WEIGHTS:-}"

safe_name() {
    local value="${1,,}"
    value="${value//[^a-z0-9_-]/_}"
    printf '%s' "$value"
}

SAFE_EXP_NAME="$(safe_name "$EXP_NAME")"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/eval_reports_${SAFE_EXP_NAME}}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

PIDS=()
LABELS=()
LOG_FILES=()

stop_children() {
    local pid
    echo
    echo "Stopping ${#PIDS[@]} eval process(es)..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_children INT TERM

cd "$WORKSPACE_DIR"

echo "Experiment: $EXP_NAME"
echo "Tactile head: loaded from experiment model_config.json"
echo "Checkpoints: ${CKPT_SELECTORS[*]}"
echo "GPUs per eval: $GPUS"
echo "Output root: $OUTPUT_ROOT"
echo

for ckpt in "${CKPT_SELECTORS[@]}"; do
    case "$ckpt" in
        rmse-best|viou-best|last|best) ;;
        *)
            echo "Invalid checkpoint selector '$ckpt'. Use rmse-best, viou-best, last, or best." >&2
            exit 2
            ;;
    esac

    for task in "${EVAL_TASKS[@]}"; do
        dataset="${task%%:*}"
        split="${task#*:}"
        label="${ckpt}/${dataset}/${split}"
        task_dir="$OUTPUT_ROOT/$ckpt/${dataset}_${split}"
        log_file="$LOG_DIR/$(safe_name "${ckpt}_${dataset}_${split}").log"
        mkdir -p "$task_dir"

        command=(
            "$PYTHON_BIN" hamer_tactile_ft/eval_tactile_fast.py
            --batch_size "$BATCH_SIZE"
            --gpus "$GPUS"
            --num_workers "$NUM_WORKERS"
            --index_workers "$INDEX_WORKERS"
            --save_diagnostics
            --exp_name "$EXP_NAME"
            --ckpt "$ckpt"
            --datasets "$dataset"
            --split "$split"
            --report_dir "$task_dir"
        )
        if [[ -n "$DINO_WEIGHTS" ]]; then
            command+=(--dino_weights "$DINO_WEIGHTS")
        fi

        echo "Starting $label"
        echo "  log: $log_file"
        "${command[@]}" >"$log_file" 2>&1 &
        PIDS+=("$!")
        LABELS+=("$label")
        LOG_FILES+=("$log_file")
    done
done

echo
echo "Started ${#PIDS[@]} eval process(es). Waiting for completion..."

failures=0
for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    label="${LABELS[$index]}"
    log_file="${LOG_FILES[$index]}"
    if wait "$pid"; then
        echo "[OK]     $label"
    else
        status=$?
        echo "[FAILED] $label (exit $status; log: $log_file)" >&2
        ((failures += 1))
    fi
done

trap - INT TERM

if (( failures > 0 )); then
    echo "$failures eval process(es) failed. See logs under $LOG_DIR." >&2
    exit 1
fi

echo "All eval processes completed successfully."
echo "Results: $OUTPUT_ROOT"
