#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: $0 PRESET_MODE [additional training options]" >&2
    exit 2
fi
shift

LOG_ROOT="${PREFLIGHT_LOG_ROOT:-$SCRIPT_DIR/preflight_logs}"
PREFLIGHT_NUM_WORKERS="${PREFLIGHT_NUM_WORKERS:-${NUM_WORKERS:-32}}"
PREFLIGHT_VAL_NUM_WORKERS="${PREFLIGHT_VAL_NUM_WORKERS:-${VAL_NUM_WORKERS:-16}}"
mkdir -p "$LOG_ROOT"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

run_stage() {
    local label="$1"
    local gpus="$2"
    local steps="$3"
    shift 3
    local log_file="$LOG_ROOT/${MODE}_${label}_${timestamp}.log"
    echo "Starting $label preflight: mode=$MODE, gpus=$gpus, steps=$steps"
    echo "Log: $log_file"
    WANDB_MODE=disabled "$SCRIPT_DIR/run_tactile_experiment.sh" "$MODE" \
        --gpus "$gpus" \
        --max_steps "$steps" \
        --skip_validation \
        --skip_checkpointing \
        --no-auto_resume \
        --num_workers "$PREFLIGHT_NUM_WORKERS" \
        --val_num_workers "$PREFLIGHT_VAL_NUM_WORKERS" \
        --exp_name "preflight_${MODE}_${label}" \
        "$@" 2>&1 | tee "$log_file"
}

run_stage single_gpu 0 20 "$@"
run_stage eight_gpu 0,1,2,3,4,5,6,7 100 "$@"

echo "Preflight complete. Compare peak_reserved against the 72 GiB limit:"
echo "  $LOG_ROOT/${MODE}_single_gpu_${timestamp}.log"
echo "  $LOG_ROOT/${MODE}_eight_gpu_${timestamp}.log"
