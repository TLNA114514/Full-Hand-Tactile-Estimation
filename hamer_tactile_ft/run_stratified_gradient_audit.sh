#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
EXP_NAME="${EXP_NAME:-mixed_zero_ordinal_residual_v19_condnll}"
GPUS_CSV="${GPUS:-0,1,2}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tactile}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCHES_PER_STRATUM="${BATCHES_PER_STRATUM:-16}"
AGGREGATE_BATCHES="${AGGREGATE_BATCHES:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
INDEX_WORKERS="${INDEX_WORKERS:-256}"
VOLUME_WORKERS="${VOLUME_WORKERS:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/eval_reports_${EXP_NAME}/gradient_audits/stratified}"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/index_cache/${EXP_NAME}_touchanything_strata_bs${BATCH_SIZE}_nb${BATCHES_PER_STRATUM}.json}"

IFS=',' read -r -a GPU_IDS <<< "$GPUS_CSV"
if (( ${#GPU_IDS[@]} == 0 )); then
    echo "GPUS must contain at least one GPU id." >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    PYTHON_COMMAND=(python -u)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_COMMAND=(conda run --no-capture-output -n "$CONDA_ENV_NAME" python -u)
else
    echo "Conda environment '$CONDA_ENV_NAME' is not active and conda is unavailable." >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$(dirname "$MANIFEST")"
cd "$WORKSPACE_DIR"

COMMON_ARGS=(
    --exp_name "$EXP_NAME"
    --domain touchanything
    --split train
    --batch_size "$BATCH_SIZE"
    --batches_per_stratum "$BATCHES_PER_STRATUM"
    --aggregate_batches "$AGGREGATE_BATCHES"
    --num_workers "$NUM_WORKERS"
    --index_workers "$INDEX_WORKERS"
    --volume_workers "$VOLUME_WORKERS"
    --strata_manifest "$MANIFEST"
)

echo "[PREPARE] Building/reusing the shared pressure-strata manifest..."
"${PYTHON_COMMAND[@]}" hamer_tactile_ft/audit_gradient_strata.py \
    "${COMMON_ARGS[@]}" \
    --prepare_only
prepare_status=$?
if (( prepare_status != 0 )); then
    echo "Manifest preparation failed with exit $prepare_status." >&2
    exit "$prepare_status"
fi

JOBS=(rmse-best viou-best last)
PIDS=()
LABELS=()
NEXT_JOB=0
FAILURES=0

stop_children() {
    local pid
    echo
    echo "Stopping active stratified gradient audits..."
    for pid in "${PIDS[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${PIDS[@]}"; do
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_children INT TERM

launch_job() {
    local slot="$1"
    local ckpt="$2"
    local gpu="${GPU_IDS[$slot]}"
    local output_dir="$OUTPUT_ROOT/${ckpt}_touchanything"
    "${PYTHON_COMMAND[@]}" hamer_tactile_ft/audit_gradient_strata.py \
        "${COMMON_ARGS[@]}" \
        --ckpt "$ckpt" \
        --gpu "$gpu" \
        --output_dir "$output_dir" \
        --progress_position "$slot" &
    PIDS[$slot]=$!
    LABELS[$slot]="$ckpt"
    echo "[START] $ckpt on GPU $gpu -> $output_dir"
}

while (( NEXT_JOB < ${#JOBS[@]} || ${#PIDS[@]} > 0 )); do
    for slot in "${!GPU_IDS[@]}"; do
        if [[ -z "${PIDS[$slot]:-}" ]] && (( NEXT_JOB < ${#JOBS[@]} )); then
            launch_job "$slot" "${JOBS[$NEXT_JOB]}"
            ((NEXT_JOB += 1))
        fi
    done

    any_running=0
    for slot in "${!GPU_IDS[@]}"; do
        pid="${PIDS[$slot]:-}"
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            any_running=1
            continue
        fi
        if wait "$pid"; then
            echo "[OK] ${LABELS[$slot]}"
        else
            status=$?
            echo "[FAILED] ${LABELS[$slot]} (exit $status)" >&2
            ((FAILURES += 1))
        fi
        unset "PIDS[$slot]" "LABELS[$slot]"
    done

    if (( NEXT_JOB >= ${#JOBS[@]} && any_running == 0 )); then
        break
    fi
    sleep 1
done

trap - INT TERM
if (( FAILURES > 0 )); then
    echo "$FAILURES stratified audit job(s) failed." >&2
    exit 1
fi
echo "All stratified gradient audits completed: $OUTPUT_ROOT"
