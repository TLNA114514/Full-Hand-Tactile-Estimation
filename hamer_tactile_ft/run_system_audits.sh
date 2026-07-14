#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

EXP_NAME="${EXP_NAME:-mixed_zero_ordinal_residual_v19_condnll}"
GPUS_CSV="${GPUS:-0,1,2,3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tactile}"
NUM_BATCHES="${NUM_BATCHES:-32}"
GRAD_BATCH_SIZE="${GRAD_BATCH_SIZE:-4}"
SAMPLES_PER_DOMAIN="${SAMPLES_PER_DOMAIN:-2000}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
INDEX_WORKERS="${INDEX_WORKERS:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/eval_reports_${EXP_NAME}/gradient_audits}"

IFS=',' read -r -a GPU_IDS <<< "$GPUS_CSV"
if (( ${#GPU_IDS[@]} == 0 )); then
    echo "GPUS must contain at least one GPU id." >&2
    exit 2
fi

for index in "${!GPU_IDS[@]}"; do
    GPU_IDS[$index]="${GPU_IDS[$index]//[[:space:]]/}"
    if [[ -z "${GPU_IDS[$index]}" ]]; then
        echo "GPUS contains an empty GPU id: '$GPUS_CSV'" >&2
        exit 2
    fi
done

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    PYTHON_COMMAND=(python -u)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_COMMAND=(conda run --no-capture-output -n "$CONDA_ENV_NAME" python -u)
else
    echo "Conda environment '$CONDA_ENV_NAME' is not active and conda is unavailable." >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
cd "$WORKSPACE_DIR"

PIDS=()
LABELS=()
GPU_FOR_PID=()
NEXT_JOB=0
FAILURES=0

JOBS=(
    "gradient:rmse-best"
    "gradient:viou-best"
    "gradient:last"
    "feature:last"
)

stop_children() {
    local pid
    echo
    echo "Stopping active audit processes..."
    for pid in "${PIDS[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${PIDS[@]}"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    exit 130
}
trap stop_children INT TERM

launch_job() {
    local slot="$1"
    local job_index="$2"
    local gpu="${GPU_IDS[$slot]}"
    local job="${JOBS[$job_index]}"
    local kind="${job%%:*}"
    local ckpt="${job#*:}"
    local label
    local output_dir

    if [[ "$kind" == "gradient" ]]; then
        label="1b/$ckpt"
        output_dir="$OUTPUT_ROOT/1b_${ckpt}_touchanything"
        "${PYTHON_COMMAND[@]}" hamer_tactile_ft/audit_gradient_subdomains.py \
            --exp_name "$EXP_NAME" \
            --ckpt "$ckpt" \
            --domain touchanything \
            --split train \
            --gpu "$gpu" \
            --batch_size "$GRAD_BATCH_SIZE" \
            --num_batches "$NUM_BATCHES" \
            --num_workers "$NUM_WORKERS" \
            --index_workers "$INDEX_WORKERS" \
            --output_dir "$output_dir" \
            --progress_position "$slot" &
    else
        label="feature-probe/$ckpt"
        output_dir="$OUTPUT_ROOT/feature_probe_${ckpt}"
        "${PYTHON_COMMAND[@]}" hamer_tactile_ft/audit_hamer_feature_probes.py \
            --exp_name "$EXP_NAME" \
            --ckpt "$ckpt" \
            --gpu "$gpu" \
            --split train \
            --samples_per_domain "$SAMPLES_PER_DOMAIN" \
            --feature_batch_size "$FEATURE_BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --index_workers "$INDEX_WORKERS" \
            --output_dir "$output_dir" \
            --progress_position "$slot" &
    fi

    PIDS[$slot]=$!
    LABELS[$slot]="$label"
    GPU_FOR_PID[$slot]="$gpu"
    echo "[START] $label on GPU $gpu -> $output_dir"
}

echo "Experiment: $EXP_NAME"
echo "GPUs: ${GPU_IDS[*]}"
echo "Jobs: ${JOBS[*]}"
echo "Output root: $OUTPUT_ROOT"
echo

while (( NEXT_JOB < ${#JOBS[@]} || ${#PIDS[@]} > 0 )); do
    for slot in "${!GPU_IDS[@]}"; do
        if [[ -z "${PIDS[$slot]:-}" ]] && (( NEXT_JOB < ${#JOBS[@]} )); then
            launch_job "$slot" "$NEXT_JOB"
            ((NEXT_JOB += 1))
        fi
    done

    any_running=0
    for slot in "${!GPU_IDS[@]}"; do
        pid="${PIDS[$slot]:-}"
        if [[ -z "$pid" ]]; then
            continue
        fi
        if kill -0 "$pid" 2>/dev/null; then
            any_running=1
            continue
        fi
        if wait "$pid"; then
            echo "[OK] ${LABELS[$slot]} on GPU ${GPU_FOR_PID[$slot]}"
        else
            status=$?
            echo "[FAILED] ${LABELS[$slot]} on GPU ${GPU_FOR_PID[$slot]} (exit $status)" >&2
            ((FAILURES += 1))
        fi
        unset "PIDS[$slot]" "LABELS[$slot]" "GPU_FOR_PID[$slot]"
    done

    if (( NEXT_JOB >= ${#JOBS[@]} && any_running == 0 )); then
        break
    fi
    sleep 1
done

trap - INT TERM

if (( FAILURES > 0 )); then
    echo "$FAILURES audit job(s) failed." >&2
    exit 1
fi

echo "All audit jobs completed successfully."
echo "Results: $OUTPUT_ROOT"
