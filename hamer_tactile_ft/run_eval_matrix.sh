#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Usage:
#   bash hamer_tactile_ft/run_eval_matrix.sh [EXP_NAME] [CKPT ...]
#
# Examples:
#   bash hamer_tactile_ft/run_eval_matrix.sh \
#     touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop16 \
#     loss-best contact-best

if (( $# == 0 )); then
    echo "Usage: $0 EXP_NAME [loss-best|contact-best|selector-best|last ...]" >&2
    exit 2
fi
EXP_NAME="$1"
shift

if (( $# > 0 )); then
    CKPT_SELECTORS=("$@")
else
    CKPT_SELECTORS=(loss-best last)
fi

# Each entry is DATASET:SPLIT. DATASET may itself be comma-separated.
# Override with semicolon-separated EVAL_TASKS_SPEC, for example:
#   EVAL_TASKS_SPEC='opentouch,touchanything:train'
if [[ -n "${EVAL_TASKS_SPEC:-}" ]]; then
    IFS=';' read -r -a EVAL_TASKS <<< "$EVAL_TASKS_SPEC"
else
    EVAL_TASKS=(
        "opentouch:test"
        "touchanything:test_seen"
        "touchanything:test_unseen"
    )
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INDEX_WORKERS="${INDEX_WORKERS:-128}"
NUM_WORKERS="${NUM_WORKERS:-32}"
DATA_BACKEND="${DATA_BACKEND:-auto}"
QUERY_MANIFESTS="${QUERY_MANIFESTS:-}"
HDF5_HANDLE_CACHE_SIZE="${HDF5_HANDLE_CACHE_SIZE:-4}"
HDF5_MANIFEST_CACHE_DIR="${HDF5_MANIFEST_CACHE_DIR:-$SCRIPT_DIR/hdf5_manifest_cache}"
DINO_WEIGHTS="${DINO_WEIGHTS:-}"
BBOX_MANIFESTS="${BBOX_MANIFESTS:-}"
SAVE_DIAGNOSTICS="${SAVE_DIAGNOSTICS:-1}"
RUN_SEQUENCE_AUDIT="${RUN_SEQUENCE_AUDIT:-0}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"
SELECTOR_CALIBRATION_FIT="${SELECTOR_CALIBRATION_FIT:-0}"
PROCESS_GRACE_SECONDS="${PROCESS_GRACE_SECONDS:-30}"
PROCESS_KILL_WAIT_SECONDS="${PROCESS_KILL_WAIT_SECONDS:-5}"

for flag_name in SAVE_DIAGNOSTICS RUN_SEQUENCE_AUDIT REBUILD_INDEX SELECTOR_CALIBRATION_FIT; do
    flag_value="${!flag_name}"
    if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
        echo "$flag_name must be 0 or 1 (got '$flag_value')." >&2
        exit 2
    fi
done
if [[ "$RUN_SEQUENCE_AUDIT" == "1" && "$SAVE_DIAGNOSTICS" != "1" ]]; then
    echo "RUN_SEQUENCE_AUDIT=1 requires SAVE_DIAGNOSTICS=1." >&2
    exit 2
fi

safe_name() {
    local value="${1,,}"
    value="${value//[^a-z0-9_-]/_}"
    printf '%s' "$value"
}

SAFE_EXP_NAME="$(safe_name "$EXP_NAME")"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/eval_reports_${SAFE_EXP_NAME}}"
SELECTOR_CALIBRATION_ROOT="${SELECTOR_CALIBRATION_ROOT:-$OUTPUT_ROOT}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

PIDS=()
LABELS=()
LOG_FILES=()
DIAGNOSTIC_DIRS=()

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
echo "Evaluation tasks: ${EVAL_TASKS[*]}"
echo "Save diagnostics: $SAVE_DIAGNOSTICS"
echo "Run sequence audit: $RUN_SEQUENCE_AUDIT"
echo "Rebuild index: $REBUILD_INDEX"
echo "Output root: $OUTPUT_ROOT"
echo "Selector calibration mode: $([[ "$SELECTOR_CALIBRATION_FIT" == "1" ]] && echo fit-validation || echo apply-if-available)"
echo "Selector calibration root: $SELECTOR_CALIBRATION_ROOT"
echo

for ckpt in "${CKPT_SELECTORS[@]}"; do
    case "$ckpt" in
        loss-best|contact-best|selector-best|last) ;;
        *)
            echo "Invalid checkpoint selector '$ckpt'. Use loss-best, contact-best, selector-best, or last." >&2
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
            --data_backend "$DATA_BACKEND"
            --hdf5_handle_cache_size "$HDF5_HANDLE_CACHE_SIZE"
            --hdf5_manifest_cache_dir "$HDF5_MANIFEST_CACHE_DIR"
            --index_workers "$INDEX_WORKERS"
            --exp_name "$EXP_NAME"
            --ckpt "$ckpt"
            --datasets "$dataset"
            --split "$split"
            --report_dir "$task_dir"
            --eval_output_root "$OUTPUT_ROOT"
        )
        calibration_path="$SELECTOR_CALIBRATION_ROOT/$ckpt/support_selector_calibration.json"
        if [[ "$SELECTOR_CALIBRATION_FIT" == "1" ]]; then
            if [[ "$split" != "val" && "$split" != "validation" ]]; then
                echo "SELECTOR_CALIBRATION_FIT=1 only accepts val/validation tasks (got '$task')." >&2
                exit 2
            fi
            mkdir -p "$(dirname "$calibration_path")"
            command+=(--selector_calibration_output "$calibration_path")
        elif [[ -f "$calibration_path" ]]; then
            command+=(--selector_calibration_input "$calibration_path")
        fi
        if [[ "$REBUILD_INDEX" == "1" ]]; then
            command+=(--rebuild_index)
        else
            command+=(--no-rebuild_index)
        fi
        if [[ "$SAVE_DIAGNOSTICS" == "1" ]]; then
            command+=(--save_diagnostics)
        fi
        if [[ -n "$DINO_WEIGHTS" ]]; then
            command+=(--dino_weights "$DINO_WEIGHTS")
        fi
        if [[ -n "$BBOX_MANIFESTS" ]]; then
            command+=(--bbox_manifests "$BBOX_MANIFESTS")
        fi
        if [[ -n "$QUERY_MANIFESTS" ]]; then
            command+=(--query_manifests "$QUERY_MANIFESTS")
        fi

        echo "Starting $label"
        echo "  log: $log_file"
        supervised_command=(
            "$PYTHON_BIN" hamer_tactile_ft/process_supervisor.py
            --registry-dir "$SCRIPT_DIR/run_processes"
            --grace-seconds "$PROCESS_GRACE_SECONDS"
            --kill-wait-seconds "$PROCESS_KILL_WAIT_SECONDS"
            --
            "${command[@]}"
        )
        "${supervised_command[@]}" >"$log_file" 2>&1 &
        PIDS+=("$!")
        LABELS+=("$label")
        LOG_FILES+=("$log_file")
        if [[ "$SAVE_DIAGNOSTICS" == "1" ]]; then
            DIAGNOSTIC_DIRS+=("$task_dir/eval_${dataset}_${split}_diagnostics")
        fi
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

if [[ "$RUN_SEQUENCE_AUDIT" == "1" ]]; then
    echo
    echo "Running sequence failure audits..."
    for index in "${!DIAGNOSTIC_DIRS[@]}"; do
        diagnostics_dir="${DIAGNOSTIC_DIRS[$index]}"
        label="${LABELS[$index]}"
        audit_log="$LOG_DIR/$(safe_name "sequence_audit_${label}").log"
        if "$PYTHON_BIN" hamer_tactile_ft/audit_sequence_failures.py \
            --diagnostics_dir "$diagnostics_dir" >"$audit_log" 2>&1; then
            echo "[AUDIT]  $label"
        else
            echo "[FAILED] sequence audit $label (log: $audit_log)" >&2
            ((failures += 1))
        fi
    done
fi

if (( failures > 0 )); then
    echo "$failures eval or audit process(es) failed. See logs under $LOG_DIR." >&2
    exit 1
fi

echo "All eval processes completed successfully."
echo "Results: $OUTPUT_ROOT"
