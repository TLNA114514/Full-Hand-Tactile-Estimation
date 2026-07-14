#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tactile}"
CAPTURE_GPUS="${CAPTURE_GPUS:-0,1,2,3,4,5,6,7}"
REPLAY_GPUS="${REPLAY_GPUS:-0,1,2}"
MAX_STEPS="${MAX_STEPS:-2500}"
SEED="${SEED:-2029}"
V19_EXP_NAME="${V19_EXP_NAME:-mixed_zero_ordinal_residual_v19_condnll}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints/$V19_EXP_NAME/best_rmse.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/amp_audits/v19_rmse_best}"
CAPTURE_DIR="$OUTPUT_ROOT/capture"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    PYTHON_COMMAND=(python -u)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_COMMAND=(conda run --no-capture-output -n "$CONDA_ENV_NAME" python -u)
else
    echo "Conda environment '$CONDA_ENV_NAME' is not active and conda is unavailable." >&2
    exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "RMSE-best checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

mkdir -p "$CAPTURE_DIR"
cd "$WORKSPACE_DIR"

if [[ "${SKIP_CAPTURE:-0}" != "1" ]]; then
    if compgen -G "$CAPTURE_DIR/nonfinite_grad_rank*.jsonl" >/dev/null || \
       compgen -G "$CAPTURE_DIR/capture_summary_rank*.json" >/dev/null; then
        echo "Capture output already exists under $CAPTURE_DIR." >&2
        echo "Use a new OUTPUT_ROOT or set SKIP_CAPTURE=1 to replay the existing capture." >&2
        exit 2
    fi
    echo "[CAPTURE] fp16 AMP audit on GPUs $CAPTURE_GPUS for $MAX_STEPS steps"
    "${PYTHON_COMMAND[@]}" hamer_tactile_ft/train.py \
        --checkpoint "$CHECKPOINT" \
        --gpus "$CAPTURE_GPUS" \
        --lr 2.5e-5 \
        --batch_size 32 \
        --epochs 90 \
        --max_steps "$MAX_STEPS" \
        --datasets opentouch,touchanything \
        --exp_name amp_audit_v19_rmse_best_capture \
        --index_cache_dir "$SCRIPT_DIR/index_cache" \
        --index_workers 256 \
        --index_chunksize 512 \
        --num_workers 32 \
        --persistent_workers \
        --prefetch_factor 4 \
        --tactile_only_forward \
        --tactile_head_type zero_ordinal_residual \
        --pool_layout hand7 \
        --pool_grid_size 7 \
        --ordinal_thresholds 0.005,0.02,0.05,0.1,0.2,0.4,0.7 \
        --zero_support_thr 0.005 \
        --support_loss_weight 0.01 \
        --positive_bin_loss_weight 0.005 \
        --positive_bin_values_path "$SCRIPT_DIR/ordinal_bin_values_v17_train.json" \
        --gradient_clip_val 1.0 \
        --tactile_loss_scale 10.0 \
        --active_pressure_thr 0.05 \
        --active_pressure_weight 2.25 \
        --active_pressure_gamma 1.0 \
        --active_pressure_max_weight 3.25 \
        --opentouch_high_pressure_thr 0.9 \
        --opentouch_high_pressure_weight 0.3 \
        --trainer_precision 16-mixed \
        --seed "$SEED" \
        --skip_validation \
        --skip_checkpointing \
        --audit_nonfinite_grads \
        --nonfinite_audit_dir "$CAPTURE_DIR"
    capture_status=$?
    if (( capture_status != 0 )); then
        echo "AMP capture failed with exit $capture_status." >&2
        exit "$capture_status"
    fi
else
    echo "[CAPTURE] skipped; reusing $CAPTURE_DIR"
fi

IFS=',' read -r -a GPU_IDS <<< "$REPLAY_GPUS"
if (( ${#GPU_IDS[@]} < 1 )); then
    echo "REPLAY_GPUS must contain at least one GPU id." >&2
    exit 2
fi

PRECISIONS=(fp16 bf16 fp32)
PIDS=()
LABELS=()
FAILURES=0

stop_children() {
    local pid
    echo
    echo "Stopping replay processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_children INT TERM

for index in "${!PRECISIONS[@]}"; do
    precision="${PRECISIONS[$index]}"
    gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    output_dir="$OUTPUT_ROOT/replay_${precision}"
    echo "[REPLAY] $precision on GPU $gpu -> $output_dir"
    "${PYTHON_COMMAND[@]}" hamer_tactile_ft/replay_nonfinite_batches.py \
        --capture "$CAPTURE_DIR" \
        --checkpoint "$CHECKPOINT" \
        --precision "$precision" \
        --gpu "$gpu" \
        --seed "$SEED" \
        --output_dir "$output_dir" &
    PIDS+=("$!")
    LABELS+=("$precision")
done

for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "[OK] ${LABELS[$index]} replay"
    else
        status=$?
        echo "[FAILED] ${LABELS[$index]} replay (exit $status)" >&2
        ((FAILURES += 1))
    fi
done
trap - INT TERM

if (( FAILURES > 0 )); then
    exit 1
fi
echo "AMP audit completed: $OUTPUT_ROOT"
