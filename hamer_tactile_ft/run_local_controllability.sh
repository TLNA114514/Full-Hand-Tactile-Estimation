#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    cat <<'EOF'
Usage: run_local_controllability.sh MODE [audit options]

Modes:
  prepare          Build/reuse the persistent validation feature cache
  feature          Reproduce the original single-budget Feature oracle
  output           Reproduce the original Output/Support/Ordinal oracle
  stage01-feature  Sequence-balanced Feature RMS sweep
  stage01-output   Sequence-balanced Output cap/Exact/Support/Ordinal sweep
  stage01-all      Prepare and run both Stage 0.1 sweeps sequentially

The first prepare builds a persistent TouchAnything validation feature cache
with eight GPUs. Later oracle runs reuse it without JPEG, HDF5, or DINO reads.

Optional environment:
  TACTILE_BASE_CHECKPOINT  FullGrid32 loss-best compact checkpoint
  DINO_WEIGHTS             DINOv3 H+/16 weights used by cache preparation
  TACTILE_FEATURE_CACHE    Persistent cache root
  CACHE_GPUS               Cache builder GPUs (default: 0,...,7)
  CACHE_BATCH_SIZE         Batch size per cache GPU (default: 128)
  LOCAL_CONTROL_DEVICE     Oracle device (default: cuda:0)
  LOCAL_CONTROL_REPORT_ROOT
EOF
    exit 2
fi
shift

DEFAULT_CHECKPOINT="$ROOT_DIR/hamer_tactile_ft/checkpoints/touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12/best_loss.ckpt"
DEFAULT_DINO="/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
DEFAULT_RUNTIME_ROOT="/home/ma-user/work/cfzhao/input_prior_full"

TACTILE_BASE_CHECKPOINT="${TACTILE_BASE_CHECKPOINT:-$DEFAULT_CHECKPOINT}"
DINO_WEIGHTS="${DINO_WEIGHTS:-$DEFAULT_DINO}"
TACTILE_FEATURE_CACHE="${TACTILE_FEATURE_CACHE:-$DEFAULT_RUNTIME_ROOT/cache/local_control/ta_val_crop12}"
CACHE_GPUS="${CACHE_GPUS:-0,1,2,3,4,5,6,7}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
LOCAL_CONTROL_DEVICE="${LOCAL_CONTROL_DEVICE:-cuda:0}"
LOCAL_CONTROL_REPORT_ROOT="${LOCAL_CONTROL_REPORT_ROOT:-$ROOT_DIR/hamer_tactile_ft/local_control_audits}"
TACTILE_PYTHON="${TACTILE_PYTHON:-/home/ma-user/work/cfzhao/tactile/bin/python}"

if [[ ! -x "$TACTILE_PYTHON" ]]; then
    TACTILE_PYTHON=python
fi

cache_ready() {
    if [[ -f "$TACTILE_FEATURE_CACHE/CACHE_DONE.json" ]]; then
        return 0
    fi
    local parts=() path
    shopt -s nullglob
    parts=("$TACTILE_FEATURE_CACHE"/part-*-of-*)
    shopt -u nullglob
    (( ${#parts[@]} > 0 )) || return 1
    local first_name expected_text expected
    first_name="$(basename "${parts[0]}")"
    expected_text="${first_name##*-of-}"
    [[ "$expected_text" =~ ^[0-9]+$ ]] || return 1
    expected=$((10#$expected_text))
    (( ${#parts[@]} == expected )) || return 1
    for path in "${parts[@]}"; do
        [[ -f "$path/CACHE_DONE.json" ]] || return 1
    done
}

prepare_cache() {
    if cache_ready; then
        echo "[local-control] Reusing finalized cache: $TACTILE_FEATURE_CACHE"
        return 0
    fi
    [[ -f "$TACTILE_BASE_CHECKPOINT" ]] || {
        echo "Missing base checkpoint: $TACTILE_BASE_CHECKPOINT" >&2
        exit 2
    }
    [[ -f "$DINO_WEIGHTS" ]] || {
        echo "Missing DINO weights: $DINO_WEIGHTS" >&2
        exit 2
    }
    mkdir -p "$TACTILE_FEATURE_CACHE"
    local lock_file="$TACTILE_FEATURE_CACHE/.prepare.lock"
    exec 9>"$lock_file"
    echo "[local-control] Waiting for shared cache preparation lock: $lock_file"
    flock 9
    if cache_ready; then
        echo "[local-control] Cache was completed by another process. Reusing it."
        flock -u 9
        return 0
    fi
    echo "[local-control] Building persistent TouchAnything val cache on GPUs $CACHE_GPUS"
    CACHE_GPUS="$CACHE_GPUS" \
    CACHE_BATCH_SIZE="$CACHE_BATCH_SIZE" \
    TACTILE_BASE_CHECKPOINT="$TACTILE_BASE_CHECKPOINT" \
    DINO_WEIGHTS="$DINO_WEIGHTS" \
    TACTILE_PYTHON="$TACTILE_PYTHON" \
    "$ROOT_DIR/tactile_input_priors/run.sh" cache-tactile-8gpu \
        --cache-dir "$TACTILE_FEATURE_CACHE" \
        --datasets touchanything \
        --split val \
        --fields z_rgb,tactile_signal,has_tactile \
        --bbox-rescale-factor 1.2 \
        --input-resolution 256x192 \
        --batch-size "$CACHE_BATCH_SIZE"
    cache_ready || {
        echo "Cache builders returned without a complete cache: $TACTILE_FEATURE_CACHE" >&2
        exit 1
    }
    flock -u 9
    echo "[local-control] Cache ready: $TACTILE_FEATURE_CACHE"
}

run_oracle() {
    local name="$1" oracles="$2"
    shift 2
    cache_ready || {
        echo "Feature cache is not ready. Run this first:" >&2
        echo "  $0 prepare" >&2
        exit 2
    }
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_local_controllability.py" \
        --feature-cache "$TACTILE_FEATURE_CACHE" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --oracles "$oracles" \
        --device "$LOCAL_CONTROL_DEVICE" \
        --output-dir "$LOCAL_CONTROL_REPORT_ROOT/$name" \
        "$@"
}

case "$MODE" in
    prepare)
        prepare_cache
        ;;
    feature)
        run_oracle feature_oracle feature "$@"
        ;;
    output)
        run_oracle output_support_ordinal output,support,ordinal \
            --support-positive-threshold 0.05 \
            "$@"
        ;;
    stage01-feature)
        run_oracle stage01_feature_sweep feature \
            --feature-rms-budgets 0.025,0.05,0.10 \
            --samples-per-error-stratum 16 \
            --max-samples-per-sequence 4 \
            "$@"
        ;;
    stage01-output)
        run_oracle stage01_output_sweep output,output_exact,support,ordinal \
            --output-logit-delta-max-values 1,2,4,6 \
            --support-positive-threshold 0.10 \
            --samples-per-error-stratum 16 \
            --max-samples-per-sequence 4 \
            "$@"
        ;;
    stage01-all)
        prepare_cache
        run_oracle stage01_feature_sweep feature \
            --feature-rms-budgets 0.025,0.05,0.10 \
            --samples-per-error-stratum 16 \
            --max-samples-per-sequence 4 \
            "$@"
        run_oracle stage01_output_sweep output,output_exact,support,ordinal \
            --output-logit-delta-max-values 1,2,4,6 \
            --support-positive-threshold 0.10 \
            --samples-per-error-stratum 16 \
            --max-samples-per-sequence 4 \
            "$@"
        ;;
    all)
        prepare_cache
        run_oracle feature_oracle feature "$@"
        run_oracle output_support_ordinal output,support,ordinal "$@"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac
