#!/usr/bin/env bash

set -euo pipefail

# Run from a private copy so source synchronization cannot replace this script
# while a long multi-GPU build is in progress.
if [[ "${TACTILE_INPUT_PRIORS_RUNTIME_OWNER_PID:-}" != "$$" ]]; then
    original_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/$(basename "${BASH_SOURCE[0]}")"
    runtime_copy="$(mktemp /tmp/tactile_input_priors_run.XXXXXX.sh)"
    cp -- "$original_script" "$runtime_copy"
    chmod 700 "$runtime_copy"
    export TACTILE_INPUT_PRIORS_RUNTIME_COPY="$runtime_copy"
    export TACTILE_INPUT_PRIORS_SCRIPT_DIR="$(dirname "$original_script")"
    export TACTILE_INPUT_PRIORS_RUNTIME_OWNER_PID="$$"
    exec /usr/bin/env bash "$runtime_copy" "$@"
fi

cleanup_runtime_copy() {
    if [[ -n "${TACTILE_INPUT_PRIORS_RUNTIME_COPY:-}" ]]; then
        rm -f -- "$TACTILE_INPUT_PRIORS_RUNTIME_COPY"
    fi
}
trap cleanup_runtime_copy EXIT

SCRIPT_DIR="$TACTILE_INPUT_PRIORS_SCRIPT_DIR"
MODE="${1:-}"
DEPTH_PYTHON="${DEPTH_PYTHON:-python}"
DEPTH_GPUS="${DEPTH_GPUS:-0,1,2,3,4,5,6,7}"
DEPTH_DATASET="${DEPTH_DATASET:-touchanything}"
DEPTH_SPLITS="${DEPTH_SPLITS:-auto}"
INPUT_PRIOR_ROOT="${INPUT_PRIOR_ROOT:-/home/ma-user/work/cfzhao/input_prior_full}"
DEPTH_BUILD_LOG_DIR="${DEPTH_BUILD_LOG_DIR:-$INPUT_PRIOR_ROOT/logs/depth_build}"

usage() {
    cat <<'EOF'
Usage: tactile_input_priors/run.sh MODE [options]

Offline MoGe depth-sidecar operations:
  depth-manifests          Discover or atomically rebuild query manifests
  depth-build              Build sidecars in one process
  depth-build-8gpu         Shard one explicit manifest over DEPTH_GPUS
  depth-build-auto-8gpu    Resolve requested splits, then build each over DEPTH_GPUS
  depth-validate           Validate sidecar coverage, hashes, and arrays

Common environment:
  DEPTH_PYTHON             Python with MoGe, NumPy, OpenCV, and h5py
  DEPTH_GPUS               Comma-separated builder GPUs (default: 0,...,7)
  DEPTH_DATASET            touchanything|opentouch (default: touchanything)
  DEPTH_SPLITS             Comma-separated splits or auto (default: auto)
  DEPTH_DATA_ROOT          Optional processed sequence-HDF5 root
  DEPTH_BUILD_LOG_DIR      Per-shard logs outside the source tree

Examples:
  ./tactile_input_priors/run.sh depth-manifests \
    --dataset touchanything --splits auto --create-missing --print-paths

  ./tactile_input_priors/run.sh depth-build \
    --manifest /path/to/touchanything_train.queries.jsonl \
    --model /path/to/moge/model.pt --output-dir /path/to/depth_sidecars

  DEPTH_GPUS=0,1,2,3,4,5,6,7 \
  ./tactile_input_priors/run.sh depth-build-auto-8gpu \
    --model /path/to/moge/model.pt --output-dir /path/to/depth_sidecars

  ./tactile_input_priors/run.sh depth-validate /path/to/depth_sidecars \
    --manifest /path/to/touchanything_train.queries.jsonl --deep
EOF
}

run_depth_build_shards() {
    IFS=',' read -r -a depth_gpus <<< "$DEPTH_GPUS"
    if (( ${#depth_gpus[@]} == 0 )); then
        echo "DEPTH_GPUS must contain at least one GPU." >&2
        return 2
    fi

    mkdir -p "$DEPTH_BUILD_LOG_DIR"
    local run_label
    run_label="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    local pids=()
    local labels=()
    local logs=()

    stop_depth_builders() {
        local pid
        for pid in "${pids[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        exit 130
    }
    trap stop_depth_builders INT TERM

    local shard_index gpu log_file
    for shard_index in "${!depth_gpus[@]}"; do
        gpu="${depth_gpus[$shard_index]}"
        log_file="$DEPTH_BUILD_LOG_DIR/${run_label}_shard_${shard_index}_gpu_${gpu}.log"
        echo "[depth-build] shard=$shard_index/${#depth_gpus[@]} gpu=$gpu log=$log_file"
        "$DEPTH_PYTHON" "$SCRIPT_DIR/build_depth_sidecars.py" \
            "$@" \
            --num-shards "${#depth_gpus[@]}" \
            --shard-index "$shard_index" \
            --device "cuda:$gpu" >"$log_file" 2>&1 &
        pids+=("$!")
        labels+=("shard_${shard_index}_gpu_${gpu}")
        logs+=("$log_file")
    done

    local failures=0 status index
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[depth-build] completed: ${labels[$index]}"
        else
            status=$?
            echo "[depth-build] failed: ${labels[$index]} (exit $status)" >&2
            tail -n 50 "${logs[$index]}" >&2 || true
            ((failures += 1))
        fi
    done
    trap - INT TERM
    if (( failures > 0 )); then
        echo "$failures depth sidecar shard(s) failed; completed sequence files remain resumable." >&2
        return 1
    fi
}

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
    usage
    exit 0
fi
shift

case "$MODE" in
    depth-manifests)
        "$DEPTH_PYTHON" "$SCRIPT_DIR/resolve_depth_manifests.py" "$@"
        ;;
    depth-build)
        "$DEPTH_PYTHON" "$SCRIPT_DIR/build_depth_sidecars.py" "$@"
        ;;
    depth-build-8gpu)
        run_depth_build_shards "$@"
        ;;
    depth-build-auto-8gpu)
        resolver_args=(
            --dataset "$DEPTH_DATASET"
            --splits "$DEPTH_SPLITS"
            --create-missing
            --print-paths
        )
        builder_root_args=()
        if [[ -n "${DEPTH_DATA_ROOT:-}" ]]; then
            resolver_args+=(--processed-root "$DEPTH_DATA_ROOT")
            builder_root_args+=(--data-root "$DEPTH_DATA_ROOT")
        fi
        if ! manifest_output="$(
            "$DEPTH_PYTHON" "$SCRIPT_DIR/resolve_depth_manifests.py" \
                "${resolver_args[@]}"
        )"; then
            echo "Depth manifest discovery/rebuild failed; no GPU builders were started." >&2
            exit 1
        fi
        mapfile -t manifests < <(printf '%s\n' "$manifest_output" | sed '/^[[:space:]]*$/d')
        if (( ${#manifests[@]} == 0 )); then
            echo "No query manifests were resolved." >&2
            exit 2
        fi
        for manifest in "${manifests[@]}"; do
            echo "[depth-build-auto] manifest=$manifest"
            run_depth_build_shards \
                --manifest "$manifest" \
                "${builder_root_args[@]}" \
                "$@"
        done
        ;;
    depth-validate)
        "$DEPTH_PYTHON" "$SCRIPT_DIR/depth_sidecar.py" validate "$@"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac
