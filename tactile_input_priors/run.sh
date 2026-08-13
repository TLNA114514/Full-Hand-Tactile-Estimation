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
    release_pipeline_lock || true
    if [[ -n "${TACTILE_INPUT_PRIORS_RUNTIME_COPY:-}" ]]; then
        rm -f -- "$TACTILE_INPUT_PRIORS_RUNTIME_COPY"
    fi
}
trap cleanup_runtime_copy EXIT

SCRIPT_DIR="$TACTILE_INPUT_PRIORS_SCRIPT_DIR"
MODE="${1:-}"
DEPTH_PYTHON="${DEPTH_PYTHON:-python}"
TACTILE_PYTHON="${TACTILE_PYTHON:-python}"
VLM_PYTHON="${VLM_PYTHON:-python}"
DEPTH_GPUS="${DEPTH_GPUS:-0,1,2,3,4,5,6,7}"
DEPTH_DATASET="${DEPTH_DATASET:-touchanything}"
DEPTH_SPLITS="${DEPTH_SPLITS:-auto}"
INPUT_PRIOR_ROOT="${INPUT_PRIOR_ROOT:-/home/ma-user/work/cfzhao/input_prior_full}"
DEPTH_BUILD_LOG_DIR="${DEPTH_BUILD_LOG_DIR:-$INPUT_PRIOR_ROOT/logs/depth_build}"
PIPELINE_LOCK_PATH=""
PIPELINE_LOCK_TOKEN=""

prefer_python() {
    local current="$1" preferred="$2"
    if [[ "$current" == "python" && -x "$preferred" ]]; then
        printf '%s\n' "$preferred"
    else
        printf '%s\n' "$current"
    fi
}

TACTILE_PYTHON="$(prefer_python "$TACTILE_PYTHON" /home/ma-user/work/cfzhao/tactile/bin/python)"
DEPTH_PYTHON="$(prefer_python "$DEPTH_PYTHON" /home/ma-user/work/cfzhao/input_prior_step0/envs/depth/bin/python)"
VLM_PYTHON="$(prefer_python "$VLM_PYTHON" /home/ma-user/work/cfzhao/input_prior_step0/envs/vlm/bin/python)"

usage() {
    cat <<'EOF'
Usage: tactile_input_priors/run.sh MODE [options]

Offline MoGe depth-sidecar operations:
  depth-manifests          Discover or atomically rebuild query manifests
  depth-build              Build sidecars in one process
  depth-build-8gpu         Shard one explicit manifest over DEPTH_GPUS
  depth-build-auto-8gpu    Resolve requested splits, then build each over DEPTH_GPUS
  depth-validate           Validate sidecar coverage, hashes, and arrays

Frozen feature cache:
  cache-build              Build/resume a generic mmap feature cache
  cache-verify             Verify a finalized feature cache
  cache-inspect            Inspect one cached sample
  cache-self-check         Run the atomic resume/read tiny check
  cache-tactile            Cache frozen z/h/logits on one GPU
  cache-tactile-8gpu       Build eight disjoint cache partitions
  cache-vlm                Cache full-frame Qwen embeddings on one GPU
  cache-vlm-8gpu           Build one VLM split over eight GPUs
  cache-vlm-auto-8gpu      Build train/val/test VLM caches over eight GPUs

Feature-level prior adapters (all implementation stays in this directory):
  train-depth-real         Aligned Depth spatial rectification
  train-depth-shuffle      Spatially shuffled Depth control
  train-vlm-real           VLM low-rank bottleneck modulation
  train-vlm-control        Context-shuffled VLM control
  train-prior              Generic adapter training entry
  eval-prior               Generic frozen-base/fused evaluation entry
  eval-prior-8gpu          DDP evaluation with exact non-padded sharding
  prior-process-list       List supervised prior-training process groups
  prior-process-stop       Stop every supervised prior-training group on this host
  prior-debug-monitor      Attach I/O diagnostics to the newest active training group

End-to-end pipelines:
  prepare-online           Prepare sidecars/VLM cache only
  train-online             Prepare missing inputs, then train only
  eval-online              Evaluate existing online checkpoints only
  prepare-cache-only       Prepare all deterministic feature caches only
  train-cache-only         Prepare missing caches, then train only
  eval-cache-only          Evaluate existing cache-only checkpoints only
  pipeline-online          Prepare, train, and evaluate one online experiment
  pipeline-cache-only      Prepare, train, and evaluate one cache-only experiment

Pipeline experiments:
  depth-real | depth-shuffle | vlm-real | vlm-control

Common environment:
  DEPTH_PYTHON             Python with MoGe, NumPy, OpenCV, and h5py
  DEPTH_GPUS               Comma-separated builder GPUs (default: 0,...,7)
  DEPTH_DATASET            touchanything|opentouch (default: touchanything)
  DEPTH_SPLITS             Comma-separated splits or auto (default: auto)
  DEPTH_DATA_ROOT          Optional processed sequence-HDF5 root
  DEPTH_BUILD_LOG_DIR      Per-shard logs outside the source tree
  TACTILE_PYTHON           Existing tactile environment Python
  TACTILE_BASE_CHECKPOINT  Compact FullGrid loss-best base checkpoint
  DINO_WEIGHTS             Local DINOv3 H+/16 weights
  DEPTH_SIDECAR_ROOT       Versioned MoGe sidecar root
  BASE_FEATURE_CACHE       Optional frozen z/h/logit cache
  VLM_FEATURE_CACHE        Cache containing vlm_embedding
  VLM_PRIOR_DIM            VLM embedding dimension (Qwen3-VL-2B: 2048)
  VLM_MODEL                Local Qwen3-VL-Embedding model directory
  QWEN_EMBED_CODE_ROOT     Official Qwen3-VL-Embedding checkout
  VLM_CACHE_ROOT           Parent of split-specific VLM caches

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

  TACTILE_BASE_CHECKPOINT=/path/to/loss-best.ckpt \
  DEPTH_SIDECAR_ROOT=/path/to/depth_sidecars \
  ./tactile_input_priors/run.sh train-depth-real --gpus 0,1,2,3,4,5,6,7

  TACTILE_BASE_CHECKPOINT=/path/to/loss-best.ckpt \
  ./tactile_input_priors/run.sh pipeline-online depth-real
EOF
}

release_pipeline_lock() {
    if [[ -z "${PIPELINE_LOCK_PATH:-}" || -z "${PIPELINE_LOCK_TOKEN:-}" ]]; then
        return 0
    fi
    local recorded=""
    if [[ -f "$PIPELINE_LOCK_PATH/owner" ]]; then
        recorded="$(sed -n '1p' "$PIPELINE_LOCK_PATH/owner" 2>/dev/null || true)"
    fi
    if [[ "$recorded" == *" $PIPELINE_LOCK_TOKEN" ]]; then
        rm -f -- "$PIPELINE_LOCK_PATH/owner"
        rmdir -- "$PIPELINE_LOCK_PATH" 2>/dev/null || true
    fi
    PIPELINE_LOCK_PATH=""
    PIPELINE_LOCK_TOKEN=""
}

acquire_pipeline_lock() {
    local path="$1"
    local timeout="${PIPELINE_LOCK_TIMEOUT_SECONDS:-86400}"
    local started now owner_host owner_pid owner_token
    started="$(date +%s)"
    mkdir -p -- "$(dirname "$path")"
    while ! mkdir -- "$path" 2>/dev/null; do
        owner_host=""
        owner_pid=""
        owner_token=""
        if [[ -f "$path/owner" ]]; then
            read -r owner_host owner_pid owner_token < "$path/owner" || true
        fi
        if [[ "$owner_host" == "$(hostname)" && "$owner_pid" =~ ^[0-9]+$ ]] && \
           ! kill -0 "$owner_pid" 2>/dev/null; then
            echo "[pipeline] reclaiming stale same-host preparation lock: $path"
            rm -f -- "$path/owner"
            rmdir -- "$path" 2>/dev/null || true
            continue
        fi
        now="$(date +%s)"
        if (( now - started >= timeout )); then
            echo "Timed out waiting for preparation lock $path; owner=$owner_host/$owner_pid" >&2
            return 1
        fi
        echo "[pipeline] waiting for shared preparation lock: $path"
        sleep 10
    done
    PIPELINE_LOCK_PATH="$path"
    PIPELINE_LOCK_TOKEN="$(hostname)-$$-$(date +%s%N)"
    printf '%s %s %s\n' "$(hostname)" "$$" "$PIPELINE_LOCK_TOKEN" > "$path/owner"
}

partition_cache_complete() {
    local root="$1"
    if [[ -f "$root/CACHE_DONE.json" ]]; then
        return 0
    fi
    local expected gpu_array part
    IFS=',' read -r -a gpu_array <<< "${CACHE_GPUS:-0,1,2,3,4,5,6,7}"
    expected="${#gpu_array[@]}"
    for ((part = 0; part < expected; part++)); do
        if [[ ! -f "$root/part-$(printf '%02d' "$part")-of-$(printf '%02d' "$expected")/CACHE_DONE.json" ]]; then
            return 1
        fi
    done
}

require_environment() {
    local name value
    for name in "$@"; do
        value="${!name:-}"
        if [[ -z "$value" ]]; then
            echo "$name is required for mode=$MODE" >&2
            return 2
        fi
    done
}

run_prior_train() {
    local adapter_type="$1" prior_dim="$2" control="$3" exp_name="$4"
    shift 4
    require_environment TACTILE_BASE_CHECKPOINT DINO_WEIGHTS
    local cache_args=()
    if [[ -n "${BASE_FEATURE_CACHE_TRAIN:-}" || -n "${BASE_FEATURE_CACHE_VAL:-}" ]]; then
        require_environment BASE_FEATURE_CACHE_TRAIN BASE_FEATURE_CACHE_VAL
        cache_args+=(
            --train-base-feature-cache "$BASE_FEATURE_CACHE_TRAIN"
            --val-base-feature-cache "$BASE_FEATURE_CACHE_VAL"
            --no-train-augmentation
            --cache-only
        )
    elif [[ -n "${BASE_FEATURE_CACHE:-}" ]]; then
        cache_args+=(--base-feature-cache "$BASE_FEATURE_CACHE")
        cache_args+=(--no-train-augmentation --cache-only)
    fi
    if [[ "$adapter_type" == "depth_spatial" ]]; then
        if [[ -n "${DEPTH_FEATURE_CACHE_TRAIN:-}" || -n "${DEPTH_FEATURE_CACHE_VAL:-}" ]]; then
            require_environment DEPTH_FEATURE_CACHE_TRAIN DEPTH_FEATURE_CACHE_VAL
            cache_args+=(
                --train-prior-feature-cache "$DEPTH_FEATURE_CACHE_TRAIN"
                --val-prior-feature-cache "$DEPTH_FEATURE_CACHE_VAL"
            )
        elif [[ -n "${DEPTH_FEATURE_CACHE:-}" ]]; then
            cache_args+=(--prior-feature-cache "$DEPTH_FEATURE_CACHE")
        elif [[ -z "${BASE_FEATURE_CACHE:-}${BASE_FEATURE_CACHE_TRAIN:-}${BASE_FEATURE_CACHE_VAL:-}" ]]; then
            require_environment DEPTH_SIDECAR_ROOT
            cache_args+=(--depth-sidecar-root "$DEPTH_SIDECAR_ROOT")
        fi
    else
        if [[ -n "${VLM_FEATURE_CACHE_TRAIN:-}" || -n "${VLM_FEATURE_CACHE_VAL:-}" ]]; then
            require_environment VLM_FEATURE_CACHE_TRAIN VLM_FEATURE_CACHE_VAL
            cache_args+=(
                --train-prior-feature-cache "$VLM_FEATURE_CACHE_TRAIN"
                --val-prior-feature-cache "$VLM_FEATURE_CACHE_VAL"
            )
        else
            require_environment VLM_FEATURE_CACHE
            cache_args+=(--prior-feature-cache "$VLM_FEATURE_CACHE")
        fi
    fi
    local process_supervisor
    process_supervisor="$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py"
    export OMP_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export OPENBLAS_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export MKL_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export NUMEXPR_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
    "$TACTILE_PYTHON" "$process_supervisor" \
        --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
        --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
        --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
        -- "$TACTILE_PYTHON" "$SCRIPT_DIR/train_prior_adapter.py" \
        --adapter-type "$adapter_type" \
        --prior-dim "$prior_dim" \
        --prior-control "$control" \
        --exp-name "$exp_name" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --dino-weights "$DINO_WEIGHTS" \
        "${cache_args[@]}" \
        "$@"
}

run_tactile_cache_partitions() {
    require_environment TACTILE_BASE_CHECKPOINT DINO_WEIGHTS
    local cache_gpus="${CACHE_GPUS:-0,1,2,3,4,5,6,7}"
    IFS=',' read -r -a gpu_array <<< "$cache_gpus"
    local count="${#gpu_array[@]}"
    if (( count == 0 )); then
        echo "CACHE_GPUS must contain at least one GPU." >&2
        return 2
    fi
    local log_root="${CACHE_LOG_DIR:-$INPUT_PRIOR_ROOT/logs/feature_cache}"
    mkdir -p "$log_root"
    local pids=() logs=() index gpu log
    stop_cache_builders() {
        local pid
        for pid in "${pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        exit 130
    }
    trap stop_cache_builders INT TERM
    for index in "${!gpu_array[@]}"; do
        gpu="${gpu_array[$index]}"
        log="$log_root/cache_part_${index}_gpu_${gpu}.log"
        echo "[cache-tactile] partition=$index/$count gpu=$gpu log=$log"
        "$TACTILE_PYTHON" "$SCRIPT_DIR/cache_tactile_features.py" \
            --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
            --dino-weights "$DINO_WEIGHTS" \
            --device "cuda:$gpu" \
            --num-partitions "$count" \
            --partition-index "$index" \
            "$@" >"$log" 2>&1 &
        pids+=("$!")
        logs+=("$log")
    done
    local failures=0 status
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[cache-tactile] completed partition=$index"
        else
            status=$?
            echo "[cache-tactile] failed partition=$index exit=$status" >&2
            tail -n 80 "${logs[$index]}" >&2 || true
            ((failures += 1))
        fi
    done
    trap - INT TERM
    (( failures == 0 ))
}

run_vlm_cache_partitions() {
    require_environment VLM_MODEL QWEN_EMBED_CODE_ROOT
    local cache_gpus="${CACHE_GPUS:-0,1,2,3,4,5,6,7}"
    IFS=',' read -r -a gpu_array <<< "$cache_gpus"
    local count="${#gpu_array[@]}"
    local log_root="${CACHE_LOG_DIR:-$INPUT_PRIOR_ROOT/logs/vlm_cache}"
    mkdir -p "$log_root"
    local pids=() logs=() index gpu log
    stop_vlm_cache_builders() {
        local pid
        for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
        exit 130
    }
    trap stop_vlm_cache_builders INT TERM
    for index in "${!gpu_array[@]}"; do
        gpu="${gpu_array[$index]}"
        log="$log_root/vlm_part_${index}_gpu_${gpu}.log"
        echo "[cache-vlm] partition=$index/$count gpu=$gpu log=$log"
        CUDA_VISIBLE_DEVICES="$gpu" "$VLM_PYTHON" "$SCRIPT_DIR/cache_vlm_embeddings.py" \
            --model "$VLM_MODEL" \
            --qwen-code-root "$QWEN_EMBED_CODE_ROOT" \
            --num-partitions "$count" \
            --partition-index "$index" \
            "$@" >"$log" 2>&1 &
        pids+=("$!")
        logs+=("$log")
    done
    local failures=0 status
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[cache-vlm] completed partition=$index"
        else
            status=$?
            echo "[cache-vlm] failed partition=$index exit=$status" >&2
            tail -n 80 "${logs[$index]}" >&2 || true
            ((failures += 1))
        fi
    done
    trap - INT TERM
    (( failures == 0 ))
}

resolve_pipeline_manifests() {
    local dataset="$1" splits="$2"
    local args=(
        --dataset "$dataset"
        --splits "$splits"
        --create-missing
        --print-paths
    )
    if [[ -n "${DEPTH_DATA_ROOT:-}" ]]; then
        args+=(--processed-root "$DEPTH_DATA_ROOT")
    fi
    "$DEPTH_PYTHON" "$SCRIPT_DIR/resolve_depth_manifests.py" "${args[@]}"
}

depth_validation_stamp_path() {
    local manifest="$1" key
    key="$(printf '%s' "$manifest" | sha256sum | cut -d' ' -f1)"
    printf '%s/.validation_cache/%s.stamp\n' "$DEPTH_SIDECAR_ROOT" "$key"
}

depth_validation_fingerprint() {
    local manifest="$1" summary_path manifest_stat summary_stat summary_hash validator_hash
    summary_path="${manifest%.queries.jsonl}.summary.json"
    manifest_stat="$(stat -Lc '%s:%Y' "$manifest")"
    if [[ -f "$summary_path" ]]; then
        summary_stat="$(stat -Lc '%s:%Y' "$summary_path")"
        summary_hash="$(sha256sum "$summary_path" | cut -d' ' -f1)"
    else
        summary_stat="missing"
        summary_hash="missing"
    fi
    validator_hash="$(sha256sum "$SCRIPT_DIR/depth_sidecar.py" | cut -d' ' -f1)"
    printf '%s\n' \
        'schema=tactile_depth_validation_stamp_v1' \
        "manifest=$manifest" \
        "manifest_stat=$manifest_stat" \
        "summary_stat=$summary_stat" \
        "summary_sha256=$summary_hash" \
        "sidecar_root=$(realpath -m "$DEPTH_SIDECAR_ROOT")" \
        "validator_sha256=$validator_hash"
}

depth_validation_cached() {
    local manifest="$1" stamp
    [[ "${FORCE_DEPTH_VALIDATE:-0}" != "1" ]] || return 1
    stamp="$(depth_validation_stamp_path "$manifest")"
    [[ -f "$stamp" ]] || return 1
    cmp -s "$stamp" <(depth_validation_fingerprint "$manifest")
}

write_depth_validation_stamp() {
    local manifest="$1" stamp temporary
    stamp="$(depth_validation_stamp_path "$manifest")"
    mkdir -p -- "$(dirname "$stamp")"
    temporary="${stamp}.tmp-$$"
    depth_validation_fingerprint "$manifest" > "$temporary"
    mv -f -- "$temporary" "$stamp"
}

ensure_depth_sidecars() {
    require_environment MOGE_MODEL DEPTH_SIDECAR_ROOT
    local manifest_output manifest
    local -a builder_root_args
    manifest_output="$(resolve_pipeline_manifests touchanything train,val,test_seen,test_unseen)"
    mapfile -t manifests < <(printf '%s\n' "$manifest_output" | sed '/^[[:space:]]*$/d')
    for manifest in "${manifests[@]}"; do
        if depth_validation_cached "$manifest"; then
            echo "[pipeline] depth sidecar validation cache hit: $(basename "$manifest")"
            continue
        fi
        echo "[pipeline] validating depth sidecar coverage: $(basename "$manifest")"
        if "$DEPTH_PYTHON" "$SCRIPT_DIR/depth_sidecar.py" validate \
            "$DEPTH_SIDECAR_ROOT" --manifest "$manifest" >/dev/null 2>&1; then
            write_depth_validation_stamp "$manifest"
            echo "[pipeline] depth sidecar validated: $(basename "$manifest")"
            continue
        fi
        echo "[pipeline] building depth sidecar: $(basename "$manifest")"
        builder_root_args=()
        if [[ -n "${DEPTH_DATA_ROOT:-}" ]]; then
            builder_root_args+=(--data-root "$DEPTH_DATA_ROOT")
        fi
        run_depth_build_shards \
            --manifest "$manifest" \
            --model "$MOGE_MODEL" \
            --output-dir "$DEPTH_SIDECAR_ROOT" \
            "${builder_root_args[@]}"
        "$DEPTH_PYTHON" "$SCRIPT_DIR/depth_sidecar.py" validate \
            "$DEPTH_SIDECAR_ROOT" --manifest "$manifest" >/dev/null
        write_depth_validation_stamp "$manifest"
    done
}

ensure_vlm_caches() {
    require_environment VLM_MODEL QWEN_EMBED_CODE_ROOT VLM_CACHE_ROOT
    local split
    resolve_pipeline_manifests touchanything train,val,test_seen,test_unseen >/dev/null
    for split in train val test_seen test_unseen; do
        if partition_cache_complete "$VLM_CACHE_ROOT/$split"; then
            echo "[pipeline] VLM cache complete: $split"
            continue
        fi
        run_vlm_cache_partitions \
            --dataset touchanything \
            --split "$split" \
            --cache-dir "$VLM_CACHE_ROOT/$split"
    done
}

ensure_tactile_cache_split() {
    local root="$1" split="$2" fields="$3"
    shift 3
    if partition_cache_complete "$root/$split"; then
        echo "[pipeline] frozen tactile cache complete: $root/$split"
        return 0
    fi
    run_tactile_cache_partitions \
        --cache-dir "$root/$split" \
        --split "$split" \
        --fields "$fields" \
        "$@"
}

run_pipeline_evaluation() {
    local exp_name="$1" profile="$2" family="$3" base_cache_root="$4"
    local selector_spec="${5:-loss-best}"
    local checkpoint_root="${PRIOR_EXPERIMENT_ROOT:-$INPUT_PRIOR_ROOT/experiments}/$exp_name/checkpoints"
    local report_root="${PRIOR_REPORT_ROOT:-$INPUT_PRIOR_ROOT/reports}/$exp_name"
    local eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    IFS=',' read -r -a eval_gpu_array <<< "$eval_gpus"
    local selector split
    local -a selectors
    local -a cache_args
    IFS=',' read -r -a selectors <<< "$selector_spec"
    for selector in "${selectors[@]}"; do
        if [[ "$selector" != "loss-best" && "$selector" != "last" ]]; then
            echo "Evaluation selectors must be loss-best and/or last, got: $selector" >&2
            return 2
        fi
        if [[ ! -f "$checkpoint_root/$selector.ckpt" ]]; then
            echo "Missing pipeline checkpoint: $checkpoint_root/$selector.ckpt" >&2
            return 1
        fi
        for split in test_seen test_unseen; do
            cache_args=()
            if [[ "$family" == "vlm" ]]; then
                cache_args+=(--prior-feature-cache "$VLM_CACHE_ROOT/$split")
            fi
            if [[ "$profile" == "cached" ]]; then
                cache_args+=(--base-feature-cache "$base_cache_root/$split" --cache-only)
            fi
            CUDA_VISIBLE_DEVICES="$eval_gpus" \
            "$TACTILE_PYTHON" -m torch.distributed.run \
                --standalone \
                --nproc_per_node "${#eval_gpu_array[@]}" \
                "$SCRIPT_DIR/eval_prior_adapter.py" \
                --checkpoint "$checkpoint_root/$selector.ckpt" \
                --split "$split" \
                --output-dir "$report_root/$selector/$split" \
                "${cache_args[@]}"
        done
    done
}

run_pipeline_stage() {
    local stage="$1" profile="$2" experiment="${3:-}"
    shift 3 || true
    TACTILE_BASE_CHECKPOINT="${TACTILE_BASE_CHECKPOINT:-/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/hamer_tactile_ft/checkpoints/touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12/best_loss.ckpt}"
    DINO_WEIGHTS="${DINO_WEIGHTS:-/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
    MOGE_MODEL="${MOGE_MODEL:-/home/ma-user/work/cfzhao/input_prior_step0/models/moge-2-vitl-normal/model.pt}"
    export TACTILE_BASE_CHECKPOINT DINO_WEIGHTS MOGE_MODEL
    require_environment TACTILE_BASE_CHECKPOINT DINO_WEIGHTS
    local family control adapter_type prior_dim exp_suffix exp_name
    case "$experiment" in
        depth-real)
            family=depth; control=real; adapter_type=depth_spatial; prior_dim=8; exp_suffix=real
            ;;
        depth-shuffle)
            family=depth; control=spatial_shuffle; adapter_type=depth_spatial; prior_dim=8; exp_suffix=shuffle
            ;;
        vlm-real)
            family=vlm; control=real; adapter_type=vlm_lowrank; prior_dim="${VLM_PRIOR_DIM:-2048}"; exp_suffix=real
            ;;
        vlm-control)
            family=vlm; control=context_shuffle; adapter_type=vlm_lowrank; prior_dim="${VLM_PRIOR_DIM:-2048}"; exp_suffix=shuffle
            ;;
        *)
            echo "Pipeline experiment must be depth-real, depth-shuffle, vlm-real, or vlm-control." >&2
            return 2
            ;;
    esac

    DEPTH_SIDECAR_ROOT="${DEPTH_SIDECAR_ROOT:-$INPUT_PRIOR_ROOT/depth_sidecars}"
    VLM_MODEL="${VLM_MODEL:-/home/ma-user/work/cfzhao/input_prior_step0/models/Qwen3-VL-Embedding-2B}"
    QWEN_EMBED_CODE_ROOT="${QWEN_EMBED_CODE_ROOT:-/home/ma-user/work/cfzhao/input_prior_step0/code/Qwen3-VL-Embedding}"
    VLM_CACHE_ROOT="${VLM_CACHE_ROOT:-$INPUT_PRIOR_ROOT/cache/qwen_fullframe}"
    PRIOR_EXPERIMENT_ROOT="${PRIOR_EXPERIMENT_ROOT:-$INPUT_PRIOR_ROOT/experiments}"
    PRIOR_REPORT_ROOT="${PRIOR_REPORT_ROOT:-$INPUT_PRIOR_ROOT/reports}"
    export DEPTH_SIDECAR_ROOT VLM_MODEL QWEN_EMBED_CODE_ROOT VLM_CACHE_ROOT
    export PRIOR_EXPERIMENT_ROOT PRIOR_REPORT_ROOT

    local base_parent base_tag lock_path base_cache_root
    base_parent="$(dirname "$(dirname "$TACTILE_BASE_CHECKPOINT")")"
    base_tag="$(basename "$base_parent")"
    base_cache_root="$INPUT_PRIOR_ROOT/cache/tactile/$base_tag/$family"
    lock_path="$INPUT_PRIOR_ROOT/state/pipeline-prepare-${family}.lock"
    if [[ "$stage" != "eval" ]]; then
        acquire_pipeline_lock "$lock_path"
        if [[ "$family" == "depth" ]]; then
            ensure_depth_sidecars
        else
            ensure_vlm_caches
        fi
        if [[ "$profile" == "cached" ]]; then
            local split
            for split in train val test_seen test_unseen; do
                if [[ "$family" == "depth" ]]; then
                    ensure_tactile_cache_split \
                        "$base_cache_root" "$split" \
                        z_rgb,depth_grid,tactile_signal,has_tactile \
                        --depth-sidecar-root "$DEPTH_SIDECAR_ROOT"
                else
                    ensure_tactile_cache_split \
                        "$base_cache_root" "$split" \
                        h_rgb,tactile_signal,has_tactile
                fi
            done
        fi
        release_pipeline_lock
    fi

    if [[ "$profile" == "online" ]]; then
        exp_name="ta_${family}_${exp_suffix}_online"
        unset BASE_FEATURE_CACHE BASE_FEATURE_CACHE_TRAIN BASE_FEATURE_CACHE_VAL
    else
        exp_name="ta_${family}_${exp_suffix}_cached"
        export BASE_FEATURE_CACHE_TRAIN="$base_cache_root/train"
        export BASE_FEATURE_CACHE_VAL="$base_cache_root/val"
    fi
    if [[ "$family" == "depth" ]]; then
        unset DEPTH_FEATURE_CACHE DEPTH_FEATURE_CACHE_TRAIN DEPTH_FEATURE_CACHE_VAL
    else
        export VLM_FEATURE_CACHE_TRAIN="$VLM_CACHE_ROOT/train"
        export VLM_FEATURE_CACHE_VAL="$VLM_CACHE_ROOT/val"
    fi

    case "$stage" in
        prepare)
            echo "[pipeline] preparation complete: experiment=$exp_name profile=$profile"
            ;;
        train)
            echo "[pipeline] training experiment=$exp_name profile=$profile"
            run_prior_train "$adapter_type" "$prior_dim" "$control" "$exp_name" "$@"
            ;;
        eval)
            local selectors="${1:-loss-best}"
            echo "[pipeline] evaluating selectors=$selectors experiment=$exp_name"
            run_pipeline_evaluation \
                "$exp_name" "$profile" "$family" "$base_cache_root" "$selectors"
            ;;
        full)
            echo "[pipeline] training experiment=$exp_name profile=$profile"
            run_prior_train "$adapter_type" "$prior_dim" "$control" "$exp_name" "$@"
            echo "[pipeline] evaluating loss-best and last for seen/unseen"
            run_pipeline_evaluation \
                "$exp_name" "$profile" "$family" "$base_cache_root" "loss-best,last"
            echo "[pipeline] complete: $exp_name"
            ;;
        *)
            echo "Unsupported pipeline stage: $stage" >&2
            return 2
            ;;
    esac
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
    cache-build)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/build_feature_cache.py" build "$@"
        ;;
    cache-verify)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/build_feature_cache.py" verify "$@"
        ;;
    cache-inspect)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/build_feature_cache.py" inspect "$@"
        ;;
    cache-self-check)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/build_feature_cache.py" self-check "$@"
        ;;
    cache-tactile)
        require_environment TACTILE_BASE_CHECKPOINT DINO_WEIGHTS
        "$TACTILE_PYTHON" "$SCRIPT_DIR/cache_tactile_features.py" \
            --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
            --dino-weights "$DINO_WEIGHTS" \
            "$@"
        ;;
    cache-tactile-8gpu)
        run_tactile_cache_partitions "$@"
        ;;
    cache-vlm)
        require_environment VLM_MODEL QWEN_EMBED_CODE_ROOT
        "$VLM_PYTHON" "$SCRIPT_DIR/cache_vlm_embeddings.py" \
            --model "$VLM_MODEL" \
            --qwen-code-root "$QWEN_EMBED_CODE_ROOT" \
            "$@"
        ;;
    cache-vlm-8gpu)
        run_vlm_cache_partitions "$@"
        ;;
    cache-vlm-auto-8gpu)
        require_environment VLM_CACHE_ROOT
        IFS=',' read -r -a vlm_splits <<< "${VLM_SPLITS:-train,val,test_seen,test_unseen}"
        for split in "${vlm_splits[@]}"; do
            echo "[cache-vlm-auto] split=$split"
            run_vlm_cache_partitions \
                --dataset "${VLM_DATASET:-touchanything}" \
                --split "$split" \
                --cache-dir "$VLM_CACHE_ROOT/$split" \
                "$@"
        done
        ;;
    train-depth-real)
        run_prior_train depth_spatial 8 real ta_depth_feature_real "$@"
        ;;
    train-depth-shuffle)
        run_prior_train depth_spatial 8 spatial_shuffle ta_depth_feature_shuffle "$@"
        ;;
    train-vlm-real)
        require_environment VLM_PRIOR_DIM
        run_prior_train vlm_lowrank "$VLM_PRIOR_DIM" real ta_vlm_feature_real "$@"
        ;;
    train-vlm-control)
        require_environment VLM_PRIOR_DIM
        run_prior_train vlm_lowrank "$VLM_PRIOR_DIM" context_shuffle ta_vlm_feature_shuffle "$@"
        ;;
    train-prior)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/train_prior_adapter.py" "$@"
        ;;
    eval-prior)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/eval_prior_adapter.py" "$@"
        ;;
    eval-prior-8gpu)
        eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
        IFS=',' read -r -a eval_gpu_array <<< "$eval_gpus"
        CUDA_VISIBLE_DEVICES="$eval_gpus" \
        "$TACTILE_PYTHON" -m torch.distributed.run \
            --standalone \
            --nproc_per_node "${#eval_gpu_array[@]}" \
            "$SCRIPT_DIR/eval_prior_adapter.py" "$@"
        ;;
    prior-process-list)
        "$TACTILE_PYTHON" "$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py" \
            --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
            --list-runs
        ;;
    prior-process-stop)
        "$TACTILE_PYTHON" "$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py" \
            --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
            --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
            --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
            --terminate-all
        ;;
    prior-debug-monitor)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/runtime_debug.py" \
            --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
            --output-dir "${PRIOR_DEBUG_ATTACH_DIR:-$INPUT_PRIOR_ROOT/logs/runtime_debug_attach}" \
            "$@"
        ;;
    pipeline-online)
        run_pipeline_stage full online "$@"
        ;;
    pipeline-cache-only)
        run_pipeline_stage full cached "$@"
        ;;
    prepare-online)
        run_pipeline_stage prepare online "$@"
        ;;
    train-online)
        run_pipeline_stage train online "$@"
        ;;
    eval-online)
        run_pipeline_stage eval online "$@"
        ;;
    prepare-cache-only)
        run_pipeline_stage prepare cached "$@"
        ;;
    train-cache-only)
        run_pipeline_stage train cached "$@"
        ;;
    eval-cache-only)
        run_pipeline_stage eval cached "$@"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac
