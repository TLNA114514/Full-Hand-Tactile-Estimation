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
PRIOR_EXPERIMENT_ROOT="${PRIOR_EXPERIMENT_ROOT:-$INPUT_PRIOR_ROOT/experiments}"
PRIOR_REPORT_ROOT="${PRIOR_REPORT_ROOT:-$INPUT_PRIOR_ROOT/reports}"
DEPTH_SIDECAR_ROOT="${DEPTH_SIDECAR_ROOT:-$INPUT_PRIOR_ROOT/depth_sidecars}"
DEPTH_BUILD_LOG_DIR="${DEPTH_BUILD_LOG_DIR:-$INPUT_PRIOR_ROOT/logs/depth_build}"
SELECTOR_CACHE_MODE="${SELECTOR_CACHE_MODE:-cached}"
SELECTOR_FEATURE_CACHE_ROOT="${SELECTOR_FEATURE_CACHE_ROOT:-$INPUT_PRIOR_ROOT/cache/selector_evidence}"
SELECTOR_CACHE_FIELDS="${SELECTOR_CACHE_FIELDS:-h_rgb,contact_neck,contact_anchor_logits,depth_grid,tactile_signal,has_tactile}"
SELECTOR_CACHE_FLOATING_DTYPE="${SELECTOR_CACHE_FLOATING_DTYPE:-float32}"
SELECTOR_CACHE_SHARD_SIZE="${SELECTOR_CACHE_SHARD_SIZE:-65536}"
TACTILE_HISTORY_REPLAY_ROOT="${TACTILE_HISTORY_REPLAY_ROOT:-$INPUT_PRIOR_ROOT/tactile_history_replay}"
TACTILE_HISTORY_GPUS="${TACTILE_HISTORY_GPUS:-0,1,2,3,4,5,6,7}"
TACTILE_HISTORY_EVAL_BATCH_SIZE="${TACTILE_HISTORY_EVAL_BATCH_SIZE:-128}"
TACTILE_HISTORY_EVAL_WORKERS="${TACTILE_HISTORY_EVAL_WORKERS:-8}"
TACTILE_HISTORY_PAIR_BATCH_SIZE="${TACTILE_HISTORY_PAIR_BATCH_SIZE:-4096}"
TACTILE_HISTORY_METRIC_DEVICE="${TACTILE_HISTORY_METRIC_DEVICE:-cuda:0}"
TACTILE_HISTORY_BOOTSTRAP_ITERATIONS="${TACTILE_HISTORY_BOOTSTRAP_ITERATIONS:-2000}"
TACTILE_HISTORY_BOOTSTRAP_CONFIDENCE="${TACTILE_HISTORY_BOOTSTRAP_CONFIDENCE:-0.95}"
TACTILE_HISTORY_BBOX_MANIFEST="${TACTILE_HISTORY_BBOX_MANIFEST:-$SCRIPT_DIR/../sam3_bbox_reconstruction/outputs/full_reconstruction_flow/touchanything/manifests/touchanything_sam3_v1_highconf.jsonl}"
TEMPORAL_FEATURE_CACHE_ROOT="${TEMPORAL_FEATURE_CACHE_ROOT:-$INPUT_PRIOR_ROOT/cache/temporal_flow}"
TEMPORAL_PAIR_ROOT="${TEMPORAL_PAIR_ROOT:-$INPUT_PRIOR_ROOT/cache/temporal_pairs}"
TEMPORAL_EXPERIMENT_ROOT="${TEMPORAL_EXPERIMENT_ROOT:-$INPUT_PRIOR_ROOT/temporal_experiments}"
TEMPORAL_REPORT_ROOT="${TEMPORAL_REPORT_ROOT:-$INPUT_PRIOR_ROOT/temporal_reports}"
TEMPORAL_CACHE_FIELDS="${TEMPORAL_CACHE_FIELDS:-palm_base_logits,palm_tactile_signal,has_tactile}"
TEMPORAL_CACHE_SHARD_SIZE="${TEMPORAL_CACHE_SHARD_SIZE:-65536}"
TEMPORAL_BBOX_MANIFEST="${TEMPORAL_BBOX_MANIFEST:-$TACTILE_HISTORY_BBOX_MANIFEST}"
PIPELINE_LOCK_PATH=""
PIPELINE_LOCK_TOKEN=""
export INPUT_PRIOR_ROOT PRIOR_EXPERIMENT_ROOT PRIOR_REPORT_ROOT

prefer_python() {
    local current="$1" preferred="$2"
    if [[ "$current" == "python" && -x "$preferred" ]]; then
        printf '%s\n' "$preferred"
    else
        printf '%s\n' "$current"
    fi
}

has_cli_option() {
    local expected="$1"
    shift
    local value
    for value in "$@"; do
        if [[ "$value" == "$expected" || "$value" == "$expected="* ]]; then
            return 0
        fi
    done
    return 1
}

cli_option_value() {
    local expected="$1"
    shift
    local previous="" value
    for value in "$@"; do
        if [[ "$previous" == "$expected" ]]; then
            printf '%s\n' "$value"
            return 0
        fi
        if [[ "$value" == "$expected="* ]]; then
            printf '%s\n' "${value#*=}"
            return 0
        fi
        previous="$value"
    done
    return 1
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
  audit-depth-crop-coverage
                           Audit current SAM3 crop coverage after Depth reprojection

Temporal tactile data audit:
  audit-tactile-dynamics   Audit HDF5 sequence continuity, persistence, transport,
                           and pressure source/sink behavior without decoding RGB
  audit-bilateral-tactile-dynamics
                           Audit exact lags, per-hand persistence, anonymous track
                           provenance, and opposite-hand counterfactual histories
  audit-bilateral-tactile-dynamics-fast
                           Full CUDA audit with large batches and no per-pair CSV
  prepare-tactile-history-replay
                           Export palm-only RGB baseline predictions for val/seen/unseen
  audit-tactile-history-replay
                           Fit history strength on val and replay fixed controls on tests
  pipeline-tactile-history-replay
                           Content-addressed prediction export followed by replay audit
  prepare-temporal-flow    Build/reuse palm-only frozen RGB caches for train/val/tests
  train-temporal-flow      Train query-aware lag-1 residual without DINO/HDF5 in epochs
  train-tflow-signed-l1    Train signed additive lag-1 residual control
  train-tflow-signed-l124  Train signed additive lag-1/2/4 residual
  eval-temporal-flow-8gpu  Evaluate real history and strict cross-sequence control
  eval-tflow-signed-l1     Evaluate signed lag-1 best_loss on seen/unseen
  eval-tflow-signed-l124   Evaluate signed lag-1/2/4 best_loss on seen/unseen
  eval-tflow-confirmatory  Replay the frozen Step-3 candidates on seen/unseen
                           with matched sequence-clustered bootstrap
  train-tflow-selector-quality
                           Train diagnostic L1/L2 down-hold-up selector with per-lag quality
  train-tflow-selector-noquality
                           Matched selector control without time/bbox quality inputs
  train-tflow-selector-dino-aligned
                           Train selector with bbox-aligned historical DINO grids
  train-tflow-selector-dino-unwarped
                           Parameter-matched control without historical grid warping
  eval-tflow-selector-8gpu Evaluate one diagnostic selector checkpoint
  eval-tflow-selector-quality
  eval-tflow-selector-noquality
                           Evaluate selector-best on seen/unseen
  eval-tflow-selector-dino-aligned
  eval-tflow-selector-dino-unwarped
                           Evaluate strict-clear-best on val/seen/unseen with DINO controls
  eval-tflow-selector-dino-aligned-selector-best
  eval-tflow-selector-dino-unwarped-selector-best
                           Evaluate selector-best on val/seen/unseen with isolated
                           gate-zero, zero-motion, and DINO-only history controls
  eval-tflow-selector-pressure-8gpu
                           Low-level down-only policy selection/replay for one split
  audit-tflow-selector-pressure
                           Select on val, then replay fixed policies on val/seen/unseen
  audit-tflow-selector-mapping
                           Attribute selector errors to anchor-to-vertex mapping with
                           label-free RGB-matched history controls
  eval-tflow-selector-mapping-8gpu
                           Low-level mapping attribution for one checkpoint/split
  audit-temporal-flow-cache
                           Multi-GPU audit of lag 1/2/4 action spaces, gate algebra,
                           selectors, persistent errors, and gradient cancellation
  audit-tflow-long-horizon Audit lag 1/2/4/8/16/32 timing, conditional value,
                           trained lag masks, residual scales, and hand-swap controls

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
  prepare-selector-cache   Build/reuse content-addressed selector caches
  selector-cache-paths     Print resolved selector cache paths

Feature-level prior adapters (all implementation stays in this directory):
  train-depth-real         Aligned Depth spatial rectification
  train-depth-shuffle      Spatially shuffled Depth control
  train-depth-film-cf      Depth-only causal FiLM with paired control
  train-depth-xattn-cf     Local Depth cross-attention with paired control
  train-vlm-real           VLM low-rank bottleneck modulation
  train-vlm-control        Context-shuffled VLM control
  train-prior              Generic adapter training entry
  eval-prior               Generic frozen-base/fused evaluation entry
  eval-prior-8gpu          Eight-process evaluation with exact non-padded sharding
  eval-depth-controls-8gpu Same-checkpoint real/shuffle/mean/zero audit

Prior-aware frozen contact selector (pressure output remains unchanged):
  train-selector-depth-map    Depth rectifies the frozen selector neck
  train-selector-depth-anchor Depth predicts an independent anchor residual
  train-selector-depth-query  Canonical anchor queries read aligned RGB/Depth tokens
  train-selector-depth-query-shuffle
                              Same query head trained with shuffled Depth control
  train-selector-depth-query-clean-real
  train-selector-depth-query-clean-shuffle
                              Symmetric 10-epoch causal controls without identity loss
  train-selector-depth-query-contact-real
  train-selector-depth-query-contact-shuffle
                              Contact-only Depth controls with lean validation
  train-selector-vlm          Generic VLM local-evidence calibrator
  train-selector-vlm-siglip   SigLIP dual-view calibrator experiment
  train-selector-vlm-qwen     Qwen structured-state calibrator experiment
  train-prior-selector        Generic selector-prior training entry
  eval-prior-selector         Generic selector-prior evaluation entry
  eval-prior-selector-8gpu    Eight-process real/control selector evaluation
  audit-selector-pressure-8gpu
                              Val sweep, fixed val replay, then seen/unseen audit
  audit-selector-matched-pareto-8gpu
                              Full RGB-vs-prior matched-budget diagnostic audit
  audit-selector-exact-topk-8gpu
                              Exact per-frame RGB/Depth/control ranking audit
  selector-pressure-tiny-check
                              CPU/GPU-independent deterministic audit checks
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
  depth-real | depth-shuffle | depth-film-cf | depth-xattn-cf | vlm-real | vlm-control

Common environment:
  DEPTH_PYTHON             Python with MoGe, NumPy, OpenCV, and h5py
  DEPTH_GPUS               Comma-separated builder GPUs (default: 0,...,7)
  DEPTH_DATASET            touchanything|opentouch (default: touchanything)
  DEPTH_SPLITS             Comma-separated splits or auto (default: auto)
  DEPTH_DATA_ROOT          Optional processed sequence-HDF5 root
  DEPTH_BUILD_LOG_DIR      Per-shard logs outside the source tree
  TACTILE_PYTHON           Existing tactile environment Python
  TACTILE_BASE_CHECKPOINT  Compact FullGrid loss-best base checkpoint
  TACTILE_SELECTOR_CHECKPOINT Binary Grid selector best checkpoint
  DINO_WEIGHTS             Local DINOv3 H+/16 weights
  DEPTH_SIDECAR_ROOT       Versioned MoGe sidecar root
                           (default: /home/ma-user/work/cfzhao/input_prior_full/depth_sidecars)
  BASE_FEATURE_CACHE       Optional frozen z/h/logit cache
  SELECTOR_CACHE_MODE      cached (default) or online
  SELECTOR_FEATURE_CACHE_ROOT Persistent selector cache parent
  SELECTOR_CACHE_LOCK_TIMEOUT_SECONDS Cross-host cache wait (default: 21600)
  SELECTOR_CACHE_BREAK_STALE_LOCK Set to 1 only to reclaim a confirmed dead owner
  VLM_FEATURE_CACHE        Cache containing vlm_embedding
  VLM_PRIOR_DIM            VLM embedding dimension (Qwen3-VL-2B: 2048)
  VLM_MODEL                Local Qwen3-VL-Embedding model directory
  QWEN_EMBED_CODE_ROOT     Official Qwen3-VL-Embedding checkout
  VLM_CACHE_ROOT           Parent of split-specific VLM caches
  TACTILE_HISTORY_REPLAY_ROOT Persistent prediction/replay artifact root
  TACTILE_HISTORY_GPUS     GPUs used for exact prediction export (default: 0,...,7)
  TACTILE_HISTORY_BBOX_MANIFEST Current reviewed TouchAnything SAM3 manifest
  TACTILE_HISTORY_BOOTSTRAP_ITERATIONS Sequence-clustered resamples (default: 2000)
  TEMPORAL_FEATURE_CACHE_ROOT Persistent palm-logit/target cache root
  TEMPORAL_DINO_CACHE_SHARD_SIZE DINO temporal cache shard rows (default: 8192)
  TEMPORAL_DINO_EVAL_BATCH_SIZE DINO selector eval batch/rank (default: 256)
  TEMPORAL_EXPERIMENT_ROOT  Temporal checkpoints and local/W&B logs

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

  ./tactile_input_priors/run.sh audit-tactile-dynamics \
    --dataset touchanything --splits train,val,test_seen,test_unseen

  ./tactile_input_priors/run.sh audit-bilateral-tactile-dynamics \
    --dataset touchanything --splits train,val,test_seen,test_unseen

  TACTILE_BASE_CHECKPOINT=/path/to/loss-best.ckpt \
  DINO_WEIGHTS=/path/to/dinov3.pth \
  ./tactile_input_priors/run.sh pipeline-tactile-history-replay

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
    if [[ -f "$root/CACHE_DONE.json" && -f "$root/cache_config.json" ]]; then
        return 0
    fi
    local -a parts
    local basename expected part
    shopt -s nullglob
    parts=("$root"/part-*-of-*)
    shopt -u nullglob
    if (( ${#parts[@]} == 0 )); then
        return 1
    fi
    basename="$(basename "${parts[0]}")"
    if [[ ! "$basename" =~ ^part-([0-9]+)-of-([0-9]+)$ ]]; then
        return 1
    fi
    expected="$((10#${BASH_REMATCH[2]}))"
    if (( ${#parts[@]} != expected )); then
        return 1
    fi
    for ((part = 0; part < expected; part++)); do
        if [[ ! -f "$root/part-$(printf '%02d' "$part")-of-$(printf '%02d' "$expected")/CACHE_DONE.json" ]]; then
            return 1
        fi
    done
    return 0
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
    if [[ "$adapter_type" == depth_* ]]; then
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

run_prior_selector_train() {
    local adapter_type="$1" prior_dim="$2" counterfactual="$3" exp_name="$4"
    shift 4
    require_environment TACTILE_SELECTOR_CHECKPOINT DINO_WEIGHTS
    local cache_args=()
    local automatic_selector_cache=0
    local train_cache="" val_cache=""
    if [[ "$SELECTOR_CACHE_MODE" != "cached" && "$SELECTOR_CACHE_MODE" != "online" ]]; then
        echo "SELECTOR_CACHE_MODE must be cached or online, got: $SELECTOR_CACHE_MODE" >&2
        return 2
    fi
    if [[ "$adapter_type" == depth_* && "$SELECTOR_CACHE_MODE" == "cached" && \
          -z "${BASE_FEATURE_CACHE:-}${BASE_FEATURE_CACHE_TRAIN:-}${BASE_FEATURE_CACHE_VAL:-}" ]]; then
        local option
        for option in --datasets --data-roots --train-query-manifests \
                      --val-query-manifests --bbox-manifests --bbox-source-policy \
                      --bbox-rescale-factor --input-resolution; do
            if has_cli_option "$option" "$@"; then
                echo "Automatic selector caching fixes the audited TA/crop1.2/r256 data contract." >&2
                echo "For $option overrides, set explicit BASE_FEATURE_CACHE_TRAIN/VAL or SELECTOR_CACHE_MODE=online." >&2
                return 2
            fi
        done
        if [[ "${SELECTOR_DEPTH_COVERAGE_AUDIT:-1}" == "1" ]]; then
            local coverage_root="${SELECTOR_DEPTH_COVERAGE_AUDIT_ROOT:-$INPUT_PRIOR_ROOT/audits/depth_crop_coverage/train}"
            "$TACTILE_PYTHON" "$SCRIPT_DIR/audit_depth_crop_coverage.py" \
                --datasets "${SELECTOR_CACHE_DATASETS:-touchanything}" \
                --split train \
                --bbox-rescale-factor "${SELECTOR_CACHE_BBOX_SCALE:-1.2}" \
                --bbox-source-policy "${SELECTOR_CACHE_BBOX_POLICY:-sam3_only}" \
                --input-resolution "${SELECTOR_CACHE_INPUT_RESOLUTION:-256x192}" \
                --depth-sidecar-root "$DEPTH_SIDECAR_ROOT" \
                --hdf5-manifest-cache-dir "${SELECTOR_HDF5_MANIFEST_CACHE_DIR:-$INPUT_PRIOR_ROOT/cache/hdf5_manifests}" \
                --max-samples "${SELECTOR_DEPTH_COVERAGE_SAMPLES:-4096}" \
                --num-workers "${SELECTOR_DEPTH_COVERAGE_WORKERS:-4}" \
                --output-dir "$coverage_root" \
                --reuse-if-current
        fi
        ensure_selector_cache_split train
        train_cache="$SELECTOR_CACHE_RESOLVED_PATH"
        ensure_selector_cache_split val
        val_cache="$SELECTOR_CACHE_RESOLVED_PATH"
        cache_args+=(
            --train-base-feature-cache "$train_cache"
            --val-base-feature-cache "$val_cache"
            --no-train-augmentation
            --cache-only
        )
        automatic_selector_cache=1
        if ! has_cli_option --batch-size "$@"; then
            cache_args+=(--batch-size "${SELECTOR_TRAIN_BATCH_SIZE:-256}")
        fi
        if ! has_cli_option --num-workers "$@"; then
            cache_args+=(--num-workers "${SELECTOR_CACHE_NUM_WORKERS:-2}")
        fi
        if ! has_cli_option --val-num-workers "$@"; then
            cache_args+=(--val-num-workers "${SELECTOR_CACHE_VAL_NUM_WORKERS:-1}")
        fi
        if ! has_cli_option --prefetch-factor "$@"; then
            cache_args+=(--prefetch-factor "${SELECTOR_CACHE_PREFETCH_FACTOR:-1}")
        fi
        echo "[selector-cache] cache-only training enabled; DINO/RGB HDF5 are bypassed"
        echo "[selector-cache] train=$train_cache"
        echo "[selector-cache] val=$val_cache"
        exp_name="${exp_name}${SELECTOR_CACHE_EXP_SUFFIX:-_cached}"
        echo "[selector-cache] isolated experiment name: $exp_name"
    fi
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
    if [[ "$adapter_type" == depth_* && "$automatic_selector_cache" == "0" ]]; then
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
    elif [[ "$adapter_type" != depth_* ]]; then
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
        -- "$TACTILE_PYTHON" "$SCRIPT_DIR/train_prior_selector.py" \
        --adapter-type "$adapter_type" \
        --prior-dim "$prior_dim" \
        --counterfactual-control "$counterfactual" \
        --exp-name "$exp_name" \
        --selector-checkpoint "$TACTILE_SELECTOR_CHECKPOINT" \
        --dino-weights "$DINO_WEIGHTS" \
        "${cache_args[@]}" \
        "$@"
}

run_depth_control_audit() {
    if (( $# < 3 )); then
        echo "Usage: run.sh eval-depth-controls-8gpu CHECKPOINT OUTPUT_ROOT SPLIT [eval options]" >&2
        return 2
    fi
    local checkpoint="$1" output_root="$2" split="$3"
    shift 3
    local eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    local -a eval_gpu_array controls
    IFS=',' read -r -a eval_gpu_array <<< "$eval_gpus"
    controls=(real spatial_shuffle sample_shuffle global_mean zero)
    local control
    for control in "${controls[@]}"; do
        echo "[depth-control-audit] split=$split control=$control"
        CUDA_VISIBLE_DEVICES="$eval_gpus" \
        "$TACTILE_PYTHON" -m torch.distributed.run \
            --standalone \
            --nproc_per_node "${#eval_gpu_array[@]}" \
            "$SCRIPT_DIR/eval_prior_adapter.py" \
            --checkpoint "$checkpoint" \
            --split "$split" \
            --prior-control "$control" \
            --output-dir "$output_root/$control/$split" \
            "$@"
    done
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

selector_cache_build_args() {
    local split="$1"
    require_environment TACTILE_SELECTOR_CHECKPOINT DINO_WEIGHTS
    local manifest_cache="${SELECTOR_HDF5_MANIFEST_CACHE_DIR:-$INPUT_PRIOR_ROOT/cache/hdf5_manifests}"
    mkdir -p "$manifest_cache"
    SELECTOR_CACHE_BUILD_ARGS=(
        --fields "$SELECTOR_CACHE_FIELDS"
        --floating-dtype "$SELECTOR_CACHE_FLOATING_DTYPE"
        --shard-size "$SELECTOR_CACHE_SHARD_SIZE"
        --datasets "${SELECTOR_CACHE_DATASETS:-touchanything}"
        --split "$split"
        --bbox-rescale-factor "${SELECTOR_CACHE_BBOX_SCALE:-1.2}"
        --bbox-source-policy "${SELECTOR_CACHE_BBOX_POLICY:-sam3_only}"
        --input-resolution "${SELECTOR_CACHE_INPUT_RESOLUTION:-256x192}"
        --hdf5-manifest-cache-dir "$manifest_cache"
        --hdf5-handle-cache-size "${SELECTOR_CACHE_HDF5_HANDLES:-4}"
        --batch-size "${SELECTOR_CACHE_BATCH_SIZE:-128}"
        --lock-timeout-seconds "${SELECTOR_CACHE_LOCK_TIMEOUT_SECONDS:-21600}"
    )
    if [[ "${SELECTOR_CACHE_BREAK_STALE_LOCK:-0}" == "1" ]]; then
        SELECTOR_CACHE_BUILD_ARGS+=(--break-stale-lock)
    fi
    if [[ ",$SELECTOR_CACHE_FIELDS," == *,depth_grid,* ]]; then
        require_environment DEPTH_SIDECAR_ROOT
        SELECTOR_CACHE_BUILD_ARGS+=(--depth-sidecar-root "$DEPTH_SIDECAR_ROOT")
    fi
    if [[ -n "${DEPTH_DATA_ROOT:-}" ]]; then
        SELECTOR_CACHE_BUILD_ARGS+=(--data-roots "$DEPTH_DATA_ROOT")
    fi
}

resolve_selector_cache_path() {
    local split="$1"
    selector_cache_build_args "$split"
    local -a gpu_array
    local key
    IFS=',' read -r -a gpu_array <<< "${CACHE_GPUS:-0,1,2,3,4,5,6,7}"
    if (( ${#gpu_array[@]} == 0 )); then
        echo "CACHE_GPUS must contain at least one GPU." >&2
        return 2
    fi
    key="$(
        "$TACTILE_PYTHON" "$SCRIPT_DIR/cache_tactile_features.py" \
            --base-checkpoint "$TACTILE_SELECTOR_CHECKPOINT" \
            --dino-weights "$DINO_WEIGHTS" \
            "${SELECTOR_CACHE_BUILD_ARGS[@]}" \
            --num-partitions "${#gpu_array[@]}" \
            --print-cache-key
    )"
    if [[ ! "$key" =~ ^[0-9a-f]{64}$ ]]; then
        echo "Could not resolve selector cache identity; got: $key" >&2
        return 1
    fi
    SELECTOR_CACHE_RESOLVED_PATH="$SELECTOR_FEATURE_CACHE_ROOT/${split}-${key:0:20}"
}

ensure_selector_cache_split() {
    local split="$1"
    resolve_selector_cache_path "$split"
    local path="$SELECTOR_CACHE_RESOLVED_PATH"
    if partition_cache_complete "$path"; then
        echo "[selector-cache] reuse split=$split path=$path"
        return 0
    fi
    echo "[selector-cache] build split=$split path=$path"
    TACTILE_BASE_CHECKPOINT="$TACTILE_SELECTOR_CHECKPOINT" \
    CACHE_LOG_DIR="$path/logs" \
    run_tactile_cache_partitions \
        --cache-dir "$path" \
        "${SELECTOR_CACHE_BUILD_ARGS[@]}"
    if ! partition_cache_complete "$path"; then
        echo "Selector cache builders returned without a complete partition set: $path" >&2
        return 1
    fi
    echo "[selector-cache] ready split=$split path=$path"
}

prepare_selector_caches() {
    local raw_splits="${SELECTOR_CACHE_SPLITS:-train,val,test_seen,test_unseen}"
    local -a splits
    local split
    IFS=',' read -r -a splits <<< "$raw_splits"
    for split in "${splits[@]}"; do
        ensure_selector_cache_split "$split"
    done
}

print_selector_cache_paths() {
    local raw_splits="${SELECTOR_CACHE_SPLITS:-train,val,test_seen,test_unseen}"
    local -a splits
    local split
    IFS=',' read -r -a splits <<< "$raw_splits"
    for split in "${splits[@]}"; do
        resolve_selector_cache_path "$split"
        printf '%s=%s\n' "$split" "$SELECTOR_CACHE_RESOLVED_PATH"
    done
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
    local family control adapter_type prior_dim exp_suffix exp_name base_exp_name
    local -a experiment_args=()
    case "$experiment" in
        depth-real)
            family=depth; control=real; adapter_type=depth_spatial; prior_dim=8; exp_suffix=real
            ;;
        depth-shuffle)
            family=depth; control=spatial_shuffle; adapter_type=depth_spatial; prior_dim=8; exp_suffix=shuffle
            ;;
        depth-film-cf)
            family=depth; control=real; adapter_type=depth_causal_film; prior_dim=8
            exp_suffix=film_cf; base_exp_name=ta_dfilm_cf_r256
            experiment_args=(
                --feature-rms-budget 0.02
                --logit-delta-max 0.25
                --prior-dropout 0
                --counterfactual-control spatial_shuffle
                --control-identity-weight 0.01
                --feature-budget-penalty-weight 0.005
                --zero-mean-logit-residual
            )
            ;;
        depth-xattn-cf)
            family=depth; control=real; adapter_type=depth_local_xattn; prior_dim=8
            exp_suffix=xattn_cf; base_exp_name=ta_dxattn_cf_r256
            experiment_args=(
                --feature-rms-budget 0.02
                --logit-delta-max 0.25
                --prior-dropout 0
                --counterfactual-control spatial_shuffle
                --control-identity-weight 0.01
                --feature-budget-penalty-weight 0.005
                --zero-mean-logit-residual
                --depth-attention-heads 4
                --depth-attention-window 5
            )
            ;;
        vlm-real)
            family=vlm; control=real; adapter_type=vlm_lowrank; prior_dim="${VLM_PRIOR_DIM:-2048}"; exp_suffix=real
            ;;
        vlm-control)
            family=vlm; control=context_shuffle; adapter_type=vlm_lowrank; prior_dim="${VLM_PRIOR_DIM:-2048}"; exp_suffix=shuffle
            ;;
        *)
            echo "Pipeline experiment must be depth-real, depth-shuffle, depth-film-cf, depth-xattn-cf, vlm-real, or vlm-control." >&2
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
        exp_name="${base_exp_name:-ta_${family}_${exp_suffix}_online}"
        unset BASE_FEATURE_CACHE BASE_FEATURE_CACHE_TRAIN BASE_FEATURE_CACHE_VAL
    else
        exp_name="${base_exp_name:-ta_${family}_${exp_suffix}}_cached"
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
            run_prior_train "$adapter_type" "$prior_dim" "$control" "$exp_name" \
                "${experiment_args[@]}" "$@"
            ;;
        eval)
            local selectors="${1:-loss-best}"
            echo "[pipeline] evaluating selectors=$selectors experiment=$exp_name"
            run_pipeline_evaluation \
                "$exp_name" "$profile" "$family" "$base_cache_root" "$selectors"
            ;;
        full)
            echo "[pipeline] training experiment=$exp_name profile=$profile"
            run_prior_train "$adapter_type" "$prior_dim" "$control" "$exp_name" \
                "${experiment_args[@]}" "$@"
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

resolve_selector_eval_dependencies() {
    local prior_checkpoint="$1"
    if [[ -z "${TACTILE_SELECTOR_CHECKPOINT:-}" ]]; then
        TACTILE_SELECTOR_CHECKPOINT="$(
            "$TACTILE_PYTHON" -c \
                'import sys; from tactile_input_priors.runtime import load_torch_checkpoint; print(load_torch_checkpoint(sys.argv[1]).get("selector_checkpoint", ""))' \
                "$prior_checkpoint"
        )"
    fi
    if [[ -z "${DINO_WEIGHTS:-}" ]]; then
        DINO_WEIGHTS="$(
            "$TACTILE_PYTHON" -c \
                'import sys; from tactile_input_priors.runtime import load_torch_checkpoint; print(load_torch_checkpoint(sys.argv[1]).get("dino_weights", ""))' \
                "$prior_checkpoint"
        )"
    fi
    require_environment TACTILE_SELECTOR_CHECKPOINT DINO_WEIGHTS
    if [[ ! -f "$TACTILE_SELECTOR_CHECKPOINT" || ! -f "$DINO_WEIGHTS" ]]; then
        echo "Selector/DINO paths embedded in the prior checkpoint are unavailable." >&2
        echo "Set TACTILE_SELECTOR_CHECKPOINT and DINO_WEIGHTS explicitly." >&2
        return 2
    fi
}

selector_eval_cache_args() {
    local prior_checkpoint="$1" split="$2"
    shift 2
    SELECTOR_EVAL_CACHE_ARGS=()
    if [[ "$SELECTOR_CACHE_MODE" == "online" ]] || \
       has_cli_option --base-feature-cache "$@" || \
       has_cli_option --val-base-feature-cache "$@"; then
        return 0
    fi
    if [[ "$SELECTOR_CACHE_MODE" != "cached" ]]; then
        echo "SELECTOR_CACHE_MODE must be cached or online, got: $SELECTOR_CACHE_MODE" >&2
        return 2
    fi
    resolve_selector_eval_dependencies "$prior_checkpoint"
    local explicit_depth=""
    explicit_depth="$(cli_option_value --depth-sidecar-root "$@" || true)"
    if [[ -n "$explicit_depth" ]]; then
        DEPTH_SIDECAR_ROOT="$explicit_depth"
    fi
    ensure_selector_cache_split "$split"
    SELECTOR_EVAL_CACHE_ARGS=(
        --base-feature-cache "$SELECTOR_CACHE_RESOLVED_PATH"
        --cache-only
    )
    if ! has_cli_option --num-workers "$@"; then
        SELECTOR_EVAL_CACHE_ARGS+=(--num-workers "${SELECTOR_CACHE_EVAL_WORKERS:-4}")
    fi
    if ! has_cli_option --prefetch-factor "$@"; then
        SELECTOR_EVAL_CACHE_ARGS+=(--prefetch-factor "${SELECTOR_CACHE_PREFETCH_FACTOR:-1}")
    fi
}

run_prior_selector_eval() {
    local distributed="$1"
    shift
    local checkpoint split
    checkpoint="$(cli_option_value --checkpoint "$@" || true)"
    split="$(cli_option_value --split "$@" || true)"
    if [[ -z "$checkpoint" || -z "$split" ]]; then
        echo "Cached selector evaluation requires --checkpoint and --split." >&2
        return 2
    fi
    selector_eval_cache_args "$checkpoint" "$split" "$@"
    if [[ "$distributed" == "1" ]]; then
        local eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
        local -a eval_gpu_array
        IFS=',' read -r -a eval_gpu_array <<< "$eval_gpus"
        CUDA_VISIBLE_DEVICES="$eval_gpus" \
        "$TACTILE_PYTHON" -m torch.distributed.run \
            --standalone \
            --nproc_per_node "${#eval_gpu_array[@]}" \
            "$SCRIPT_DIR/eval_prior_selector.py" \
            "${SELECTOR_EVAL_CACHE_ARGS[@]}" "$@"
    else
        "$TACTILE_PYTHON" "$SCRIPT_DIR/eval_prior_selector.py" \
            "${SELECTOR_EVAL_CACHE_ARGS[@]}" "$@"
    fi
}

run_selector_pressure_audit() {
    if (( $# < 2 )); then
        echo "Usage: run.sh audit-selector-pressure-8gpu CHECKPOINT OUTPUT_ROOT [data options]" >&2
        return 2
    fi
    local checkpoint="$1" output_root="$2"
    shift 2
    if [[ ! -f "$checkpoint" && "$checkpoint" == /ta_* ]]; then
        local fallback_checkpoint="$PRIOR_EXPERIMENT_ROOT/${checkpoint#/}"
        if [[ -f "$fallback_checkpoint" ]]; then
            echo "[selector-audit] repaired empty-root checkpoint path: $fallback_checkpoint"
            checkpoint="$fallback_checkpoint"
        fi
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "Selector-prior checkpoint does not exist: $checkpoint" >&2
        local checkpoint_dir
        checkpoint_dir="$(dirname "$checkpoint")"
        if [[ -d "$checkpoint_dir" ]]; then
            echo "Available checkpoints under $checkpoint_dir:" >&2
            find "$checkpoint_dir" -maxdepth 1 -type f -name '*.ckpt' -printf '  %f\n' \
                | sort >&2
        else
            echo "Expected experiment root: $PRIOR_EXPERIMENT_ROOT" >&2
        fi
        return 2
    fi
    checkpoint="$(realpath "$checkpoint")"
    if [[ "$output_root" == /ta_* ]]; then
        output_root="$PRIOR_REPORT_ROOT/${output_root#/}"
        echo "[selector-audit] repaired empty-root report path: $output_root"
    fi
    mkdir -p "$output_root"
    output_root="$(realpath "$output_root")"
    local eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    local controls="${SELECTOR_AUDIT_CONTROLS:-real,cross_sequence,same_sequence_far,wrong_query,spatial_shuffle,global_mean,zero}"
    local -a eval_gpu_array eval_data_args
    IFS=',' read -r -a eval_gpu_array <<< "$eval_gpus"
    eval_data_args=("$@")
    if ! has_cli_option --depth-sidecar-root "${eval_data_args[@]}"; then
        if [[ ! -d "$DEPTH_SIDECAR_ROOT" ]]; then
            echo "Default Depth sidecar root does not exist: $DEPTH_SIDECAR_ROOT" >&2
            echo "Set DEPTH_SIDECAR_ROOT or pass --depth-sidecar-root explicitly." >&2
            return 2
        fi
        eval_data_args+=(--depth-sidecar-root "$DEPTH_SIDECAR_ROOT")
        echo "[selector-audit] depth sidecar root: $DEPTH_SIDECAR_ROOT"
    fi

    selector_eval_cache_args "$checkpoint" val "${eval_data_args[@]}"
    local -a val_cache_args=("${SELECTOR_EVAL_CACHE_ARGS[@]}")

    echo "[selector-audit] validation sweep -> $output_root/val"
    CUDA_VISIBLE_DEVICES="$eval_gpus" \
    "$TACTILE_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${#eval_gpu_array[@]}" \
        "$SCRIPT_DIR/eval_prior_selector.py" \
        "${val_cache_args[@]}" \
        "${eval_data_args[@]}" \
        --checkpoint "$checkpoint" \
        --split val \
        --output-dir "$output_root/val" \
        --controls "$controls" \
        --policy-sweep

    local selection="$output_root/val/policy_selection.json"
    if [[ ! -f "$selection" ]]; then
        echo "Validation did not produce $selection" >&2
        return 1
    fi
    echo "[selector-audit] fixed-policy control replay: val -> $output_root/val_control_replay"
    CUDA_VISIBLE_DEVICES="$eval_gpus" \
    "$TACTILE_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${#eval_gpu_array[@]}" \
        "$SCRIPT_DIR/eval_prior_selector.py" \
        "${val_cache_args[@]}" \
        "${eval_data_args[@]}" \
        --checkpoint "$checkpoint" \
        --split val \
        --output-dir "$output_root/val_control_replay" \
        --controls "$controls" \
        --policy-selection-in "$selection"

    local split
    for split in test_seen test_unseen; do
        selector_eval_cache_args "$checkpoint" "$split" "${eval_data_args[@]}"
        echo "[selector-audit] fixed validation policies: $split -> $output_root/$split"
        CUDA_VISIBLE_DEVICES="$eval_gpus" \
        "$TACTILE_PYTHON" -m torch.distributed.run \
            --standalone \
            --nproc_per_node "${#eval_gpu_array[@]}" \
            "$SCRIPT_DIR/eval_prior_selector.py" \
            "${SELECTOR_EVAL_CACHE_ARGS[@]}" \
            "${eval_data_args[@]}" \
            --checkpoint "$checkpoint" \
            --split "$split" \
            --output-dir "$output_root/$split" \
            --controls "$controls" \
            --policy-selection-in "$selection"
    done
    echo "[selector-audit] complete: $output_root"
}

run_selector_matched_pareto_audit() {
    if (( $# < 2 )); then
        echo "Usage: run.sh audit-selector-matched-pareto-8gpu CHECKPOINT OUTPUT_ROOT [data options]" >&2
        return 2
    fi
    local checkpoint="$1" output_root="$2"
    shift 2
    if [[ ! -f "$checkpoint" && "$checkpoint" == /ta_* ]]; then
        local fallback_checkpoint="$PRIOR_EXPERIMENT_ROOT/${checkpoint#/}"
        if [[ -f "$fallback_checkpoint" ]]; then
            echo "[selector-matched] repaired empty-root checkpoint path: $fallback_checkpoint"
            checkpoint="$fallback_checkpoint"
        fi
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "Selector-prior checkpoint does not exist: $checkpoint" >&2
        local checkpoint_dir
        checkpoint_dir="$(dirname "$checkpoint")"
        if [[ -d "$checkpoint_dir" ]]; then
            echo "Available checkpoints under $checkpoint_dir:" >&2
            find "$checkpoint_dir" -maxdepth 1 -type f -name '*.ckpt' -printf '  %f\n' \
                | sort >&2
        else
            echo "Expected experiment root: $PRIOR_EXPERIMENT_ROOT" >&2
        fi
        return 2
    fi
    checkpoint="$(realpath "$checkpoint")"
    if [[ "$output_root" == /ta_* ]]; then
        output_root="$PRIOR_REPORT_ROOT/${output_root#/}"
        echo "[selector-matched] repaired empty-root report path: $output_root"
    fi
    mkdir -p "$output_root"
    output_root="$(realpath "$output_root")"
    local eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    local -a eval_gpu_array eval_data_args
    IFS=',' read -r -a eval_gpu_array <<< "$eval_gpus"
    eval_data_args=("$@")
    if ! has_cli_option --depth-sidecar-root "${eval_data_args[@]}"; then
        if [[ ! -d "$DEPTH_SIDECAR_ROOT" ]]; then
            echo "Default Depth sidecar root does not exist: $DEPTH_SIDECAR_ROOT" >&2
            echo "Set DEPTH_SIDECAR_ROOT or pass --depth-sidecar-root explicitly." >&2
            return 2
        fi
        eval_data_args+=(--depth-sidecar-root "$DEPTH_SIDECAR_ROOT")
        echo "[selector-matched] depth sidecar root: $DEPTH_SIDECAR_ROOT"
    fi

    local split
    for split in val test_seen test_unseen; do
        selector_eval_cache_args "$checkpoint" "$split" "${eval_data_args[@]}"
        echo "[selector-matched] full-grid diagnostic: $split -> $output_root/$split"
        CUDA_VISIBLE_DEVICES="$eval_gpus" \
        "$TACTILE_PYTHON" -m torch.distributed.run \
            --standalone \
            --nproc_per_node "${#eval_gpu_array[@]}" \
            "$SCRIPT_DIR/eval_prior_selector.py" \
            "${SELECTOR_EVAL_CACHE_ARGS[@]}" \
            "${eval_data_args[@]}" \
            --checkpoint "$checkpoint" \
            --split "$split" \
            --output-dir "$output_root/$split" \
            --controls real \
            --policy-matched-pareto
    done
    echo "[selector-matched] complete: $output_root"
}

run_selector_exact_topk_audit() {
    if (( $# < 2 )); then
        echo "Usage: run.sh audit-selector-exact-topk-8gpu CHECKPOINT OUTPUT_ROOT [data options]" >&2
        return 2
    fi
    local checkpoint="$1" output_root="$2"
    shift 2
    if [[ ! -f "$checkpoint" && "$checkpoint" == /ta_* ]]; then
        local fallback_checkpoint="$PRIOR_EXPERIMENT_ROOT/${checkpoint#/}"
        if [[ -f "$fallback_checkpoint" ]]; then
            echo "[selector-exact] repaired empty-root checkpoint path: $fallback_checkpoint"
            checkpoint="$fallback_checkpoint"
        fi
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "Selector-prior checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    checkpoint="$(realpath "$checkpoint")"
    if [[ "$output_root" == /ta_* ]]; then
        output_root="$PRIOR_REPORT_ROOT/${output_root#/}"
        echo "[selector-exact] repaired empty-root report path: $output_root"
    fi
    mkdir -p "$output_root"
    output_root="$(realpath "$output_root")"
    local eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    local -a eval_gpu_array eval_data_args exact_common
    IFS=',' read -r -a eval_gpu_array <<< "$eval_gpus"
    eval_data_args=("$@")
    if ! has_cli_option --depth-sidecar-root "${eval_data_args[@]}"; then
        if [[ ! -d "$DEPTH_SIDECAR_ROOT" ]]; then
            echo "Default Depth sidecar root does not exist: $DEPTH_SIDECAR_ROOT" >&2
            echo "Set DEPTH_SIDECAR_ROOT or pass --depth-sidecar-root explicitly." >&2
            return 2
        fi
        eval_data_args+=(--depth-sidecar-root "$DEPTH_SIDECAR_ROOT")
    fi
    exact_common=(
        --controls real,spatial_shuffle,global_mean,zero
        --batch-size "${EXACT_TOPK_BATCH_SIZE:-512}"
        --num-workers "${EXACT_TOPK_NUM_WORKERS:-16}"
        --prefetch-factor 1
        --policy-bootstrap-iterations "${EXACT_TOPK_BOOTSTRAP_ITERATIONS:-2000}"
        --policy-bootstrap-seed "${EXACT_TOPK_BOOTSTRAP_SEED:-521}"
    )

    selector_eval_cache_args "$checkpoint" val "${eval_data_args[@]}"
    echo "[selector-exact] validation selection -> $output_root/val"
    CUDA_VISIBLE_DEVICES="$eval_gpus" \
    "$TACTILE_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${#eval_gpu_array[@]}" \
        "$SCRIPT_DIR/eval_prior_selector.py" \
        "${SELECTOR_EVAL_CACHE_ARGS[@]}" \
        "${eval_data_args[@]}" \
        "${exact_common[@]}" \
        --checkpoint "$checkpoint" \
        --split val \
        --output-dir "$output_root/val" \
        --policy-exact-topk \
        --exact-topk-values "${EXACT_TOPK_VALUES:-0,1,2,4,8,16,32,64}" \
        --exact-topk-alpha "${EXACT_TOPK_ALPHA:-1.0}" \
        --exact-topk-target-floor "${EXACT_TOPK_TARGET_FLOOR:-0.02}" \
        --exact-topk-target-removal-fraction "${EXACT_TOPK_TARGET_REMOVAL_FRACTION:-0.03}"

    local selection="$output_root/val/exact_topk_selection.json"
    if [[ ! -f "$selection" ]]; then
        echo "Validation did not produce $selection" >&2
        return 1
    fi
    local split
    for split in test_seen test_unseen; do
        selector_eval_cache_args "$checkpoint" "$split" "${eval_data_args[@]}"
        echo "[selector-exact] fixed validation top-k: $split -> $output_root/$split"
        CUDA_VISIBLE_DEVICES="$eval_gpus" \
        "$TACTILE_PYTHON" -m torch.distributed.run \
            --standalone \
            --nproc_per_node "${#eval_gpu_array[@]}" \
            "$SCRIPT_DIR/eval_prior_selector.py" \
            "${SELECTOR_EVAL_CACHE_ARGS[@]}" \
            "${eval_data_args[@]}" \
            "${exact_common[@]}" \
            --checkpoint "$checkpoint" \
            --split "$split" \
            --output-dir "$output_root/$split" \
            --exact-topk-selection-in "$selection"
    done
    echo "[selector-exact] complete: $output_root"
}

tactile_history_checkpoint_root() {
    require_environment TACTILE_BASE_CHECKPOINT DINO_WEIGHTS
    if [[ ! -f "$TACTILE_BASE_CHECKPOINT" ]]; then
        echo "Tactile base checkpoint does not exist: $TACTILE_BASE_CHECKPOINT" >&2
        return 2
    fi
    if [[ ! -f "$DINO_WEIGHTS" ]]; then
        echo "DINO weights do not exist: $DINO_WEIGHTS" >&2
        return 2
    fi
    if [[ ! -f "$TACTILE_HISTORY_BBOX_MANIFEST" ]]; then
        echo "TouchAnything SAM3 bbox manifest does not exist: $TACTILE_HISTORY_BBOX_MANIFEST" >&2
        return 2
    fi
    local checkpoint_sha
    checkpoint_sha="$(sha256sum "$TACTILE_BASE_CHECKPOINT" | cut -d' ' -f1)"
    printf '%s/%s\n' "$TACTILE_HISTORY_REPLAY_ROOT" "${checkpoint_sha:0:16}"
}

tactile_history_manifests() {
    local -a args=(
        --dataset touchanything
        --splits val,test_seen,test_unseen
        --create-missing
        --print-paths
    )
    if [[ -n "${DEPTH_DATA_ROOT:-}" ]]; then
        args+=(--processed-root "$DEPTH_DATA_ROOT")
    fi
    "$TACTILE_PYTHON" "$SCRIPT_DIR/resolve_depth_manifests.py" "${args[@]}"
}

tactile_history_split() {
    local name
    name="$(basename "$1")"
    name="${name#touchanything_}"
    printf '%s\n' "${name%.queries.jsonl}"
}

tactile_history_prediction_root() {
    local run_root="$1" manifest="$2" split manifest_sha bbox_sha artifact_sha
    split="$(tactile_history_split "$manifest")"
    manifest_sha="$(sha256sum "$manifest" | cut -d' ' -f1)"
    bbox_sha="$(sha256sum "$TACTILE_HISTORY_BBOX_MANIFEST" | cut -d' ' -f1)"
    artifact_sha="$(printf '%s:%s' "$manifest_sha" "$bbox_sha" | sha256sum | cut -d' ' -f1)"
    printf '%s/predictions/%s_%s\n' "$run_root" "$split" "${artifact_sha:0:16}"
}

temporal_manifest() {
    local split="$1"
    local -a args=(
        --dataset touchanything
        --splits "$split"
        --create-missing
        --print-paths
    )
    if [[ -n "${DEPTH_DATA_ROOT:-}" ]]; then
        args+=(--processed-root "$DEPTH_DATA_ROOT")
    fi
    "$TACTILE_PYTHON" "$SCRIPT_DIR/resolve_depth_manifests.py" "${args[@]}"
}

resolve_temporal_cache_path() {
    local split="$1" manifest key
    require_environment TACTILE_BASE_CHECKPOINT DINO_WEIGHTS
    manifest="$(temporal_manifest "$split")"
    local -a gpu_array
    IFS=',' read -r -a gpu_array <<< "${CACHE_GPUS:-0,1,2,3,4,5,6,7}"
    TEMPORAL_CACHE_ARGS=(
        --fields "$TEMPORAL_CACHE_FIELDS"
        --floating-dtype float16
        --shard-size "$TEMPORAL_CACHE_SHARD_SIZE"
        --datasets touchanything
        --split "$split"
        --query-manifests "$manifest"
        --bbox-manifests "$TEMPORAL_BBOX_MANIFEST"
        --bbox-source-policy sam3_only
        --bbox-rescale-factor 1.2
        --input-resolution 256x192
        --hdf5-manifest-cache-dir "${TEMPORAL_HDF5_MANIFEST_CACHE_DIR:-$INPUT_PRIOR_ROOT/cache/hdf5_manifests}"
        --hdf5-handle-cache-size "${TEMPORAL_CACHE_HDF5_HANDLES:-4}"
        --batch-size "${TEMPORAL_CACHE_BATCH_SIZE:-128}"
        --lock-timeout-seconds "${TEMPORAL_CACHE_LOCK_TIMEOUT_SECONDS:-21600}"
    )
    if [[ -n "${DEPTH_DATA_ROOT:-}" ]]; then
        TEMPORAL_CACHE_ARGS+=(--data-roots "$DEPTH_DATA_ROOT")
    fi
    key="$(
        "$TACTILE_PYTHON" "$SCRIPT_DIR/cache_tactile_features.py" \
            --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
            --dino-weights "$DINO_WEIGHTS" \
            "${TEMPORAL_CACHE_ARGS[@]}" \
            --num-partitions "${#gpu_array[@]}" \
            --print-cache-key
    )"
    if [[ ! "$key" =~ ^[0-9a-f]{64}$ ]]; then
        echo "Could not resolve temporal cache key; got: $key" >&2
        return 1
    fi
    TEMPORAL_CACHE_RESOLVED_PATH="$TEMPORAL_FEATURE_CACHE_ROOT/${split}-${key:0:20}"
    TEMPORAL_MANIFEST_RESOLVED_PATH="$manifest"
}

ensure_temporal_cache_split() {
    local split="$1"
    resolve_temporal_cache_path "$split"
    local path="$TEMPORAL_CACHE_RESOLVED_PATH"
    if partition_cache_complete "$path"; then
        echo "[temporal-cache] reuse split=$split path=$path"
        return 0
    fi
    echo "[temporal-cache] build split=$split path=$path"
    CACHE_LOG_DIR="$path/logs" run_tactile_cache_partitions \
        --cache-dir "$path" "${TEMPORAL_CACHE_ARGS[@]}"
    if ! partition_cache_complete "$path"; then
        echo "Temporal cache partition set is incomplete: $path" >&2
        return 1
    fi
}

prepare_temporal_flow() {
    local raw="${TEMPORAL_CACHE_SPLITS:-train,val,test_seen,test_unseen}"
    local -a splits
    local split
    IFS=',' read -r -a splits <<< "$raw"
    for split in "${splits[@]}"; do
        ensure_temporal_cache_split "$split"
    done
}

train_temporal_flow() {
    ensure_temporal_cache_split train
    local train_cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local train_manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    ensure_temporal_cache_split val
    local val_cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local val_manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    local process_supervisor
    process_supervisor="$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py"
    export OMP_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export OPENBLAS_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export MKL_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export NUMEXPR_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    "$TACTILE_PYTHON" "$process_supervisor" \
        --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
        --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
        --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
        -- "$TACTILE_PYTHON" "$SCRIPT_DIR/train_temporal_flow.py" \
        --train-cache "$train_cache" \
        --val-cache "$val_cache" \
        --train-query-manifests "$train_manifest" \
        --val-query-manifests "$val_manifest" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --pair-index-root "$TEMPORAL_PAIR_ROOT" \
        --output-root "$TEMPORAL_EXPERIMENT_ROOT" \
        "$@"
}

train_temporal_flow_signed_l1() {
    train_temporal_flow \
        --exp-name ta_tflow_sadd_l1_r256 \
        --temporal-architecture signed_additive \
        --history-lags 1 \
        --transition-loss-weight 0.01 \
        --history-gate-loss-weight 0.01 \
        --delta-l1-weight 0 \
        "$@"
}

train_temporal_flow_signed_l124() {
    train_temporal_flow \
        --exp-name ta_tflow_sadd_l124_r256 \
        --temporal-architecture signed_additive \
        --history-lags 1,2,4 \
        --transition-loss-weight 0.01 \
        --history-gate-loss-weight 0.01 \
        --delta-l1-weight 0 \
        "$@"
}

train_temporal_selector() {
    ensure_temporal_cache_split train
    local train_cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local train_manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    ensure_temporal_cache_split val
    local val_cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local val_manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    local process_supervisor
    process_supervisor="$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py"
    export OMP_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export OPENBLAS_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export MKL_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    export NUMEXPR_NUM_THREADS="${PRIOR_CPU_THREADS_PER_PROCESS:-1}"
    "$TACTILE_PYTHON" "$process_supervisor" \
        --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
        --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
        --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
        -- "$TACTILE_PYTHON" "$SCRIPT_DIR/train_temporal_selector.py" \
        --train-cache "$train_cache" \
        --val-cache "$val_cache" \
        --train-query-manifests "$train_manifest" \
        --val-query-manifests "$val_manifest" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --pair-index-root "$TEMPORAL_PAIR_ROOT" \
        --output-root "$TEMPORAL_EXPERIMENT_ROOT" \
        "$@"
}

train_tflow_selector_quality() {
    train_temporal_selector \
        --exp-name ta_tsel_l12_q_r256 \
        --history-lags 1,2 \
        --use-per-lag-quality \
        "$@"
}

train_tflow_selector_noquality() {
    train_temporal_selector \
        --exp-name ta_tsel_l12_noq_r256 \
        --history-lags 1,2 \
        --no-use-per-lag-quality \
        "$@"
}

train_tflow_selector_dino_aligned() {
    local TEMPORAL_CACHE_FIELDS="z_rgb,palm_base_logits,palm_tactile_signal,has_tactile"
    local TEMPORAL_CACHE_SHARD_SIZE="${TEMPORAL_DINO_CACHE_SHARD_SIZE:-8192}"
    train_temporal_selector \
        --exp-name ta_tsel_dino_align_r256 \
        --history-lags 1,2 \
        --no-use-per-lag-quality \
        --use-dino-history \
        --dino-alignment-mode aligned \
        "$@"
}

train_tflow_selector_dino_unwarped() {
    local TEMPORAL_CACHE_FIELDS="z_rgb,palm_base_logits,palm_tactile_signal,has_tactile"
    local TEMPORAL_CACHE_SHARD_SIZE="${TEMPORAL_DINO_CACHE_SHARD_SIZE:-8192}"
    train_temporal_selector \
        --exp-name ta_tsel_dino_unwarp_r256 \
        --history-lags 1,2 \
        --no-use-per-lag-quality \
        --use-dino-history \
        --dino-alignment-mode unwarped \
        "$@"
}

eval_temporal_flow_8gpu() {
    if (( $# < 2 )); then
        echo "Usage: run.sh eval-temporal-flow-8gpu CHECKPOINT SPLIT [eval options]" >&2
        return 2
    fi
    local checkpoint="$1" split="$2"
    shift 2
    if [[ ! -f "$checkpoint" ]]; then
        echo "Temporal checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    if [[ -z "${TACTILE_BASE_CHECKPOINT:-}" ]]; then
        TACTILE_BASE_CHECKPOINT="$(
            "$TACTILE_PYTHON" -c \
                'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu").get("base_checkpoint",""))' \
                "$checkpoint"
        )"
    fi
    ensure_temporal_cache_split "$split"
    local cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    local experiment_dir output_dir eval_gpus
    experiment_dir="$(dirname "$(dirname "$checkpoint")")"
    output_dir="${TEMPORAL_EVAL_OUTPUT_DIR:-$TEMPORAL_REPORT_ROOT/$(basename "$experiment_dir")/$(basename "$checkpoint" .ckpt)/$split}"
    eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    local -a gpu_array
    IFS=',' read -r -a gpu_array <<< "$eval_gpus"
    local process_supervisor
    process_supervisor="$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py"
    CUDA_VISIBLE_DEVICES="$eval_gpus" \
    "$TACTILE_PYTHON" "$process_supervisor" \
        --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
        --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
        --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
        -- "$TACTILE_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${#gpu_array[@]}" \
        "$SCRIPT_DIR/eval_temporal_flow.py" \
        --checkpoint "$checkpoint" \
        --cache "$cache" \
        --query-manifests "$manifest" \
        --pair-index-root "$TEMPORAL_PAIR_ROOT" \
        --split "$split" \
        --output-dir "$output_dir" \
        --copy-val-metrics-from "$experiment_dir/val_metrics.csv" \
        "$@"
}

eval_temporal_selector_8gpu() {
    if (( $# < 2 )); then
        echo "Usage: run.sh eval-tflow-selector-8gpu CHECKPOINT SPLIT [eval options]" >&2
        return 2
    fi
    local checkpoint="$1" split="$2"
    shift 2
    if [[ ! -f "$checkpoint" ]]; then
        echo "Temporal selector checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    if [[ -z "${TACTILE_BASE_CHECKPOINT:-}" ]]; then
        TACTILE_BASE_CHECKPOINT="$(
            "$TACTILE_PYTHON" -c \
                'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu").get("base_checkpoint",""))' \
                "$checkpoint"
        )"
    fi
    local checkpoint_uses_dino
    checkpoint_uses_dino="$(
        "$TACTILE_PYTHON" -c \
            'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu"); print(int(int(p.get("model_config",{}).get("dino_grid_channels",0)) > 0))' \
            "$checkpoint"
    )"
    local TEMPORAL_CACHE_FIELDS="$TEMPORAL_CACHE_FIELDS"
    local TEMPORAL_CACHE_SHARD_SIZE="$TEMPORAL_CACHE_SHARD_SIZE"
    if [[ "$checkpoint_uses_dino" == "1" ]]; then
        TEMPORAL_CACHE_FIELDS="z_rgb,palm_base_logits,palm_tactile_signal,has_tactile"
        TEMPORAL_CACHE_SHARD_SIZE="${TEMPORAL_DINO_CACHE_SHARD_SIZE:-8192}"
    fi
    ensure_temporal_cache_split "$split"
    local cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    local experiment_dir output_dir eval_gpus process_supervisor
    experiment_dir="$(dirname "$(dirname "$checkpoint")")"
    output_dir="${TEMPORAL_SELECTOR_EVAL_OUTPUT_DIR:-$TEMPORAL_REPORT_ROOT/$(basename "$experiment_dir")/$(basename "$checkpoint" .ckpt)/$split}"
    eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    local -a gpu_array
    IFS=',' read -r -a gpu_array <<< "$eval_gpus"
    process_supervisor="$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py"
    CUDA_VISIBLE_DEVICES="$eval_gpus" \
    "$TACTILE_PYTHON" "$process_supervisor" \
        --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
        --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
        --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
        -- "$TACTILE_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${#gpu_array[@]}" \
        "$SCRIPT_DIR/eval_temporal_selector.py" \
        --checkpoint "$checkpoint" \
        --cache "$cache" \
        --query-manifests "$manifest" \
        --pair-index-root "$TEMPORAL_PAIR_ROOT" \
        --split "$split" \
        --output-dir "$output_dir" \
        --copy-val-metrics-from "$experiment_dir/val_metrics.csv" \
        "$@"
}

eval_temporal_selector_experiment() {
    local experiment_name="$1"
    shift
    local checkpoint_name="${TEMPORAL_SELECTOR_EVAL_CHECKPOINT:-selector-best}"
    local checkpoint="$TEMPORAL_EXPERIMENT_ROOT/$experiment_name/checkpoints/$checkpoint_name.ckpt"
    local raw_splits="${TEMPORAL_EVAL_SPLITS:-test_seen,test_unseen}"
    local -a splits
    local split
    IFS=',' read -r -a splits <<< "$raw_splits"
    for split in "${splits[@]}"; do
        eval_temporal_selector_8gpu "$checkpoint" "$split" "$@"
    done
}

eval_tflow_selector_dino_experiment() {
    local experiment_name="$1"
    shift
    local TEMPORAL_SELECTOR_EVAL_CHECKPOINT="${TEMPORAL_SELECTOR_EVAL_CHECKPOINT:-strict-clear-best}"
    local TEMPORAL_EVAL_SPLITS="${TEMPORAL_EVAL_SPLITS:-val,test_seen,test_unseen}"
    eval_temporal_selector_experiment "$experiment_name" \
        --batch-size "${TEMPORAL_DINO_EVAL_BATCH_SIZE:-256}" \
        "$@"
}

eval_tflow_selector_dino_selector_best() {
    local experiment_name="$1"
    shift
    # strict-clear-best captured the early ranking optimum, while
    # selector-best captures the later generic action optimum. The causal DINO
    # controls must be run on both before choosing a visual-history branch.
    local TEMPORAL_SELECTOR_EVAL_CHECKPOINT="selector-best"
    eval_tflow_selector_dino_experiment "$experiment_name" "$@"
}

eval_temporal_selector_pressure_8gpu() {
    if (( $# < 3 )); then
        echo "Usage: run.sh eval-tflow-selector-pressure-8gpu CHECKPOINT SPLIT OUTPUT_DIR [audit options]" >&2
        return 2
    fi
    local checkpoint="$1" split="$2" output_dir="$3"
    shift 3
    if [[ ! -f "$checkpoint" ]]; then
        echo "Temporal selector checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    if [[ -z "${TACTILE_BASE_CHECKPOINT:-}" ]]; then
        TACTILE_BASE_CHECKPOINT="$(
            "$TACTILE_PYTHON" -c \
                'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu").get("base_checkpoint",""))' \
                "$checkpoint"
        )"
    fi
    ensure_temporal_cache_split "$split"
    local cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    local experiment_dir eval_gpus process_supervisor
    experiment_dir="$(dirname "$(dirname "$checkpoint")")"
    eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    local -a gpu_array
    IFS=',' read -r -a gpu_array <<< "$eval_gpus"
    process_supervisor="$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py"
    CUDA_VISIBLE_DEVICES="$eval_gpus" \
    "$TACTILE_PYTHON" "$process_supervisor" \
        --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
        --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
        --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
        -- "$TACTILE_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${#gpu_array[@]}" \
        "$SCRIPT_DIR/audit_temporal_selector_pressure.py" \
        --checkpoint "$checkpoint" \
        --cache "$cache" \
        --query-manifests "$manifest" \
        --pair-index-root "$TEMPORAL_PAIR_ROOT" \
        --split "$split" \
        --output-dir "$output_dir" \
        --batch-size "${TEMPORAL_POLICY_BATCH_SIZE:-512}" \
        --num-workers "${TEMPORAL_POLICY_WORKERS:-1}" \
        --prefetch-factor "${TEMPORAL_POLICY_PREFETCH_FACTOR:-1}" \
        --policy-chunk-size "${TEMPORAL_POLICY_CHUNK_SIZE:-4}" \
        --bootstrap-iterations "${TEMPORAL_POLICY_BOOTSTRAP_ITERATIONS:-2000}" \
        --copy-val-metrics-from "$experiment_dir/val_metrics.csv" \
        "$@"
}

audit_tflow_selector_pressure() {
    local checkpoint output_root experiment_dir checkpoint_name selection raw_splits split
    checkpoint="$TEMPORAL_EXPERIMENT_ROOT/ta_tsel_l12_noq_r256/checkpoints/selector-best.ckpt"
    if (( $# > 0 )) && [[ "$1" != --* ]]; then
        checkpoint="$1"
        shift
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "Temporal selector checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    experiment_dir="$(dirname "$(dirname "$checkpoint")")"
    checkpoint_name="$(basename "$checkpoint" .ckpt)"
    output_root="${TEMPORAL_SELECTOR_PRESSURE_OUTPUT_ROOT:-$TEMPORAL_REPORT_ROOT/$(basename "$experiment_dir")/$checkpoint_name/down_policy_v1}"
    selection="$output_root/val_selection/policy_selection.json"

    echo "[temporal-selector-policy] validation sweep -> $output_root/val_selection"
    eval_temporal_selector_pressure_8gpu \
        "$checkpoint" val "$output_root/val_selection" "$@"
    if [[ ! -f "$selection" ]]; then
        echo "Validation pressure-policy selection was not created: $selection" >&2
        return 1
    fi

    raw_splits="${TEMPORAL_SELECTOR_PRESSURE_SPLITS:-val,test_seen,test_unseen}"
    local -a splits
    IFS=',' read -r -a splits <<< "$raw_splits"
    for split in "${splits[@]}"; do
        echo "[temporal-selector-policy] fixed validation policies: $split -> $output_root/$split"
        eval_temporal_selector_pressure_8gpu \
            "$checkpoint" "$split" "$output_root/$split" \
            --policy-selection-in "$selection" "$@"
    done
    echo "[temporal-selector-policy] complete: $output_root"
}

eval_temporal_selector_mapping_8gpu() {
    if (( $# < 3 )); then
        echo "Usage: run.sh eval-tflow-selector-mapping-8gpu CHECKPOINT SPLIT OUTPUT_DIR [audit options]" >&2
        return 2
    fi
    local checkpoint="$1" split="$2" output_dir="$3"
    shift 3
    if [[ ! -f "$checkpoint" ]]; then
        echo "Temporal selector checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    if [[ -z "${TACTILE_BASE_CHECKPOINT:-}" ]]; then
        TACTILE_BASE_CHECKPOINT="$(
            "$TACTILE_PYTHON" -c \
                'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu").get("base_checkpoint",""))' \
                "$checkpoint"
        )"
    fi
    ensure_temporal_cache_split "$split"
    local cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    local experiment_dir eval_gpus process_supervisor
    experiment_dir="$(dirname "$(dirname "$checkpoint")")"
    eval_gpus="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
    local -a gpu_array
    IFS=',' read -r -a gpu_array <<< "$eval_gpus"
    process_supervisor="$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py"
    CUDA_VISIBLE_DEVICES="$eval_gpus" \
    "$TACTILE_PYTHON" "$process_supervisor" \
        --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
        --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
        --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
        -- "$TACTILE_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${#gpu_array[@]}" \
        "$SCRIPT_DIR/audit_temporal_selector_mapping.py" \
        --checkpoint "$checkpoint" \
        --cache "$cache" \
        --query-manifests "$manifest" \
        --pair-index-root "$TEMPORAL_PAIR_ROOT" \
        --split "$split" \
        --output-dir "$output_dir" \
        --batch-size "${TEMPORAL_MAPPING_BATCH_SIZE:-512}" \
        --num-workers "${TEMPORAL_MAPPING_WORKERS:-1}" \
        --prefetch-factor "${TEMPORAL_MAPPING_PREFETCH_FACTOR:-1}" \
        --policy-chunk-size "${TEMPORAL_MAPPING_POLICY_CHUNK_SIZE:-4}" \
        --bootstrap-bins "${TEMPORAL_MAPPING_BOOTSTRAP_BINS:-256}" \
        --bootstrap-iterations "${TEMPORAL_MAPPING_BOOTSTRAP_ITERATIONS:-2000}" \
        --bootstrap-confidence "${TEMPORAL_MAPPING_BOOTSTRAP_CONFIDENCE:-0.95}" \
        --bootstrap-seed "${TEMPORAL_MAPPING_BOOTSTRAP_SEED:-521}" \
        --copy-val-metrics-from "$experiment_dir/val_metrics.csv" \
        "$@"
}

audit_tflow_selector_mapping() {
    local checkpoint experiment_dir checkpoint_name output_root raw_splits split
    checkpoint="$TEMPORAL_EXPERIMENT_ROOT/ta_tsel_l12_noq_r256/checkpoints/selector-best.ckpt"
    if (( $# > 0 )) && [[ "$1" != --* ]]; then
        checkpoint="$1"
        shift
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "Temporal selector checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    experiment_dir="$(dirname "$(dirname "$checkpoint")")"
    checkpoint_name="$(basename "$checkpoint" .ckpt)"
    output_root="${TEMPORAL_SELECTOR_MAPPING_OUTPUT_ROOT:-$TEMPORAL_REPORT_ROOT/$(basename "$experiment_dir")/$checkpoint_name/mapping_attribution_v3}"
    raw_splits="${TEMPORAL_SELECTOR_MAPPING_SPLITS:-val,test_seen,test_unseen}"
    local -a splits
    IFS=',' read -r -a splits <<< "$raw_splits"
    for split in "${splits[@]}"; do
        echo "[temporal-selector-mapping] split=$split -> $output_root/$split"
        eval_temporal_selector_mapping_8gpu \
            "$checkpoint" "$split" "$output_root/$split" "$@"
    done
    echo "[temporal-selector-mapping] complete: $output_root"
}

eval_tflow_confirmatory() {
    local checkpoint split output_root raw_splits
    checkpoint="$TEMPORAL_EXPERIMENT_ROOT/ta_tflow_sadd_l124_r256/checkpoints/temporal-best.ckpt"
    if (( $# > 0 )) && [[ "$1" != --* ]]; then
        checkpoint="$1"
        shift
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "Confirmatory temporal checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    raw_splits="${TEMPORAL_CONFIRM_SPLITS:-test_seen,test_unseen}"
    output_root="${TEMPORAL_CONFIRM_OUTPUT_ROOT:-$TEMPORAL_REPORT_ROOT/ta_tflow_sadd_l124_r256/temporal-best/confirmatory_step3}"
    local -a splits confirm_options
    IFS=',' read -r -a splits <<< "$raw_splits"
    confirm_options=(--confirmatory-suite)
    if ! has_cli_option --bootstrap-iterations "$@"; then
        confirm_options+=(
            --bootstrap-iterations "${TEMPORAL_CONFIRM_BOOTSTRAP_ITERATIONS:-2000}"
        )
    fi
    if ! has_cli_option --bootstrap-confidence "$@"; then
        confirm_options+=(
            --bootstrap-confidence "${TEMPORAL_CONFIRM_BOOTSTRAP_CONFIDENCE:-0.95}"
        )
    fi
    if ! has_cli_option --bootstrap-seed "$@"; then
        confirm_options+=(--bootstrap-seed "${TEMPORAL_CONFIRM_BOOTSTRAP_SEED:-521}")
    fi
    for split in "${splits[@]}"; do
        TEMPORAL_EVAL_OUTPUT_DIR="$output_root/$split" \
        eval_temporal_flow_8gpu \
            "$checkpoint" "$split" "${confirm_options[@]}" "$@"
    done
}

eval_temporal_experiment() {
    local experiment_name="$1"
    shift
    local checkpoint_name="${TEMPORAL_EVAL_CHECKPOINT:-best_loss}"
    local checkpoint="$TEMPORAL_EXPERIMENT_ROOT/$experiment_name/checkpoints/$checkpoint_name.ckpt"
    local raw_splits="${TEMPORAL_EVAL_SPLITS:-test_seen,test_unseen}"
    local -a splits
    local split
    IFS=',' read -r -a splits <<< "$raw_splits"
    for split in "${splits[@]}"; do
        eval_temporal_flow_8gpu "$checkpoint" "$split" "$@"
    done
}

audit_temporal_flow_cache() {
    if (( $# < 2 )); then
        echo "Usage: run.sh audit-temporal-flow-cache CHECKPOINT SPLIT [audit options]" >&2
        return 2
    fi
    local checkpoint="$1" split="$2"
    shift 2
    if [[ ! -f "$checkpoint" ]]; then
        echo "Temporal checkpoint does not exist: $checkpoint" >&2
        return 2
    fi
    if [[ -z "${TACTILE_BASE_CHECKPOINT:-}" ]]; then
        TACTILE_BASE_CHECKPOINT="$(
            "$TACTILE_PYTHON" -c \
                'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu").get("base_checkpoint",""))' \
                "$checkpoint"
        )"
    fi
    if [[ ! -f "$TACTILE_BASE_CHECKPOINT" ]]; then
        echo "Temporal RGB base checkpoint does not exist: $TACTILE_BASE_CHECKPOINT" >&2
        return 2
    fi
    ensure_temporal_cache_split "$split"
    local cache="$TEMPORAL_CACHE_RESOLVED_PATH"
    local manifest="$TEMPORAL_MANIFEST_RESOLVED_PATH"
    local experiment_dir output_dir audit_gpus process_supervisor
    experiment_dir="$(dirname "$(dirname "$checkpoint")")"
    output_dir="${TEMPORAL_AUDIT_OUTPUT_DIR:-$TEMPORAL_REPORT_ROOT/$(basename "$experiment_dir")/$(basename "$checkpoint" .ckpt)/cache_audit_$split}"
    audit_gpus="${TEMPORAL_AUDIT_GPUS:-${TEMPORAL_AUDIT_GPU:-0,1,2,3,4,5,6,7}}"
    local -a gpu_array
    IFS=',' read -r -a gpu_array <<< "$audit_gpus"
    if (( ${#gpu_array[@]} == 0 )); then
        echo "TEMPORAL_AUDIT_GPUS must contain at least one GPU index." >&2
        return 2
    fi
    process_supervisor="$(dirname "$SCRIPT_DIR")/hamer_tactile_ft/process_supervisor.py"
    PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}" \
    TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}" \
    CUDA_VISIBLE_DEVICES="$audit_gpus" \
    "$TACTILE_PYTHON" "$process_supervisor" \
        --registry-dir "${PRIOR_PROCESS_REGISTRY:-$INPUT_PRIOR_ROOT/state/run_processes}" \
        --grace-seconds "${PRIOR_PROCESS_GRACE_SECONDS:-60}" \
        --kill-wait-seconds "${PRIOR_PROCESS_KILL_WAIT_SECONDS:-5}" \
        -- "$TACTILE_PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${#gpu_array[@]}" \
        "$SCRIPT_DIR/audit_temporal_flow_cache.py" \
        --checkpoint "$checkpoint" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --cache "$cache" \
        --query-manifests "$manifest" \
        --pair-index-root "$TEMPORAL_PAIR_ROOT" \
        --split "$split" \
        --output-dir "$output_dir" \
        "$@"
}

audit_tflow_long_horizon() {
    local checkpoint split default_checkpoint output_dir
    default_checkpoint="$TEMPORAL_EXPERIMENT_ROOT/ta_tflow_sadd_l124_r256/checkpoints/temporal-best.ckpt"
    checkpoint="$default_checkpoint"
    split="val"
    if (( $# > 0 )) && [[ "$1" != --* ]]; then
        checkpoint="$1"
        shift
    fi
    if (( $# > 0 )) && [[ "$1" != --* ]]; then
        split="$1"
        shift
    fi
    output_dir="$TEMPORAL_REPORT_ROOT/$(basename "$(dirname "$(dirname "$checkpoint")")")/$(basename "$checkpoint" .ckpt)/long_horizon_$split"
    TEMPORAL_AUDIT_OUTPUT_DIR="$output_dir" \
    audit_temporal_flow_cache "$checkpoint" "$split" \
        --lags 1,2,4,8,16,32 \
        --alphas=-0.025,0,0.025,0.05,0.1,0.2,0.4 \
        --evaluation-subset matched_all_lags \
        --model-lag-masks auto \
        --residual-scales 0,0.25,0.5,0.75,1 \
        --model-controls real,cross_sequence,contralateral,reset \
        --gradient-batches 0 \
        "$@"
}

prepare_tactile_history_replay() {
    local run_root manifest split prediction_root
    local -a manifests
    run_root="$(tactile_history_checkpoint_root)"
    mapfile -t manifests < <(tactile_history_manifests)
    if (( ${#manifests[@]} != 3 )); then
        echo "Expected val/test_seen/test_unseen manifests, found ${#manifests[@]}." >&2
        return 2
    fi
    mkdir -p "$run_root/predictions"
    for manifest in "${manifests[@]}"; do
        split="$(tactile_history_split "$manifest")"
        prediction_root="$(tactile_history_prediction_root "$run_root" "$manifest")"
        if [[ -f "$prediction_root/_COMPLETE" && -f "$prediction_root/prediction_config.json" ]]; then
            echo "[tactile-history] reuse split=$split: $prediction_root"
            continue
        fi
        echo "[tactile-history] exporting split=$split -> $prediction_root"
        "$TACTILE_PYTHON" "$SCRIPT_DIR/../hamer_tactile_ft/eval_tactile_fast.py" \
            --checkpoint "$TACTILE_BASE_CHECKPOINT" \
            --dino_weights "$DINO_WEIGHTS" \
            --datasets touchanything \
            --split "$split" \
            --gpus "$TACTILE_HISTORY_GPUS" \
            --batch_size "$TACTILE_HISTORY_EVAL_BATCH_SIZE" \
            --num_workers "$TACTILE_HISTORY_EVAL_WORKERS" \
            --data_backend sequence_hdf5 \
            --query_manifests "$manifest" \
            --bbox_manifests "$TACTILE_HISTORY_BBOX_MANIFEST" \
            --prediction_output_dir "$prediction_root" \
            --prediction_palm_only \
            --report_dir "$prediction_root/eval_report" \
            --report_name "eval_touchanything_${split}.txt" \
            --no-rebuild_index \
            "$@"
    done
    printf '[tactile-history] prediction root: %s\n' "$run_root/predictions"
}

audit_tactile_history_replay() {
    local run_root manifest prediction_root audit_material audit_key audit_root
    local -a manifests prediction_args
    run_root="$(tactile_history_checkpoint_root)"
    mapfile -t manifests < <(tactile_history_manifests)
    if (( ${#manifests[@]} != 3 )); then
        echo "Expected val/test_seen/test_unseen manifests, found ${#manifests[@]}." >&2
        return 2
    fi
    audit_material=""
    prediction_args=()
    for manifest in "${manifests[@]}"; do
        prediction_root="$(tactile_history_prediction_root "$run_root" "$manifest")"
        if [[ ! -f "$prediction_root/_COMPLETE" || ! -f "$prediction_root/prediction_config.json" ]]; then
            echo "Missing prediction export: $prediction_root" >&2
            echo "Run prepare-tactile-history-replay first." >&2
            return 2
        fi
        prediction_args+=(--prediction-root "$prediction_root")
        audit_material+="$(sha256sum "$manifest")"
    done
    audit_material+="$(sha256sum "$TACTILE_HISTORY_BBOX_MANIFEST")"
    audit_material+="$(sha256sum "$SCRIPT_DIR/audit_predicted_tactile_history.py")"
    audit_key="$(printf '%s' "$audit_material" | sha256sum | cut -d' ' -f1)"
    audit_root="$run_root/audits/${audit_key:0:16}"
    if [[ "${FORCE_TACTILE_HISTORY_AUDIT:-0}" != "1" && -f "$audit_root/summary.json" ]]; then
        echo "[tactile-history] reuse completed audit: $audit_root"
        return 0
    fi
    echo "[tactile-history] replay audit -> $audit_root"
    "$TACTILE_PYTHON" "$SCRIPT_DIR/audit_predicted_tactile_history.py" \
        "${prediction_args[@]}" \
        --output-dir "$audit_root" \
        --metric-device "$TACTILE_HISTORY_METRIC_DEVICE" \
        --pair-batch-size "$TACTILE_HISTORY_PAIR_BATCH_SIZE" \
        --bootstrap-iterations "$TACTILE_HISTORY_BOOTSTRAP_ITERATIONS" \
        --bootstrap-confidence "$TACTILE_HISTORY_BOOTSTRAP_CONFIDENCE" \
        "$@"
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
    audit-depth-crop-coverage)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/audit_depth_crop_coverage.py" "$@"
        ;;
    audit-tactile-dynamics)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/../hamer_tactile_ft/audit_tactile_dynamics.py" "$@"
        ;;
    audit-bilateral-tactile-dynamics)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/../hamer_tactile_ft/audit_tactile_dynamics.py" \
            --mode bilateral "$@"
        ;;
    audit-bilateral-tactile-dynamics-fast)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/../hamer_tactile_ft/audit_tactile_dynamics.py" \
            --mode bilateral \
            --max-bilateral-pressure-pairs 0 \
            --pressure-metric-device cuda \
            --pressure-batch-size "${BILATERAL_PRESSURE_BATCH_SIZE:-8192}" \
            --pressure-metric-chunk-size "${BILATERAL_PRESSURE_METRIC_CHUNK_SIZE:-32768}" \
            --pressure-metric-dtype float32 \
            --pair-csv-limit 0 \
            "$@"
        ;;
    prepare-tactile-history-replay)
        prepare_tactile_history_replay "$@"
        ;;
    audit-tactile-history-replay)
        audit_tactile_history_replay "$@"
        ;;
    pipeline-tactile-history-replay)
        if (( $# != 0 )); then
            echo "pipeline-tactile-history-replay accepts tuning through TACTILE_HISTORY_* environment variables." >&2
            exit 2
        fi
        prepare_tactile_history_replay
        audit_tactile_history_replay
        ;;
    prepare-temporal-flow)
        prepare_temporal_flow "$@"
        ;;
    train-temporal-flow)
        train_temporal_flow "$@"
        ;;
    train-tflow-signed-l1)
        train_temporal_flow_signed_l1 "$@"
        ;;
    train-tflow-signed-l124)
        train_temporal_flow_signed_l124 "$@"
        ;;
    train-tflow-selector-quality)
        train_tflow_selector_quality "$@"
        ;;
    train-tflow-selector-noquality)
        train_tflow_selector_noquality "$@"
        ;;
    train-tflow-selector-dino-aligned)
        train_tflow_selector_dino_aligned "$@"
        ;;
    train-tflow-selector-dino-unwarped)
        train_tflow_selector_dino_unwarped "$@"
        ;;
    eval-temporal-flow-8gpu)
        eval_temporal_flow_8gpu "$@"
        ;;
    eval-tflow-signed-l1)
        eval_temporal_experiment ta_tflow_sadd_l1_r256 "$@"
        ;;
    eval-tflow-signed-l124)
        eval_temporal_experiment ta_tflow_sadd_l124_r256 "$@"
        ;;
    eval-tflow-confirmatory)
        eval_tflow_confirmatory "$@"
        ;;
    eval-tflow-selector-8gpu)
        eval_temporal_selector_8gpu "$@"
        ;;
    eval-tflow-selector-quality)
        eval_temporal_selector_experiment ta_tsel_l12_q_r256 "$@"
        ;;
    eval-tflow-selector-noquality)
        eval_temporal_selector_experiment ta_tsel_l12_noq_r256 "$@"
        ;;
    eval-tflow-selector-dino-aligned)
        eval_tflow_selector_dino_experiment ta_tsel_dino_align_r256 "$@"
        ;;
    eval-tflow-selector-dino-unwarped)
        eval_tflow_selector_dino_experiment ta_tsel_dino_unwarp_r256 "$@"
        ;;
    eval-tflow-selector-dino-aligned-selector-best)
        eval_tflow_selector_dino_selector_best ta_tsel_dino_align_r256 "$@"
        ;;
    eval-tflow-selector-dino-unwarped-selector-best)
        eval_tflow_selector_dino_selector_best ta_tsel_dino_unwarp_r256 "$@"
        ;;
    eval-tflow-selector-pressure-8gpu)
        eval_temporal_selector_pressure_8gpu "$@"
        ;;
    audit-tflow-selector-pressure)
        audit_tflow_selector_pressure "$@"
        ;;
    eval-tflow-selector-mapping-8gpu)
        eval_temporal_selector_mapping_8gpu "$@"
        ;;
    audit-tflow-selector-mapping)
        audit_tflow_selector_mapping "$@"
        ;;
    audit-temporal-flow-cache)
        audit_temporal_flow_cache "$@"
        ;;
    audit-tflow-long-horizon)
        audit_tflow_long_horizon "$@"
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
    prepare-selector-cache)
        prepare_selector_caches
        ;;
    selector-cache-paths)
        print_selector_cache_paths
        ;;
    train-depth-real)
        run_prior_train depth_spatial 8 real ta_depth_feature_real "$@"
        ;;
    train-depth-shuffle)
        run_prior_train depth_spatial 8 spatial_shuffle ta_depth_feature_shuffle "$@"
        ;;
    train-depth-film-cf)
        run_prior_train depth_causal_film 8 real ta_dfilm_cf_r256 \
            --feature-rms-budget 0.02 \
            --logit-delta-max 0.25 \
            --prior-dropout 0 \
            --counterfactual-control spatial_shuffle \
            --control-identity-weight 0.01 \
            --feature-budget-penalty-weight 0.005 \
            --zero-mean-logit-residual \
            "$@"
        ;;
    train-depth-xattn-cf)
        run_prior_train depth_local_xattn 8 real ta_dxattn_cf_r256 \
            --feature-rms-budget 0.02 \
            --logit-delta-max 0.25 \
            --prior-dropout 0 \
            --counterfactual-control spatial_shuffle \
            --control-identity-weight 0.01 \
            --feature-budget-penalty-weight 0.005 \
            --zero-mean-logit-residual \
            --depth-attention-heads 4 \
            --depth-attention-window 5 \
            "$@"
        ;;
    train-vlm-real)
        require_environment VLM_PRIOR_DIM
        run_prior_train vlm_lowrank "$VLM_PRIOR_DIM" real ta_vlm_feature_real "$@"
        ;;
    train-vlm-control)
        require_environment VLM_PRIOR_DIM
        run_prior_train vlm_lowrank "$VLM_PRIOR_DIM" context_shuffle ta_vlm_feature_shuffle "$@"
        ;;
    train-selector-depth-map)
        run_prior_selector_train \
            depth_mapping_rectifier 8 spatial_shuffle ta_dsel_map_r256 "$@"
        ;;
    train-selector-depth-anchor)
        run_prior_selector_train \
            depth_anchor_residual 8 spatial_shuffle ta_dsel_anchor_r256 "$@"
        ;;
    train-selector-depth-query)
        run_prior_selector_train \
            depth_anchor_query 8 spatial_shuffle ta_dquery_real_r256 \
            --false-high-loss-weight 1.0 "$@"
        ;;
    train-selector-depth-query-shuffle)
        run_prior_selector_train \
            depth_anchor_query 8 real ta_dquery_shuffle_r256 \
            --prior-control spatial_shuffle \
            --control-identity-weight 0 \
            --false-high-loss-weight 1.0 "$@"
        ;;
    train-selector-depth-query-clean-real)
        run_prior_selector_train \
            depth_anchor_query 8 spatial_shuffle ta_dquery_clean_real_r256 \
            --control-identity-weight 0 \
            --false-high-loss-weight 1.0 \
            --epochs 10 \
            --warmup-epochs 1 "$@"
        ;;
    train-selector-depth-query-clean-shuffle)
        run_prior_selector_train \
            depth_anchor_query 8 real ta_dquery_clean_shuffle_r256 \
            --prior-control spatial_shuffle \
            --control-identity-weight 0 \
            --false-high-loss-weight 1.0 \
            --epochs 10 \
            --warmup-epochs 1 "$@"
        ;;
    train-selector-depth-query-contact-real)
        run_prior_selector_train \
            depth_anchor_query 8 spatial_shuffle ta_dquery_contact_real_r256 \
            --control-identity-weight 0 \
            --false-high-loss-weight 0 \
            --false-high-score-source contact \
            --no-paired-controls \
            --no-validation-reference-metrics \
            --no-validation-sequence-metrics \
            --batch-size 512 \
            --val-batch-size 512 \
            --epochs 10 \
            --warmup-epochs 1 "$@"
        ;;
    train-selector-depth-query-contact-shuffle)
        run_prior_selector_train \
            depth_anchor_query 8 real ta_dquery_contact_shuffle_r256 \
            --prior-control spatial_shuffle \
            --control-identity-weight 0 \
            --false-high-loss-weight 0 \
            --false-high-score-source contact \
            --no-paired-controls \
            --no-validation-reference-metrics \
            --no-validation-sequence-metrics \
            --batch-size 512 \
            --val-batch-size 512 \
            --epochs 10 \
            --warmup-epochs 1 "$@"
        ;;
    train-selector-vlm)
        require_environment VLM_PRIOR_DIM
        run_prior_selector_train \
            vlm_global_calibrator "$VLM_PRIOR_DIM" context_shuffle ta_vsel_cal_r256 "$@"
        ;;
    train-selector-vlm-siglip)
        require_environment VLM_PRIOR_DIM
        run_prior_selector_train \
            vlm_global_calibrator "$VLM_PRIOR_DIM" context_shuffle ta_vsel_siglip_r256 "$@"
        ;;
    train-selector-vlm-qwen)
        require_environment VLM_PRIOR_DIM
        run_prior_selector_train \
            vlm_global_calibrator "$VLM_PRIOR_DIM" context_shuffle ta_vsel_qwen_r256 "$@"
        ;;
    train-prior-selector)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/train_prior_selector.py" "$@"
        ;;
    eval-prior-selector)
        run_prior_selector_eval 0 "$@"
        ;;
    eval-prior-selector-8gpu)
        run_prior_selector_eval 1 "$@"
        ;;
    audit-selector-pressure-8gpu)
        run_selector_pressure_audit "$@"
        ;;
    audit-selector-matched-pareto-8gpu)
        run_selector_matched_pareto_audit "$@"
        ;;
    audit-selector-exact-topk-8gpu)
        run_selector_exact_topk_audit "$@"
        ;;
    selector-pressure-tiny-check)
        "$TACTILE_PYTHON" "$SCRIPT_DIR/eval_prior_selector.py" \
            --checkpoint ignored --split val --output-dir /tmp/selector-pressure-tiny \
            --tiny-check
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
    eval-depth-controls-8gpu)
        run_depth_control_audit "$@"
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
