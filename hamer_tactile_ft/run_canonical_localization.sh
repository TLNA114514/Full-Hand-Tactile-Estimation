#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    cat <<'EOF'
Usage: run_canonical_localization.sh MODE [audit options]

Modes:
  prepare    Build/reuse the persistent crop1.2 validation feature cache
  stage0     Compatibility alias for the current canonical diagnosis
  stage1     Run the stratified sensor-independent surface-basis audit
  cleanup    Run Stage 0.3 basis identifiability/trainability cleanup
  capacity   Run Stage 0.4 standalone 1536..6144 coefficient sweep
  density    Run Stage 0.4b adaptive-overlap high-density sweep on 3 GPUs
  learnability-prepare
             Build/reuse shared FullGrid/ridge-teacher probe arrays
  learnability
             Run the 8-GPU decoder/loss/memorization diagnosis matrix
  spatial-dependency
             Audit how the frozen FullGrid decoder depends on token placement
  attribution-prepare
             Build official split caches and Stage 0.7 basis oracles
  attribution-train
             Train parameter-matched basis/direct heads from prepared splits
  attribution
             Run the complete Stage 0.7 attribution audit
  routing-geometry
             Precompute 256/512-anchor geodesic ownership once
  routing-train
             Train the eight Stage 2 routing/control probes on 8 GPUs
  routing-eval
             Evaluate current loss-best routing checkpoints and aggregate
  routing
             Run routing geometry, training, evaluation, and aggregation
  routing-v2-prepare
             Align cached 256-channel ReZero grids to Stage 0.7 samples
  routing-v2-train
             Train eight strict evidence-only Stage 2.1 probes on 8 GPUs
  routing-v2-eval
             Evaluate current Stage 2.1 loss-best checkpoints and aggregate
  routing-v2
             Prepare, train, evaluate, and aggregate Stage 2.1
  all        Prepare the cache if needed, then run Stage 1
  self-test  Run deterministic canonical, basis, and learnability checks

Stages 0.1-0.4 never train on oracle values. Stage 0.5 uses ridge coefficients
only as a diagnostic auxiliary target; they are never model inputs or formal
evaluation labels.

Optional environment:
  TACTILE_BASE_CHECKPOINT     FullGrid32 crop1.2 loss-best checkpoint
  TACTILE_FEATURE_CACHE       Existing persistent feature cache
  CANONICAL_LOCALIZATION_DIR  Output directory (default: stage1_surface_basis)
  LOCALIZATION_DEVICE         Audit device (default: cuda:0)
  LOCALIZATION_SAMPLE_LIMIT   Main audit sample count (default: 50000)
  LOCALIZATION_BATCH_SIZE     Frozen decoder batch size (default: 256)
  COMPONENT_SAMPLE_LIMIT      Topology audit sample count (default: 12000)
  AMBIGUITY_SAMPLE_LIMIT      Matched-mass ambiguity sample count (default: 12000)
  BASIS_SAMPLE_LIMIT          Surface-basis reconstruction sample count (default: 2048)
  BASIS_BATCH_SIZE            Surface-basis ridge batch size (default: 64)
  BASIS_STAGE1_DIR            Completed Stage 0.2 directory (auto-detected)
  BASIS_CLEANUP_DIR           Stage 0.3 output directory
  BASIS_CAPACITY_DIR          Stage 0.4 capacity-sweep output directory
  BASIS_CAPACITY_COUNTS       Stage 0.4 dimensions (default: 1536..6144)
  BASIS_DENSITY_DIR           Stage 0.4b aggregate/output directory
  BASIS_DENSITY_COUNTS        Stage 0.4b dimensions (default: 3072..6144)
  BASIS_DENSITY_SUPPORTS      Target support counts (default: 4,6,8)
  BASIS_DENSITY_GPUS          One GPU per support config (default: 0,1,2)
  BASIS_CLEANUP_BATCH_SIZE    Stage 0.3 GPU batch size (default: 64)
  BASIS_BOOTSTRAP_REPEATS     Stage 0.3 bootstrap repeats (default: 500)
  BASIS_NNLS_ITERATIONS       Positive-basis iterations (default: 100)
  SURFACE_BASIS_RUNTIME       K4096 support-4 runtime basis artifact
  SURFACE_LEARNABILITY_DIR    Layered diagnosis output directory
  LEARNABILITY_GPUS           Eight comma-separated GPUs (default: 0..7)
  LEARNABILITY_SAMPLE_LIMIT   Shared sequence-split sample count (default: 32768)
  LEARNABILITY_EPOCHS         Generalization probe epochs (default: 30)
  LEARNABILITY_MEM_EPOCHS     1K memorization probe epochs (default: 160)
  FULLGRID_SPATIAL_DIR        Spatial-dependency audit output directory
  FULLGRID_SPATIAL_SAMPLES    Spatial audit sample count (default: 32768)
  FULLGRID_SPATIAL_BATCH_SIZE Spatial audit decoder batch size (default: 256)
  FULLGRID_SPATIAL_VARIANTS   Comma-separated perturbation controls
  SURFACE_ATTRIBUTION_CACHE_ROOT  Persistent official-split cache root
  SURFACE_ATTRIBUTION_DIR     Stage 0.7 output directory
  ATTRIBUTION_CACHE_GPUS      Cache builder GPUs (default: 0..7)
  ATTRIBUTION_PREP_GPUS       Four oracle preparation GPUs (default: 0,1,2,3)
  ATTRIBUTION_TRAIN_GPUS      Basis/direct training GPUs (default: 0,1)
  ATTRIBUTION_TRAIN_SAMPLES   Stable train subset size (default: 131072)
  ATTRIBUTION_EVAL_SAMPLES    Stable samples per eval split (default: 32768)
  ATTRIBUTION_ORACLE_SAMPLES  Basis-oracle rows per split (default: 8192)
  CANONICAL_ROUTING_DIR       Stage 2 routing output directory
  ROUTING_GPUS                Eight comma-separated GPUs (default: 0..7)
  ROUTING_ANCHOR_COUNTS       Geometry anchors (default: 256,512)
  ROUTING_MODES               competitive,independent controls
  ROUTING_SOURCES             spatial,global_control controls
  ROUTING_DIM                 Per-anchor hidden size (default: 128)
  ROUTING_BATCH_SIZE          Cached-feature train batch (default: 512)
  ROUTING_EVAL_BATCH_SIZE     Cached-feature eval batch (default: 1024)
  CANONICAL_ROUTING_V2_DIR    Strict Stage 2.1 output directory
  ROUTING_V2_GPUS             Eight comma-separated GPUs (default: 0..7)
  ROUTING_V2_FEATURE_SOURCES  projected32,rezero256 comparison
  ROUTING_V2_SEEDS            Four controlled seeds (default: 521,2029,3407,4099)
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
CANONICAL_LOCALIZATION_DIR="${CANONICAL_LOCALIZATION_DIR:-$ROOT_DIR/hamer_tactile_ft/canonical_localization_audits/stage1_surface_basis}"
LOCALIZATION_DEVICE="${LOCALIZATION_DEVICE:-cuda:0}"
LOCALIZATION_SAMPLE_LIMIT="${LOCALIZATION_SAMPLE_LIMIT:-50000}"
LOCALIZATION_BATCH_SIZE="${LOCALIZATION_BATCH_SIZE:-256}"
COMPONENT_SAMPLE_LIMIT="${COMPONENT_SAMPLE_LIMIT:-12000}"
AMBIGUITY_SAMPLE_LIMIT="${AMBIGUITY_SAMPLE_LIMIT:-12000}"
BASIS_SAMPLE_LIMIT="${BASIS_SAMPLE_LIMIT:-2048}"
BASIS_BATCH_SIZE="${BASIS_BATCH_SIZE:-64}"
BASIS_STAGE1_DIR="${BASIS_STAGE1_DIR:-}"
BASIS_CLEANUP_DIR="${BASIS_CLEANUP_DIR:-$ROOT_DIR/hamer_tactile_ft/canonical_localization_audits/stage0_3_basis_cleanup}"
BASIS_CAPACITY_DIR="${BASIS_CAPACITY_DIR:-$ROOT_DIR/hamer_tactile_ft/stage0_4_basis_capacity}"
BASIS_CAPACITY_COUNTS="${BASIS_CAPACITY_COUNTS:-1536,2048,3072,4096,5120,6144}"
BASIS_DENSITY_DIR="${BASIS_DENSITY_DIR:-$ROOT_DIR/hamer_tactile_ft/stage0_4b_basis_density}"
BASIS_DENSITY_COUNTS="${BASIS_DENSITY_COUNTS:-3072,4096,5120,6144}"
BASIS_DENSITY_SUPPORTS="${BASIS_DENSITY_SUPPORTS:-4,6,8}"
BASIS_DENSITY_GPUS="${BASIS_DENSITY_GPUS:-0,1,2}"
BASIS_CLEANUP_BATCH_SIZE="${BASIS_CLEANUP_BATCH_SIZE:-64}"
BASIS_BOOTSTRAP_REPEATS="${BASIS_BOOTSTRAP_REPEATS:-500}"
BASIS_NNLS_ITERATIONS="${BASIS_NNLS_ITERATIONS:-100}"
SURFACE_BASIS_RUNTIME_ROOT="${SURFACE_BASIS_RUNTIME_ROOT:-$DEFAULT_RUNTIME_ROOT/surface_basis/stage0_4b_s4}"
SURFACE_BASIS_RUNTIME="${SURFACE_BASIS_RUNTIME:-$SURFACE_BASIS_RUNTIME_ROOT/runtime_basis_k4096_s4.pt}"
SURFACE_LEARNABILITY_DIR="${SURFACE_LEARNABILITY_DIR:-$ROOT_DIR/hamer_tactile_ft/surface_decoder_learnability}"
LEARNABILITY_GPUS="${LEARNABILITY_GPUS:-0,1,2,3,4,5,6,7}"
LEARNABILITY_SAMPLE_LIMIT="${LEARNABILITY_SAMPLE_LIMIT:-32768}"
LEARNABILITY_PREP_BATCH_SIZE="${LEARNABILITY_PREP_BATCH_SIZE:-128}"
LEARNABILITY_BATCH_SIZE="${LEARNABILITY_BATCH_SIZE:-256}"
LEARNABILITY_EVAL_BATCH_SIZE="${LEARNABILITY_EVAL_BATCH_SIZE:-512}"
LEARNABILITY_EPOCHS="${LEARNABILITY_EPOCHS:-30}"
LEARNABILITY_MEM_EPOCHS="${LEARNABILITY_MEM_EPOCHS:-160}"
LEARNABILITY_MEM_SAMPLES="${LEARNABILITY_MEM_SAMPLES:-1024}"
FULLGRID_SPATIAL_DIR="${FULLGRID_SPATIAL_DIR:-$ROOT_DIR/hamer_tactile_ft/fullgrid_spatial_dependency}"
FULLGRID_SPATIAL_SAMPLES="${FULLGRID_SPATIAL_SAMPLES:-32768}"
FULLGRID_SPATIAL_BATCH_SIZE="${FULLGRID_SPATIAL_BATCH_SIZE:-256}"
FULLGRID_SPATIAL_VARIANTS="${FULLGRID_SPATIAL_VARIANTS:-identity,global_mean,spatial_shuffle,block_shuffle,cyclic_shift}"
SURFACE_ATTRIBUTION_CACHE_ROOT="${SURFACE_ATTRIBUTION_CACHE_ROOT:-$DEFAULT_RUNTIME_ROOT/cache/surface_mapping_attribution}"
SURFACE_ATTRIBUTION_DIR="${SURFACE_ATTRIBUTION_DIR:-$ROOT_DIR/hamer_tactile_ft/surface_mapping_attribution}"
ATTRIBUTION_CACHE_GPUS="${ATTRIBUTION_CACHE_GPUS:-0,1,2,3,4,5,6,7}"
ATTRIBUTION_PREP_GPUS="${ATTRIBUTION_PREP_GPUS:-0,1,2,3}"
ATTRIBUTION_TRAIN_GPUS="${ATTRIBUTION_TRAIN_GPUS:-0,1}"
ATTRIBUTION_TRAIN_SAMPLES="${ATTRIBUTION_TRAIN_SAMPLES:-131072}"
ATTRIBUTION_EVAL_SAMPLES="${ATTRIBUTION_EVAL_SAMPLES:-32768}"
ATTRIBUTION_ORACLE_SAMPLES="${ATTRIBUTION_ORACLE_SAMPLES:-8192}"
ATTRIBUTION_TRAIN_PER_SEQUENCE="${ATTRIBUTION_TRAIN_PER_SEQUENCE:-64}"
ATTRIBUTION_EVAL_PER_SEQUENCE="${ATTRIBUTION_EVAL_PER_SEQUENCE:-384}"
ATTRIBUTION_CACHE_BATCH_SIZE="${ATTRIBUTION_CACHE_BATCH_SIZE:-128}"
ATTRIBUTION_PREP_BATCH_SIZE="${ATTRIBUTION_PREP_BATCH_SIZE:-128}"
ATTRIBUTION_BATCH_SIZE="${ATTRIBUTION_BATCH_SIZE:-1024}"
ATTRIBUTION_EVAL_BATCH_SIZE="${ATTRIBUTION_EVAL_BATCH_SIZE:-2048}"
ATTRIBUTION_EPOCHS="${ATTRIBUTION_EPOCHS:-30}"
CANONICAL_ROUTING_DIR="${CANONICAL_ROUTING_DIR:-$ROOT_DIR/hamer_tactile_ft/canonical_anchor_routing}"
ROUTING_GPUS="${ROUTING_GPUS:-0,1,2,3,4,5,6,7}"
ROUTING_ANCHOR_COUNTS="${ROUTING_ANCHOR_COUNTS:-256,512}"
ROUTING_MODES="${ROUTING_MODES:-competitive,independent}"
ROUTING_SOURCES="${ROUTING_SOURCES:-spatial,global_control}"
ROUTING_DIM="${ROUTING_DIM:-128}"
ROUTING_HEADS="${ROUTING_HEADS:-4}"
ROUTING_LAYERS="${ROUTING_LAYERS:-2}"
ROUTING_DROPOUT="${ROUTING_DROPOUT:-0.1}"
ROUTING_MAX_LOGIT_DELTA="${ROUTING_MAX_LOGIT_DELTA:-2.0}"
ROUTING_EPOCHS="${ROUTING_EPOCHS:-30}"
ROUTING_BATCH_SIZE="${ROUTING_BATCH_SIZE:-512}"
ROUTING_EVAL_BATCH_SIZE="${ROUTING_EVAL_BATCH_SIZE:-1024}"
CANONICAL_ROUTING_V2_DIR="${CANONICAL_ROUTING_V2_DIR:-$ROOT_DIR/hamer_tactile_ft/canonical_anchor_routing_v2}"
ROUTING_V2_GPUS="${ROUTING_V2_GPUS:-0,1,2,3,4,5,6,7}"
ROUTING_V2_FEATURE_SOURCES="${ROUTING_V2_FEATURE_SOURCES:-projected32,rezero256}"
ROUTING_V2_SEEDS="${ROUTING_V2_SEEDS:-521,2029,3407,4099}"
ROUTING_V2_ANCHOR_COUNT="${ROUTING_V2_ANCHOR_COUNT:-256}"
ROUTING_V2_DIM="${ROUTING_V2_DIM:-128}"
ROUTING_V2_HEADS="${ROUTING_V2_HEADS:-4}"
ROUTING_V2_LAYERS="${ROUTING_V2_LAYERS:-2}"
ROUTING_V2_DROPOUT="${ROUTING_V2_DROPOUT:-0.1}"
ROUTING_V2_MAX_LOGIT_DELTA="${ROUTING_V2_MAX_LOGIT_DELTA:-2.0}"
ROUTING_V2_EPOCHS="${ROUTING_V2_EPOCHS:-30}"
ROUTING_V2_BATCH_SIZE="${ROUTING_V2_BATCH_SIZE:-512}"
ROUTING_V2_EVAL_BATCH_SIZE="${ROUTING_V2_EVAL_BATCH_SIZE:-1024}"
ROUTING_V2_PREP_BATCH_SIZE="${ROUTING_V2_PREP_BATCH_SIZE:-256}"
TACTILE_PYTHON="${TACTILE_PYTHON:-/home/ma-user/work/cfzhao/tactile/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$DEFAULT_RUNTIME_ROOT/state/matplotlib}"

if [[ ! -x "$TACTILE_PYTHON" ]]; then
    TACTILE_PYTHON=python
fi
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

cache_ready() {
    if [[ -f "$TACTILE_FEATURE_CACHE/CACHE_DONE.json" ]]; then
        return 0
    fi
    local parts=() path first_name expected_text expected
    shopt -s nullglob
    parts=("$TACTILE_FEATURE_CACHE"/part-*-of-*)
    shopt -u nullglob
    (( ${#parts[@]} > 0 )) || return 1
    first_name="$(basename "${parts[0]}")"
    expected_text="${first_name##*-of-}"
    [[ "$expected_text" =~ ^[0-9]+$ ]] || return 1
    expected=$((10#$expected_text))
    (( ${#parts[@]} == expected )) || return 1
    for path in "${parts[@]}"; do
        [[ -f "$path/CACHE_DONE.json" ]] || return 1
    done
}

cache_path_ready() {
    local root="$1" parts=() path first_name expected_text expected
    if [[ -f "$root/CACHE_DONE.json" ]]; then
        return 0
    fi
    shopt -s nullglob
    parts=("$root"/part-*-of-*)
    shopt -u nullglob
    (( ${#parts[@]} > 0 )) || return 1
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
    TACTILE_BASE_CHECKPOINT="$TACTILE_BASE_CHECKPOINT" \
    TACTILE_FEATURE_CACHE="$TACTILE_FEATURE_CACHE" \
    TACTILE_PYTHON="$TACTILE_PYTHON" \
    "$ROOT_DIR/hamer_tactile_ft/run_local_controllability.sh" prepare
}

run_stage0() {
    cache_ready || {
        echo "Feature cache is not ready. Run this first:" >&2
        echo "  $0 prepare" >&2
        exit 2
    }
    [[ -f "$TACTILE_BASE_CHECKPOINT" ]] || {
        echo "Missing base checkpoint: $TACTILE_BASE_CHECKPOINT" >&2
        exit 2
    }
    mkdir -p "$CANONICAL_LOCALIZATION_DIR"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_canonical_localization.py" \
        --feature-cache "$TACTILE_FEATURE_CACHE" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --output-dir "$CANONICAL_LOCALIZATION_DIR" \
        --device "$LOCALIZATION_DEVICE" \
        --sample-limit "$LOCALIZATION_SAMPLE_LIMIT" \
        --batch-size "$LOCALIZATION_BATCH_SIZE" \
        --component-sample-limit "$COMPONENT_SAMPLE_LIMIT" \
        --ambiguity-sample-limit "$AMBIGUITY_SAMPLE_LIMIT" \
        --basis-sample-limit "$BASIS_SAMPLE_LIMIT" \
        --basis-batch-size "$BASIS_BATCH_SIZE" \
        "$@"
}

resolve_stage1_dir() {
    local candidate
    if [[ -n "$BASIS_STAGE1_DIR" ]]; then
        candidate="$BASIS_STAGE1_DIR"
        [[ -f "$candidate/summary.json" && -f "$candidate/subaudit_samples.csv" ]] || {
            echo "BASIS_STAGE1_DIR is incomplete: $candidate" >&2
            return 1
        }
        printf '%s\n' "$candidate"
        return 0
    fi
    for candidate in \
        "$ROOT_DIR/hamer_tactile_ft/stage1_surface_basis" \
        "$ROOT_DIR/hamer_tactile_ft/reports/stage1_surface_basis" \
        "$CANONICAL_LOCALIZATION_DIR" \
        "$ROOT_DIR/hamer_tactile_ft/canonical_localization_audits/stage1_surface_basis"
    do
        if [[ -f "$candidate/summary.json" && -f "$candidate/subaudit_samples.csv" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "Could not auto-detect a completed Stage 0.2 directory." >&2
    echo "Set BASIS_STAGE1_DIR=/path/to/stage1_surface_basis." >&2
    return 1
}

run_cleanup() {
    local stage1_dir
    stage1_dir="$(resolve_stage1_dir)"
    mkdir -p "$BASIS_CLEANUP_DIR"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_surface_basis_cleanup.py" \
        --stage1-dir "$stage1_dir" \
        --output-dir "$BASIS_CLEANUP_DIR" \
        --feature-cache "$TACTILE_FEATURE_CACHE" \
        --device "$LOCALIZATION_DEVICE" \
        --batch-size "$BASIS_CLEANUP_BATCH_SIZE" \
        --bootstrap-repeats "$BASIS_BOOTSTRAP_REPEATS" \
        --nnls-iterations "$BASIS_NNLS_ITERATIONS" \
        "$@"
}

run_capacity() {
    local stage1_dir
    stage1_dir="$(resolve_stage1_dir)"
    mkdir -p "$BASIS_CAPACITY_DIR"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_surface_basis_cleanup.py" \
        --stage1-dir "$stage1_dir" \
        --output-dir "$BASIS_CAPACITY_DIR" \
        --feature-cache "$TACTILE_FEATURE_CACHE" \
        --device "$LOCALIZATION_DEVICE" \
        --batch-size "$BASIS_CLEANUP_BATCH_SIZE" \
        --bootstrap-repeats "$BASIS_BOOTSTRAP_REPEATS" \
        --anchor-counts "$BASIS_CAPACITY_COUNTS" \
        --standalone-counts "$BASIS_CAPACITY_COUNTS" \
        --skip-multiscale-controls \
        --skip-nnls \
        "$@"
}

run_density() {
    local stage1_dir failed index support gpu child_dir log_path
    local -a supports gpus pids labels
    stage1_dir="$(resolve_stage1_dir)"
    IFS=',' read -r -a supports <<< "$BASIS_DENSITY_SUPPORTS"
    IFS=',' read -r -a gpus <<< "$BASIS_DENSITY_GPUS"
    (( ${#supports[@]} > 0 )) || {
        echo "BASIS_DENSITY_SUPPORTS must contain at least one value" >&2
        return 2
    }
    (( ${#gpus[@]} > 0 )) || {
        echo "BASIS_DENSITY_GPUS must contain at least one GPU" >&2
        return 2
    }
    mkdir -p "$BASIS_DENSITY_DIR"
    pids=()
    labels=()

    density_interrupt() {
        local pid
        trap - INT TERM
        echo "Stopping Stage 0.4b workers; completed partials remain resumable." >&2
        for pid in "${pids[@]}"; do
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap density_interrupt INT TERM

    for index in "${!supports[@]}"; do
        support="${supports[$index]//[[:space:]]/}"
        [[ "$support" =~ ^[1-9][0-9]*$ ]] || {
            echo "Invalid target support count: ${supports[$index]}" >&2
            density_interrupt
        }
        gpu="${gpus[$((index % ${#gpus[@]}))]//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ ]] || {
            echo "Invalid GPU id: $gpu" >&2
            density_interrupt
        }
        child_dir="$BASIS_DENSITY_DIR/support_$support"
        log_path="$BASIS_DENSITY_DIR/support_$support.log"
        mkdir -p "$child_dir"
        echo "[stage0.4b] support=$support gpu=$gpu log=$log_path"
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_surface_basis_cleanup.py" \
                --stage1-dir "$stage1_dir" \
                --output-dir "$child_dir" \
                --feature-cache "$TACTILE_FEATURE_CACHE" \
                --device cuda:0 \
                --batch-size "$BASIS_CLEANUP_BATCH_SIZE" \
                --bootstrap-repeats "$BASIS_BOOTSTRAP_REPEATS" \
                --anchor-counts "$BASIS_DENSITY_COUNTS" \
                --standalone-counts "$BASIS_DENSITY_COUNTS" \
                --basis-bandwidth-policy target_overlap \
                --basis-target-support-count "$support" \
                --result-source stage0_4b_density_cleanup \
                --skip-multiscale-controls \
                --skip-nnls \
                "$@" >"$log_path" 2>&1 &
        pids+=("$!")
        labels+=("support_$support")
    done

    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[stage0.4b] completed ${labels[$index]}"
        else
            echo "[stage0.4b] failed ${labels[$index]}" >&2
            tail -n 40 "$BASIS_DENSITY_DIR/${labels[$index]}.log" >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 )) || return 1

    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" \
        "$ROOT_DIR/hamer_tactile_ft/aggregate_basis_density_sweep.py" \
        --input-root "$BASIS_DENSITY_DIR" \
        --output-dir "$BASIS_DENSITY_DIR"
}

resolve_density_support4_dir() {
    local candidate
    for candidate in \
        "$ROOT_DIR/hamer_tactile_ft/reports/stage0_4b_basis_density/support_4" \
        "$ROOT_DIR/hamer_tactile_ft/stage0_4b_basis_density/support_4"
    do
        if [[ -f "$candidate/summary.json" && -f "$candidate/canonical_surface_basis_cleanup.npz" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "Could not find the completed Stage 0.4b support-4 audit." >&2
    echo "Run '$0 density' first or copy the audited support_4 directory." >&2
    return 1
}

ensure_surface_basis_runtime() {
    local audit_dir
    if [[ -f "$SURFACE_BASIS_RUNTIME" ]]; then
        return 0
    fi
    audit_dir="$(resolve_density_support4_dir)"
    mkdir -p "$(dirname "$SURFACE_BASIS_RUNTIME")"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/export_surface_basis_runtime.py" \
        --audit-dir "$audit_dir" \
        --coefficient-dim 4096 \
        --output "$SURFACE_BASIS_RUNTIME"
}

run_learnability_prepare() {
    cache_ready || {
        echo "Feature cache is not ready. Run this first:" >&2
        echo "  $0 prepare" >&2
        return 2
    }
    [[ -f "$TACTILE_BASE_CHECKPOINT" ]] || {
        echo "Missing base checkpoint: $TACTILE_BASE_CHECKPOINT" >&2
        return 2
    }
    ensure_surface_basis_runtime
    mkdir -p "$SURFACE_LEARNABILITY_DIR"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_surface_decoder_learnability.py" \
        prepare \
        --feature-cache "$TACTILE_FEATURE_CACHE" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --surface-basis "$SURFACE_BASIS_RUNTIME" \
        --output-dir "$SURFACE_LEARNABILITY_DIR/prepared" \
        --device "$LOCALIZATION_DEVICE" \
        --sample-limit "$LEARNABILITY_SAMPLE_LIMIT" \
        --batch-size "$LEARNABILITY_PREP_BATCH_SIZE" \
        "$@"
}

run_learnability_matrix() {
    local failed index gpu label architecture objective population epochs log_path pid
    local -a gpus labels architectures objectives populations pids
    run_learnability_prepare
    IFS=',' read -r -a gpus <<< "$LEARNABILITY_GPUS"
    (( ${#gpus[@]} >= 8 )) || {
        echo "LEARNABILITY_GPUS must provide eight GPU ids; got ${#gpus[@]}." >&2
        return 2
    }
    labels=(
        general_linear_pressure general_linear_coefficient
        general_nonlinear_pressure general_nonlinear_coefficient
        memorize_linear_pressure memorize_linear_coefficient
        memorize_nonlinear_pressure memorize_nonlinear_coefficient
    )
    architectures=(linear linear nonlinear nonlinear linear linear nonlinear nonlinear)
    objectives=(pressure coefficient pressure coefficient pressure coefficient pressure coefficient)
    populations=(general general general general memorize memorize memorize memorize)
    pids=()

    learnability_interrupt() {
        local child
        trap - INT TERM
        echo "Stopping surface learnability workers; completed probes remain reusable." >&2
        for child in "${pids[@]}"; do
            kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap learnability_interrupt INT TERM

    for index in "${!labels[@]}"; do
        gpu="${gpus[$index]//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ ]] || {
            echo "Invalid GPU id: $gpu" >&2
            learnability_interrupt
        }
        label="${labels[$index]}"
        architecture="${architectures[$index]}"
        objective="${objectives[$index]}"
        population="${populations[$index]}"
        epochs="$LEARNABILITY_EPOCHS"
        local -a population_args=()
        if [[ "$population" == memorize ]]; then
            epochs="$LEARNABILITY_MEM_EPOCHS"
            population_args=(--memorization --memorization-samples "$LEARNABILITY_MEM_SAMPLES")
        fi
        log_path="$SURFACE_LEARNABILITY_DIR/$label.log"
        mkdir -p "$SURFACE_LEARNABILITY_DIR/$label"
        echo "[surface-learnability] $label gpu=$gpu log=$log_path"
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_surface_decoder_learnability.py" \
                run \
                --prepared-dir "$SURFACE_LEARNABILITY_DIR/prepared" \
                --surface-basis "$SURFACE_BASIS_RUNTIME" \
                --output-dir "$SURFACE_LEARNABILITY_DIR/$label" \
                --architecture "$architecture" \
                --objective "$objective" \
                --device cuda:0 \
                --epochs "$epochs" \
                --batch-size "$LEARNABILITY_BATCH_SIZE" \
                --eval-batch-size "$LEARNABILITY_EVAL_BATCH_SIZE" \
                "${population_args[@]}" >"$log_path" 2>&1 &
        pid=$!
        pids+=("$pid")
    done

    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[surface-learnability] completed ${labels[$index]}"
        else
            echo "[surface-learnability] failed ${labels[$index]}" >&2
            tail -n 60 "$SURFACE_LEARNABILITY_DIR/${labels[$index]}.log" >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 )) || return 1
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_surface_decoder_learnability.py" \
        aggregate --input-root "$SURFACE_LEARNABILITY_DIR"
}

run_spatial_dependency() {
    cache_ready || {
        echo "Feature cache is not ready. Run this first:" >&2
        echo "  $0 prepare" >&2
        return 2
    }
    [[ -f "$TACTILE_BASE_CHECKPOINT" ]] || {
        echo "Missing base checkpoint: $TACTILE_BASE_CHECKPOINT" >&2
        return 2
    }
    mkdir -p "$FULLGRID_SPATIAL_DIR"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_fullgrid_spatial_dependency.py" \
        --feature-cache "$TACTILE_FEATURE_CACHE" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --output-dir "$FULLGRID_SPATIAL_DIR" \
        --device "$LOCALIZATION_DEVICE" \
        --sample-limit "$FULLGRID_SPATIAL_SAMPLES" \
        --batch-size "$FULLGRID_SPATIAL_BATCH_SIZE" \
        --variants "$FULLGRID_SPATIAL_VARIANTS" \
        "$@"
}

attribution_cache_args() {
    local split="$1" sample_limit per_sequence
    if [[ "$split" == train ]]; then
        sample_limit="$ATTRIBUTION_TRAIN_SAMPLES"
        per_sequence="$ATTRIBUTION_TRAIN_PER_SEQUENCE"
    else
        sample_limit="$ATTRIBUTION_EVAL_SAMPLES"
        per_sequence="$ATTRIBUTION_EVAL_PER_SEQUENCE"
    fi
    ATTRIBUTION_CACHE_BUILD_ARGS=(
        --fields z_rgb,tactile_signal,has_tactile
        --datasets touchanything
        --split "$split"
        --bbox-rescale-factor 1.2
        --bbox-source-policy sam3_only
        --input-resolution 256x192
        --batch-size "$ATTRIBUTION_CACHE_BATCH_SIZE"
        --sample-limit "$sample_limit"
        --max-samples-per-sequence "$per_sequence"
        --sample-seed 521
    )
}

ensure_attribution_selection() {
    local split="$1" sample_limit per_sequence selection_path
    if [[ "$split" == train ]]; then
        sample_limit="$ATTRIBUTION_TRAIN_SAMPLES"
        per_sequence="$ATTRIBUTION_TRAIN_PER_SEQUENCE"
    else
        sample_limit="$ATTRIBUTION_EVAL_SAMPLES"
        per_sequence="$ATTRIBUTION_EVAL_PER_SEQUENCE"
    fi
    selection_path="$SURFACE_ATTRIBUTION_CACHE_ROOT/selections/${split}_n${sample_limit}_q${per_sequence}_s521.npy"
    mkdir -p "$(dirname "$selection_path")"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/tactile_input_priors/cache_tactile_features.py" \
        --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
        --dino-weights "$DINO_WEIGHTS" \
        --selection-output "$selection_path" \
        "${ATTRIBUTION_CACHE_BUILD_ARGS[@]}"
    ATTRIBUTION_CACHE_BUILD_ARGS+=(--selected-indices-file "$selection_path")
}

resolve_attribution_cache_split() {
    local split="$1" key
    local -a gpu_array
    attribution_cache_args "$split"
    ensure_attribution_selection "$split"
    IFS=',' read -r -a gpu_array <<< "$ATTRIBUTION_CACHE_GPUS"
    (( ${#gpu_array[@]} > 0 )) || {
        echo "ATTRIBUTION_CACHE_GPUS must contain at least one GPU." >&2
        return 2
    }
    key="$(
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TACTILE_PYTHON" "$ROOT_DIR/tactile_input_priors/cache_tactile_features.py" \
            --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
            --dino-weights "$DINO_WEIGHTS" \
            --num-partitions "${#gpu_array[@]}" \
            --print-cache-key \
            "${ATTRIBUTION_CACHE_BUILD_ARGS[@]}"
    )"
    [[ "$key" =~ ^[0-9a-f]{64}$ ]] || {
        echo "Could not resolve attribution cache identity for split=$split: $key" >&2
        return 1
    }
    ATTRIBUTION_CACHE_RESOLVED_PATH="$SURFACE_ATTRIBUTION_CACHE_ROOT/${split}-${key:0:20}"
}

ensure_attribution_cache_split() {
    local split="$1" path
    resolve_attribution_cache_split "$split"
    path="$ATTRIBUTION_CACHE_RESOLVED_PATH"
    if cache_path_ready "$path"; then
        echo "[surface-attribution] reuse cache split=$split path=$path"
        return 0
    fi
    echo "[surface-attribution] build cache split=$split path=$path"
    mkdir -p "$path/logs"
    TACTILE_BASE_CHECKPOINT="$TACTILE_BASE_CHECKPOINT" \
    DINO_WEIGHTS="$DINO_WEIGHTS" \
    CACHE_GPUS="$ATTRIBUTION_CACHE_GPUS" \
    CACHE_LOG_DIR="$path/logs" \
    TACTILE_PYTHON="$TACTILE_PYTHON" \
    "$ROOT_DIR/tactile_input_priors/run.sh" cache-tactile-8gpu \
        --cache-dir "$path" \
        "${ATTRIBUTION_CACHE_BUILD_ARGS[@]}"
    cache_path_ready "$path" || {
        echo "Attribution cache is incomplete: $path" >&2
        return 1
    }
}

run_attribution_prepare() {
    local split index gpu log_path failed pid
    local -a splits=(train val test_seen test_unseen) gpus pids labels cache_paths
    [[ -f "$TACTILE_BASE_CHECKPOINT" ]] || {
        echo "Missing base checkpoint: $TACTILE_BASE_CHECKPOINT" >&2
        return 2
    }
    [[ -f "$DINO_WEIGHTS" ]] || {
        echo "Missing DINO weights: $DINO_WEIGHTS" >&2
        return 2
    }
    ensure_surface_basis_runtime
    mkdir -p "$SURFACE_ATTRIBUTION_DIR/prepared" "$SURFACE_ATTRIBUTION_DIR/logs"
    cache_paths=()
    for split in "${splits[@]}"; do
        ensure_attribution_cache_split "$split"
        cache_paths+=("$ATTRIBUTION_CACHE_RESOLVED_PATH")
    done
    IFS=',' read -r -a gpus <<< "$ATTRIBUTION_PREP_GPUS"
    (( ${#gpus[@]} >= ${#splits[@]} )) || {
        echo "ATTRIBUTION_PREP_GPUS must provide four GPU ids." >&2
        return 2
    }
    pids=()
    labels=()
    attribution_prepare_interrupt() {
        local child
        trap - INT TERM
        echo "Stopping attribution preparation workers; completed splits remain reusable." >&2
        for child in "${pids[@]}"; do
            kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap attribution_prepare_interrupt INT TERM
    for index in "${!splits[@]}"; do
        split="${splits[$index]}"
        gpu="${gpus[$index]//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ ]] || {
            echo "Invalid attribution preparation GPU: $gpu" >&2
            attribution_prepare_interrupt
        }
        log_path="$SURFACE_ATTRIBUTION_DIR/logs/prepare_${split}.log"
        echo "[surface-attribution] prepare split=$split gpu=$gpu log=$log_path"
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_surface_mapping_attribution.py" \
                prepare-split \
                --feature-cache "${cache_paths[$index]}" \
                --base-checkpoint "$TACTILE_BASE_CHECKPOINT" \
                --surface-basis "$SURFACE_BASIS_RUNTIME" \
                --split "$split" \
                --output-dir "$SURFACE_ATTRIBUTION_DIR/prepared/$split" \
                --device cuda:0 \
                --batch-size "$ATTRIBUTION_PREP_BATCH_SIZE" \
                --oracle-sample-limit "$ATTRIBUTION_ORACLE_SAMPLES" \
                >"$log_path" 2>&1 &
        pid=$!
        pids+=("$pid")
        labels+=("$split")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[surface-attribution] prepared ${labels[$index]}"
        else
            echo "[surface-attribution] failed preparing ${labels[$index]}" >&2
            tail -n 80 "$SURFACE_ATTRIBUTION_DIR/logs/prepare_${labels[$index]}.log" >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 ))
}

run_attribution_train() {
    local index variant gpu log_path failed pid split
    local -a variants=(basis direct) gpus pids
    ensure_surface_basis_runtime
    for split in train val test_seen test_unseen; do
        [[ -f "$SURFACE_ATTRIBUTION_DIR/prepared/$split/PREPARED.json" ]] || {
            echo "Missing prepared attribution split: $split" >&2
            echo "Run '$0 attribution-prepare' first." >&2
            return 2
        }
    done
    IFS=',' read -r -a gpus <<< "$ATTRIBUTION_TRAIN_GPUS"
    (( ${#gpus[@]} >= ${#variants[@]} )) || {
        echo "ATTRIBUTION_TRAIN_GPUS must provide two GPU ids." >&2
        return 2
    }
    mkdir -p "$SURFACE_ATTRIBUTION_DIR/logs"
    pids=()
    attribution_train_interrupt() {
        local child
        trap - INT TERM
        echo "Stopping attribution heads; completed checkpoints remain reusable." >&2
        for child in "${pids[@]}"; do
            kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap attribution_train_interrupt INT TERM
    for index in "${!variants[@]}"; do
        variant="${variants[$index]}"
        gpu="${gpus[$index]//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ ]] || {
            echo "Invalid attribution training GPU: $gpu" >&2
            attribution_train_interrupt
        }
        log_path="$SURFACE_ATTRIBUTION_DIR/logs/train_${variant}.log"
        echo "[surface-attribution] train variant=$variant gpu=$gpu log=$log_path"
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_surface_mapping_attribution.py" \
                train \
                --prepared-root "$SURFACE_ATTRIBUTION_DIR/prepared" \
                --surface-basis "$SURFACE_BASIS_RUNTIME" \
                --output-dir "$SURFACE_ATTRIBUTION_DIR/$variant" \
                --variant "$variant" \
                --device cuda:0 \
                --epochs "$ATTRIBUTION_EPOCHS" \
                --batch-size "$ATTRIBUTION_BATCH_SIZE" \
                --eval-batch-size "$ATTRIBUTION_EVAL_BATCH_SIZE" \
                >"$log_path" 2>&1 &
        pid=$!
        pids+=("$pid")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[surface-attribution] completed ${variants[$index]}"
        else
            echo "[surface-attribution] failed ${variants[$index]}" >&2
            tail -n 80 "$SURFACE_ATTRIBUTION_DIR/logs/train_${variants[$index]}.log" >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 )) || return 1
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_surface_mapping_attribution.py" \
        aggregate --input-root "$SURFACE_ATTRIBUTION_DIR"
}

ensure_routing_prepared() {
    local split
    for split in train val test_seen test_unseen; do
        if [[ ! -f "$SURFACE_ATTRIBUTION_DIR/prepared/$split/PREPARED.json" ]]; then
            echo "[canonical-routing] Stage 0.7 prepared splits are missing; building them now."
            run_attribution_prepare
            break
        fi
    done
    for split in train val test_seen test_unseen; do
        [[ -f "$SURFACE_ATTRIBUTION_DIR/prepared/$split/PREPARED.json" ]] || {
            echo "Missing prepared routing split: $split" >&2
            return 2
        }
    done
}

run_routing_geometry() {
    local anchor output
    local -a anchors
    ensure_surface_basis_runtime
    mkdir -p "$CANONICAL_ROUTING_DIR/geometry"
    IFS=',' read -r -a anchors <<< "$ROUTING_ANCHOR_COUNTS"
    for anchor in "${anchors[@]}"; do
        anchor="${anchor//[[:space:]]/}"
        [[ "$anchor" == 256 || "$anchor" == 512 ]] || {
            echo "Unsupported routing anchor count: $anchor" >&2
            return 2
        }
        output="$CANONICAL_ROUTING_DIR/geometry/a${anchor}.pt"
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
            prepare-geometry \
            --surface-basis "$SURFACE_BASIS_RUNTIME" \
            --anchor-count "$anchor" \
            --output "$output"
    done
}

build_routing_matrix() {
    local anchor mode source short_source
    local -a anchors modes sources
    ROUTING_LABELS=()
    ROUTING_MATRIX_ANCHORS=()
    ROUTING_MATRIX_MODES=()
    ROUTING_MATRIX_SOURCES=()
    IFS=',' read -r -a anchors <<< "$ROUTING_ANCHOR_COUNTS"
    IFS=',' read -r -a modes <<< "$ROUTING_MODES"
    IFS=',' read -r -a sources <<< "$ROUTING_SOURCES"
    for anchor in "${anchors[@]}"; do
        anchor="${anchor//[[:space:]]/}"
        for mode in "${modes[@]}"; do
            mode="${mode//[[:space:]]/}"
            for source in "${sources[@]}"; do
                source="${source//[[:space:]]/}"
                [[ "$anchor" == 256 || "$anchor" == 512 ]] || {
                    echo "Unsupported routing anchor count: $anchor" >&2
                    return 2
                }
                [[ "$mode" == competitive || "$mode" == independent ]] || {
                    echo "Unsupported routing mode: $mode" >&2
                    return 2
                }
                [[ "$source" == spatial || "$source" == global_control ]] || {
                    echo "Unsupported routing source: $source" >&2
                    return 2
                }
                short_source="sp"
                [[ "$source" == global_control ]] && short_source="glb"
                ROUTING_LABELS+=("${mode}_${short_source}_a${anchor}")
                ROUTING_MATRIX_ANCHORS+=("$anchor")
                ROUTING_MATRIX_MODES+=("$mode")
                ROUTING_MATRIX_SOURCES+=("$source")
            done
        done
    done
}

run_routing_train() {
    local index gpu label anchor mode source log_path pid failed
    local -a gpus pids
    ensure_routing_prepared
    run_routing_geometry
    build_routing_matrix
    IFS=',' read -r -a gpus <<< "$ROUTING_GPUS"
    (( ${#gpus[@]} >= ${#ROUTING_LABELS[@]} )) || {
        echo "ROUTING_GPUS must provide ${#ROUTING_LABELS[@]} GPU ids." >&2
        return 2
    }
    mkdir -p "$CANONICAL_ROUTING_DIR/logs"
    pids=()
    routing_train_interrupt() {
        local child
        trap - INT TERM
        echo "Stopping routing trainers; atomic best/last checkpoints remain resumable." >&2
        for child in "${pids[@]}"; do
            kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap routing_train_interrupt INT TERM
    for index in "${!ROUTING_LABELS[@]}"; do
        gpu="${gpus[$index]//[[:space:]]/}"
        label="${ROUTING_LABELS[$index]}"
        anchor="${ROUTING_MATRIX_ANCHORS[$index]}"
        mode="${ROUTING_MATRIX_MODES[$index]}"
        source="${ROUTING_MATRIX_SOURCES[$index]}"
        [[ "$gpu" =~ ^[0-9]+$ ]] || {
            echo "Invalid routing GPU: $gpu" >&2
            routing_train_interrupt
        }
        log_path="$CANONICAL_ROUTING_DIR/logs/train_${label}.log"
        echo "[canonical-routing] train $label gpu=$gpu log=$log_path"
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
                train \
                --prepared-root "$SURFACE_ATTRIBUTION_DIR/prepared" \
                --surface-basis "$SURFACE_BASIS_RUNTIME" \
                --geometry-cache "$CANONICAL_ROUTING_DIR/geometry/a${anchor}.pt" \
                --output-dir "$CANONICAL_ROUTING_DIR/$label" \
                --device cuda:0 \
                --anchor-count "$anchor" \
                --dimension "$ROUTING_DIM" \
                --heads "$ROUTING_HEADS" \
                --layers "$ROUTING_LAYERS" \
                --routing-mode "$mode" \
                --source "$source" \
                --dropout "$ROUTING_DROPOUT" \
                --max-logit-delta "$ROUTING_MAX_LOGIT_DELTA" \
                --epochs "$ROUTING_EPOCHS" \
                --batch-size "$ROUTING_BATCH_SIZE" \
                --eval-batch-size "$ROUTING_EVAL_BATCH_SIZE" \
                --bf16 \
                >"$log_path" 2>&1 &
        pid=$!
        pids+=("$pid")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[canonical-routing] trained ${ROUTING_LABELS[$index]}"
        else
            echo "[canonical-routing] failed ${ROUTING_LABELS[$index]}" >&2
            tail -n 100 \
                "$CANONICAL_ROUTING_DIR/logs/train_${ROUTING_LABELS[$index]}.log" \
                >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 ))
}

run_routing_eval() {
    local index gpu label anchor log_path checkpoint pid failed
    local -a gpus pids
    ensure_routing_prepared
    run_routing_geometry
    build_routing_matrix
    IFS=',' read -r -a gpus <<< "$ROUTING_GPUS"
    (( ${#gpus[@]} >= ${#ROUTING_LABELS[@]} )) || {
        echo "ROUTING_GPUS must provide ${#ROUTING_LABELS[@]} GPU ids." >&2
        return 2
    }
    mkdir -p "$CANONICAL_ROUTING_DIR/logs"
    pids=()
    routing_eval_interrupt() {
        local child
        trap - INT TERM
        echo "Stopping routing evaluators; completed reports remain reusable." >&2
        for child in "${pids[@]}"; do
            kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap routing_eval_interrupt INT TERM
    for index in "${!ROUTING_LABELS[@]}"; do
        gpu="${gpus[$index]//[[:space:]]/}"
        label="${ROUTING_LABELS[$index]}"
        anchor="${ROUTING_MATRIX_ANCHORS[$index]}"
        checkpoint="$CANONICAL_ROUTING_DIR/$label/best_loss.pt"
        [[ -f "$checkpoint" ]] || {
            echo "Missing routing checkpoint: $checkpoint" >&2
            routing_eval_interrupt
        }
        log_path="$CANONICAL_ROUTING_DIR/logs/eval_${label}.log"
        echo "[canonical-routing] evaluate $label gpu=$gpu log=$log_path"
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
                evaluate \
                --prepared-root "$SURFACE_ATTRIBUTION_DIR/prepared" \
                --surface-basis "$SURFACE_BASIS_RUNTIME" \
                --geometry-cache "$CANONICAL_ROUTING_DIR/geometry/a${anchor}.pt" \
                --checkpoint "$checkpoint" \
                --output-dir "$CANONICAL_ROUTING_DIR/$label" \
                --device cuda:0 \
                --batch-size "$ROUTING_EVAL_BATCH_SIZE" \
                --bf16 \
                >"$log_path" 2>&1 &
        pid=$!
        pids+=("$pid")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[canonical-routing] evaluated ${ROUTING_LABELS[$index]}"
        else
            echo "[canonical-routing] failed evaluating ${ROUTING_LABELS[$index]}" >&2
            tail -n 100 \
                "$CANONICAL_ROUTING_DIR/logs/eval_${ROUTING_LABELS[$index]}.log" \
                >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 )) || return 1
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" \
        "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
        aggregate --input-root "$CANONICAL_ROUTING_DIR"
}

run_routing_v2_geometry() {
    local output="$CANONICAL_ROUTING_V2_DIR/geometry/a${ROUTING_V2_ANCHOR_COUNT}.pt"
    [[ "$ROUTING_V2_ANCHOR_COUNT" == 256 ]] || {
        echo "Stage 2.1 currently fixes ROUTING_V2_ANCHOR_COUNT=256." >&2
        return 2
    }
    ensure_surface_basis_runtime
    mkdir -p "$(dirname "$output")"
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" \
        "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
        prepare-geometry \
        --surface-basis "$SURFACE_BASIS_RUNTIME" \
        --anchor-count "$ROUTING_V2_ANCHOR_COUNT" \
        --output "$output"
}

run_routing_v2_prepare() {
    local split index cache_path log_path pid failed
    local -a splits=(train val test_seen test_unseen) cache_paths pids
    ensure_routing_prepared
    mkdir -p "$CANONICAL_ROUTING_V2_DIR/raw_prepared" \
        "$CANONICAL_ROUTING_V2_DIR/logs"
    cache_paths=()
    for split in "${splits[@]}"; do
        ensure_attribution_cache_split "$split"
        cache_paths+=("$ATTRIBUTION_CACHE_RESOLVED_PATH")
    done
    pids=()
    routing_v2_prepare_interrupt() {
        local child
        trap - INT TERM
        echo "Stopping Stage 2.1 feature alignment; completed splits remain reusable." >&2
        for child in "${pids[@]}"; do
            kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap routing_v2_prepare_interrupt INT TERM
    for index in "${!splits[@]}"; do
        split="${splits[$index]}"
        cache_path="${cache_paths[$index]}"
        log_path="$CANONICAL_ROUTING_V2_DIR/logs/prepare_rezero_${split}.log"
        echo "[canonical-routing-v2] align split=$split log=$log_path"
        setsid env \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
                prepare-features \
                --prepared-split "$SURFACE_ATTRIBUTION_DIR/prepared/$split" \
                --feature-cache "$cache_path" \
                --output-dir "$CANONICAL_ROUTING_V2_DIR/raw_prepared/$split" \
                --batch-size "$ROUTING_V2_PREP_BATCH_SIZE" \
                >"$log_path" 2>&1 &
        pid=$!
        pids+=("$pid")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[canonical-routing-v2] aligned ${splits[$index]}"
        else
            echo "[canonical-routing-v2] failed aligning ${splits[$index]}" >&2
            tail -n 100 \
                "$CANONICAL_ROUTING_V2_DIR/logs/prepare_rezero_${splits[$index]}.log" \
                >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 ))
}

build_routing_v2_matrix() {
    local source seed short_source
    local -a sources seeds
    ROUTING_V2_LABELS=()
    ROUTING_V2_MATRIX_SOURCES=()
    ROUTING_V2_MATRIX_SEEDS=()
    IFS=',' read -r -a sources <<< "$ROUTING_V2_FEATURE_SOURCES"
    IFS=',' read -r -a seeds <<< "$ROUTING_V2_SEEDS"
    for source in "${sources[@]}"; do
        source="${source//[[:space:]]/}"
        [[ "$source" == projected32 || "$source" == rezero256 ]] || {
            echo "Unsupported Stage 2.1 feature source: $source" >&2
            return 2
        }
        short_source="p32"
        [[ "$source" == rezero256 ]] && short_source="r256"
        for seed in "${seeds[@]}"; do
            seed="${seed//[[:space:]]/}"
            [[ "$seed" =~ ^[0-9]+$ ]] || {
                echo "Invalid Stage 2.1 seed: $seed" >&2
                return 2
            }
            ROUTING_V2_LABELS+=("${short_source}_s${seed}")
            ROUTING_V2_MATRIX_SOURCES+=("$source")
            ROUTING_V2_MATRIX_SEEDS+=("$seed")
        done
    done
}

run_routing_v2_train() {
    local index gpu label feature_source seed log_path pid failed
    local -a gpus pids
    run_routing_v2_prepare
    run_routing_v2_geometry
    build_routing_v2_matrix
    IFS=',' read -r -a gpus <<< "$ROUTING_V2_GPUS"
    (( ${#gpus[@]} >= ${#ROUTING_V2_LABELS[@]} )) || {
        echo "ROUTING_V2_GPUS must provide ${#ROUTING_V2_LABELS[@]} GPU ids." >&2
        return 2
    }
    pids=()
    routing_v2_train_interrupt() {
        local child
        trap - INT TERM
        echo "Stopping Stage 2.1 trainers; atomic last.pt files remain resumable." >&2
        for child in "${pids[@]}"; do
            kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap routing_v2_train_interrupt INT TERM
    for index in "${!ROUTING_V2_LABELS[@]}"; do
        gpu="${gpus[$index]//[[:space:]]/}"
        label="${ROUTING_V2_LABELS[$index]}"
        feature_source="${ROUTING_V2_MATRIX_SOURCES[$index]}"
        seed="${ROUTING_V2_MATRIX_SEEDS[$index]}"
        [[ "$gpu" =~ ^[0-9]+$ ]] || {
            echo "Invalid Stage 2.1 GPU: $gpu" >&2
            routing_v2_train_interrupt
        }
        log_path="$CANONICAL_ROUTING_V2_DIR/logs/train_${label}.log"
        echo "[canonical-routing-v2] train $label gpu=$gpu log=$log_path"
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
                train \
                --prepared-root "$SURFACE_ATTRIBUTION_DIR/prepared" \
                --raw-prepared-root "$CANONICAL_ROUTING_V2_DIR/raw_prepared" \
                --surface-basis "$SURFACE_BASIS_RUNTIME" \
                --geometry-cache "$CANONICAL_ROUTING_V2_DIR/geometry/a${ROUTING_V2_ANCHOR_COUNT}.pt" \
                --output-dir "$CANONICAL_ROUTING_V2_DIR/$label" \
                --device cuda:0 \
                --anchor-count "$ROUTING_V2_ANCHOR_COUNT" \
                --dimension "$ROUTING_V2_DIM" \
                --heads "$ROUTING_V2_HEADS" \
                --layers "$ROUTING_V2_LAYERS" \
                --routing-mode competitive \
                --source spatial \
                --architecture evidence_only \
                --feature-source "$feature_source" \
                --dropout "$ROUTING_V2_DROPOUT" \
                --max-logit-delta "$ROUTING_V2_MAX_LOGIT_DELTA" \
                --epochs "$ROUTING_V2_EPOCHS" \
                --batch-size "$ROUTING_V2_BATCH_SIZE" \
                --eval-batch-size "$ROUTING_V2_EVAL_BATCH_SIZE" \
                --seed "$seed" \
                --bf16 \
                >"$log_path" 2>&1 &
        pid=$!
        pids+=("$pid")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[canonical-routing-v2] trained ${ROUTING_V2_LABELS[$index]}"
        else
            echo "[canonical-routing-v2] failed ${ROUTING_V2_LABELS[$index]}" >&2
            tail -n 100 \
                "$CANONICAL_ROUTING_V2_DIR/logs/train_${ROUTING_V2_LABELS[$index]}.log" \
                >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 ))
}

run_routing_v2_eval() {
    local index gpu label checkpoint log_path pid failed
    local -a gpus pids
    run_routing_v2_prepare
    run_routing_v2_geometry
    build_routing_v2_matrix
    IFS=',' read -r -a gpus <<< "$ROUTING_V2_GPUS"
    (( ${#gpus[@]} >= ${#ROUTING_V2_LABELS[@]} )) || {
        echo "ROUTING_V2_GPUS must provide ${#ROUTING_V2_LABELS[@]} GPU ids." >&2
        return 2
    }
    pids=()
    routing_v2_eval_interrupt() {
        local child
        trap - INT TERM
        echo "Stopping Stage 2.1 evaluators; completed reports remain reusable." >&2
        for child in "${pids[@]}"; do
            kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
        done
        wait || true
        exit 130
    }
    trap routing_v2_eval_interrupt INT TERM
    for index in "${!ROUTING_V2_LABELS[@]}"; do
        gpu="${gpus[$index]//[[:space:]]/}"
        label="${ROUTING_V2_LABELS[$index]}"
        checkpoint="$CANONICAL_ROUTING_V2_DIR/$label/best_loss.pt"
        [[ -f "$checkpoint" ]] || {
            echo "Missing Stage 2.1 checkpoint: $checkpoint" >&2
            routing_v2_eval_interrupt
        }
        log_path="$CANONICAL_ROUTING_V2_DIR/logs/eval_${label}.log"
        echo "[canonical-routing-v2] evaluate $label gpu=$gpu log=$log_path"
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TACTILE_PYTHON" \
            "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
                evaluate \
                --prepared-root "$SURFACE_ATTRIBUTION_DIR/prepared" \
                --raw-prepared-root "$CANONICAL_ROUTING_V2_DIR/raw_prepared" \
                --surface-basis "$SURFACE_BASIS_RUNTIME" \
                --geometry-cache "$CANONICAL_ROUTING_V2_DIR/geometry/a${ROUTING_V2_ANCHOR_COUNT}.pt" \
                --checkpoint "$checkpoint" \
                --output-dir "$CANONICAL_ROUTING_V2_DIR/$label" \
                --device cuda:0 \
                --batch-size "$ROUTING_V2_EVAL_BATCH_SIZE" \
                --bf16 \
                >"$log_path" 2>&1 &
        pid=$!
        pids+=("$pid")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[canonical-routing-v2] evaluated ${ROUTING_V2_LABELS[$index]}"
        else
            echo "[canonical-routing-v2] failed evaluating ${ROUTING_V2_LABELS[$index]}" >&2
            tail -n 100 \
                "$CANONICAL_ROUTING_V2_DIR/logs/eval_${ROUTING_V2_LABELS[$index]}.log" \
                >&2 || true
            failed=1
        fi
    done
    trap - INT TERM
    (( failed == 0 )) || return 1
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$TACTILE_PYTHON" \
        "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
        aggregate --input-root "$CANONICAL_ROUTING_V2_DIR"
}

case "$MODE" in
    prepare)
        prepare_cache
        ;;
    stage0|stage1)
        run_stage0 "$@"
        ;;
    cleanup|stage0.3)
        run_cleanup "$@"
        ;;
    capacity|stage0.4)
        run_capacity "$@"
        ;;
    density|stage0.4b)
        run_density "$@"
        ;;
    learnability-prepare|stage0.5-prepare)
        run_learnability_prepare "$@"
        ;;
    learnability|stage0.5)
        run_learnability_matrix
        ;;
    spatial-dependency|stage0.6)
        run_spatial_dependency "$@"
        ;;
    attribution-prepare|stage0.7-prepare)
        run_attribution_prepare
        ;;
    attribution-train|stage0.7-train)
        run_attribution_train
        ;;
    attribution|stage0.7)
        run_attribution_prepare
        run_attribution_train
        ;;
    routing-geometry|stage2-geometry)
        run_routing_geometry
        ;;
    routing-train|stage2-train)
        run_routing_train
        ;;
    routing-eval|stage2-eval)
        run_routing_eval
        ;;
    routing|stage2)
        run_routing_train
        run_routing_eval
        ;;
    routing-v2-prepare|stage2.1-prepare)
        run_routing_v2_prepare
        ;;
    routing-v2-train|stage2.1-train)
        run_routing_v2_train
        ;;
    routing-v2-eval|stage2.1-eval)
        run_routing_v2_eval
        ;;
    routing-v2|stage2.1)
        run_routing_v2_train
        run_routing_v2_eval
        ;;
    all)
        prepare_cache
        run_stage0 "$@"
        ;;
    self-test)
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_canonical_localization.py" \
            --self-test
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_surface_basis_cleanup.py" \
            --self-test
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_surface_decoder_learnability.py" \
            self-test
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_fullgrid_spatial_dependency.py" \
            --self-test
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_surface_mapping_attribution.py" \
            self-test
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TACTILE_PYTHON" "$ROOT_DIR/hamer_tactile_ft/audit_canonical_anchor_routing.py" \
            self-test
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac
