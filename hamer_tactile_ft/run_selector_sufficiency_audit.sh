#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
MODE="${1:-all}"
SIGNAL="${2:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"
INDEX_WORKERS="${INDEX_WORKERS:-128}"
DATA_BACKEND="${DATA_BACKEND:-auto}"
HDF5_HANDLE_CACHE_SIZE="${HDF5_HANDLE_CACHE_SIZE:-4}"
HDF5_MANIFEST_CACHE_DIR="${HDF5_MANIFEST_CACHE_DIR:-$SCRIPT_DIR/hdf5_manifest_cache}"
DINO_WEIGHTS="${DINO_WEIGHTS:-/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
BBOX_MANIFESTS="${BBOX_MANIFESTS:-}"
QUERY_MANIFESTS="${QUERY_MANIFESTS:-}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRIPT_DIR/selector_sufficiency_artifacts}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/selector_sufficiency_audits/rgb_contact_ordinal}"
AUDIT_SPLITS="${AUDIT_SPLITS:-val,test_seen,test_unseen}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
FIT_VERTICES_PER_FRAME="${FIT_VERTICES_PER_FRAME:-64}"
FIT_MAX_ROWS="${FIT_MAX_ROWS:-2000000}"
CONTACT_CKPT="${CONTACT_CKPT:-selector-best}"
# This older run predates selector-best. Its validation AP/IoU and ordinal-bin
# MAE are all better at last than at loss-best, so last is the locked fallback.
ORDINAL_CKPT="${ORDINAL_CKPT:-last}"
DOWN_CKPT="${DOWN_CKPT:-selector-best}"
DOWN_CONTROL_CKPT="${DOWN_CONTROL_CKPT:-selector-best}"
PROCESS_GRACE_SECONDS="${PROCESS_GRACE_SECONDS:-30}"
PROCESS_KILL_WAIT_SECONDS="${PROCESS_KILL_WAIT_SECONDS:-5}"

signal_experiment() {
    case "$1" in
        contact) printf '%s' "ta_selector_grid_r256" ;;
        ordinal) printf '%s' "ta_selector_ordinal_r256" ;;
        down) printf '%s' "ta_selector_down_r256" ;;
        down_control) printf '%s' "ta_selector_down_ctl_r256" ;;
        *)
            echo "Unknown signal '$1'; use contact, ordinal, down, or down_control." >&2
            return 2
            ;;
    esac
}

signal_checkpoint() {
    case "$1" in
        contact) printf '%s' "$CONTACT_CKPT" ;;
        ordinal) printf '%s' "$ORDINAL_CKPT" ;;
        down) printf '%s' "$DOWN_CKPT" ;;
        down_control) printf '%s' "$DOWN_CONTROL_CKPT" ;;
        *)
            echo "Unknown signal '$1'; use contact, ordinal, down, or down_control." >&2
            return 2
            ;;
    esac
}

artifact_is_complete() {
    local artifact_dir="$1"
    local experiment="$2"
    local checkpoint="$3"
    [[ -f "$artifact_dir/_COMPLETE" && -f "$artifact_dir/artifact_config.json" ]] || return 1
    "$PYTHON_BIN" - "$artifact_dir/artifact_config.json" "$experiment" "$checkpoint" <<'PY'
import json
import sys

path, experiment, checkpoint = sys.argv[1:]
try:
    config = json.load(open(path, "r", encoding="utf-8"))
except Exception:
    raise SystemExit(1)
valid = (
    config.get("schema") == "tactile_selector_vertex_artifacts_v1"
    and config.get("status") == "complete"
    and str(config.get("exp_name")) == experiment
    and str(config.get("ckpt")) == checkpoint
)
raise SystemExit(0 if valid else 1)
PY
}

export_signal() {
    local signal="$1"
    local experiment
    experiment="$(signal_experiment "$signal")"
    local checkpoint
    checkpoint="$(signal_checkpoint "$signal")"
    local include_reference="--no-selector_artifact_include_reference"
    if [[ "$signal" == "contact" ]]; then
        include_reference="--selector_artifact_include_reference"
    fi
    IFS=',' read -r -a splits <<< "$AUDIT_SPLITS"
    for split in "${splits[@]}"; do
        split="${split//[[:space:]]/}"
        [[ -n "$split" ]] || continue
        local artifact_dir="$ARTIFACT_ROOT/$signal/$split"
        local report_dir="$ARTIFACT_ROOT/_eval_reports/$signal/$split"
        local log_dir="$ARTIFACT_ROOT/logs"
        local log_file="$log_dir/${signal}_${split}.log"
        mkdir -p "$artifact_dir" "$report_dir" "$log_dir"
        if artifact_is_complete "$artifact_dir" "$experiment" "$checkpoint"; then
            echo "[selector-audit] already complete; skipping signal=$signal split=$split ckpt=$checkpoint"
            continue
        fi
        command=(
            "$PYTHON_BIN" hamer_tactile_ft/eval_tactile_fast.py
            --exp_name "$experiment"
            --ckpt "$checkpoint"
            --datasets touchanything
            --split "$split"
            --gpus "$GPUS"
            --batch_size "$BATCH_SIZE"
            --num_workers "$NUM_WORKERS"
            --index_workers "$INDEX_WORKERS"
            --data_backend "$DATA_BACKEND"
            --hdf5_handle_cache_size "$HDF5_HANDLE_CACHE_SIZE"
            --hdf5_manifest_cache_dir "$HDF5_MANIFEST_CACHE_DIR"
            --dino_weights "$DINO_WEIGHTS"
            --report_dir "$report_dir"
            --selector_artifact_output_dir "$artifact_dir"
            "$include_reference"
            --no-rebuild_index
        )
        if [[ -n "$BBOX_MANIFESTS" ]]; then
            command+=(--bbox_manifests "$BBOX_MANIFESTS")
        fi
        if [[ -n "$QUERY_MANIFESTS" ]]; then
            command+=(--query_manifests "$QUERY_MANIFESTS")
        fi
        supervised=(
            "$PYTHON_BIN" hamer_tactile_ft/process_supervisor.py
            --registry-dir "$SCRIPT_DIR/run_processes"
            --grace-seconds "$PROCESS_GRACE_SECONDS"
            --kill-wait-seconds "$PROCESS_KILL_WAIT_SECONDS"
            --
            "${command[@]}"
        )
        echo "[selector-audit] exporting signal=$signal split=$split ckpt=$checkpoint"
        echo "[selector-audit] log=$log_file"
        "${supervised[@]}" 2>&1 | tee "$log_file"
    done
}

run_analysis() {
    echo "[selector-audit] running CPU sufficiency analysis"
    "$PYTHON_BIN" hamer_tactile_ft/audit_selector_sufficiency.py \
        --artifact-root "$ARTIFACT_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --splits "$AUDIT_SPLITS" \
        --fit-vertices-per-frame "$FIT_VERTICES_PER_FRAME" \
        --fit-max-rows "$FIT_MAX_ROWS" \
        --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
}

cd "$WORKSPACE_DIR"

case "$MODE" in
    export)
        [[ -n "$SIGNAL" ]] || {
            echo "Usage: $0 export contact|ordinal|down|down_control" >&2
            exit 2
        }
        export_signal "$SIGNAL"
        ;;
    export-all)
        for signal in contact ordinal down down_control; do
            export_signal "$signal"
        done
        ;;
    analyze)
        run_analysis
        ;;
    self-test)
        "$PYTHON_BIN" hamer_tactile_ft/audit_selector_sufficiency.py --self-test
        ;;
    all)
        for signal in contact ordinal down down_control; do
            export_signal "$signal"
        done
        run_analysis
        ;;
    *)
        echo "Usage: $0 export SIGNAL|export-all|analyze|self-test|all" >&2
        exit 2
        ;;
esac
