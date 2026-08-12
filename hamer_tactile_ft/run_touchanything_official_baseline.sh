#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT="${1:-/home/ma-user/work/baseline_ckpt/touchanything/checkpoint_epoch_25.pth}"
CONFIG="${2:-$WORKSPACE_DIR/TouchAnything/configs/touchanything_with_glove_aug_wilor.yaml}"

DINOV2_REPO="${DINOV2_REPO:-/home/ma-user/.cache/torch/hub/facebookresearch_dinov2_main}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/home/ma-user/work/cfzhao/EgoTouch/extracted_frames}"
RAW_ROOT="${RAW_ROOT:-$(dirname "$PROCESSED_ROOT")}"
BBOX_MANIFESTS="${BBOX_MANIFESTS:-}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
VIEW_CONFIGS="${VIEW_CONFIGS:-${VIEWS:-ego,ego+left,ego+right,all}}"
POSE_SOURCE="${POSE_SOURCE:-}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
MAPPING_BATCH_SIZE="${MAPPING_BATCH_SIZE:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/eval_reports_touchanything_official_epoch25}"
SPLITS="${SPLITS:-test_seen,test_unseen}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "TouchAnything checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "TouchAnything config not found: $CONFIG" >&2
    exit 2
fi
if [[ ! -f "$DINOV2_REPO/hubconf.py" ]]; then
    echo "DINOv2 source repository not found: $DINOV2_REPO" >&2
    echo "Clone https://github.com/facebookresearch/dinov2.git there, or set DINOV2_REPO to another local clone." >&2
    exit 2
fi
if [[ ! -d "$PROCESSED_ROOT" ]]; then
    echo "Processed TouchAnything root not found: $PROCESSED_ROOT" >&2
    exit 2
fi
if [[ ! -d "$RAW_ROOT" ]]; then
    echo "Raw TouchAnything root not found: $RAW_ROOT" >&2
    exit 2
fi

IFS=',' read -r -a SPLIT_LIST <<< "$SPLITS"
IFS=',' read -r -a VIEW_LIST <<< "$VIEW_CONFIGS"
mkdir -p "$OUTPUT_ROOT/logs"
cd "$WORKSPACE_DIR"

echo "TouchAnything official baseline"
echo "  checkpoint : $CHECKPOINT"
echo "  config     : $CONFIG"
echo "  DINOv2 repo: $DINOV2_REPO"
echo "  raw root   : $RAW_ROOT"
echo "  GT/query   : $PROCESSED_ROOT"
echo "  views      : $VIEW_CONFIGS"
echo "  GPUs       : $GPUS"
echo "  output     : $OUTPUT_ROOT"
echo

for view in "${VIEW_LIST[@]}"; do
    view="${view//[[:space:]]/}"
    case "$view" in
        ego|ego+left|ego+right|all) ;;
        *)
            echo "Unsupported view configuration: $view" >&2
            exit 2
            ;;
    esac
    view_log_name="${view//+/_plus_}"
    for split in "${SPLIT_LIST[@]}"; do
        split="${split//[[:space:]]/}"
        query_manifest="$PROCESSED_ROOT/manifests/touchanything_${split}.queries.jsonl"
        if [[ ! -f "$query_manifest" ]]; then
            echo "Canonical query manifest not found: $query_manifest" >&2
            exit 2
        fi
        split_output="$OUTPUT_ROOT/$view/$split"
        log_file="$OUTPUT_ROOT/logs/${view_log_name}_${split}.log"
        command=(
            "$PYTHON_BIN" hamer_tactile_ft/eval_touchanything_official_baseline.py
            --checkpoint "$CHECKPOINT"
            --config "$CONFIG"
            --dinov2_repo "$DINOV2_REPO"
            --split "$split"
            --raw_root "$RAW_ROOT"
            --processed_root "$PROCESSED_ROOT"
            --query_manifest "$query_manifest"
            --bbox_source_policy sam3_only
            --views "$view"
            --gpus "$GPUS"
            --inference_batch_size "$INFERENCE_BATCH_SIZE"
            --mapping_batch_size "$MAPPING_BATCH_SIZE"
            --max_trajectories "$MAX_TRAJECTORIES"
            --output_dir "$split_output"
            --save_diagnostics
        )
        if [[ -n "$BBOX_MANIFESTS" ]]; then
            command+=(--bbox_manifests "$BBOX_MANIFESTS")
        fi
        if [[ -n "$POSE_SOURCE" ]]; then
            command+=(--pose_source "$POSE_SOURCE")
        fi

        echo "Running view=$view split=$split (log: $log_file)"
        "$PYTHON_BIN" hamer_tactile_ft/process_supervisor.py \
            --registry-dir "$SCRIPT_DIR/run_processes" \
            --grace-seconds 30 \
            --kill-wait-seconds 5 \
            -- "${command[@]}" 2>&1 | tee "$log_file"
    done
done

echo
echo "Official baseline evaluation complete: $OUTPUT_ROOT"
