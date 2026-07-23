#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
DINO_WEIGHTS="${DINO_WEIGHTS:-$WORKSPACE_DIR/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
SAM3_RECONSTRUCTION_ROOT="${SAM3_RECONSTRUCTION_ROOT:-$WORKSPACE_DIR/sam3_bbox_reconstruction/outputs/full_reconstruction_flow}"
MODE="${1:-}"

usage() {
    cat <<EOF
Usage: $0 MODE [training options]

Current SAM3 FullGrid32 CoreLoc modes:
  dino-rezero-fullgrid32-coreloc-sam3-expanded
      Mixed OpenTouch + TouchAnything, bbox scale 2.0
  dino-rezero-fullgrid32-coreloc-sam3-touchanything
      TouchAnything only, bbox scale 2.0 (existing experiment name)
  dino-rezero-fullgrid32-coreloc-sam3-opentouch
      OpenTouch only, bbox scale 2.0
  dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop10
      TouchAnything only, bbox scale 1.0
  dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop12
      TouchAnything only, bbox scale 1.2
  dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop14
      TouchAnything only, bbox scale 1.4
  dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop16
      TouchAnything only, bbox scale 1.6
  dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop18
      TouchAnything only, bbox scale 1.8
  dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop20
      TouchAnything only, bbox scale 2.0 reference

Additional options are forwarded after the preset and therefore take precedence.
SAM3_RECONSTRUCTION_ROOT may override the reviewed bbox reconstruction root.
For arbitrary OpenTouch, TouchAnything, or mixed configurations, call
run_tactile_ft.sh directly with --datasets.
EOF
}

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
    usage
    exit 0
fi
shift

DINO_REZERO_ARGS=(
    --visual_backbone dinov3_hplus
    --tactile_head_type dense_v2_dino_rezero
    --backbone_feature_layers 8,16,24,32
    --dino_residual_max_scale 0.10
    --dino_residual_rms_budget 0.50
    --dino_weights "$DINO_WEIGHTS"
)

FULLGRID_CORELOC_ARGS=(
    --pool_layout fullgrid32
    --tail_l1_weight 0.0
    --location_loss_weight 0.001
    --location_gt_volume_thr 1.0
    --location_distribution_power 2.0
    --location_min_gt_peak 0.05
)

SAM3_INDEX_ARGS=(
    --bbox_source_policy sam3_only
    --index_manifest ""
    --lazy_index_records
    --no-persistent_workers
)

OT_BBOX_MANIFEST="$SAM3_RECONSTRUCTION_ROOT/opentouch/manifests/opentouch_sam3_v1.jsonl"
TA_BBOX_MANIFEST="$SAM3_RECONSTRUCTION_ROOT/touchanything/manifests/touchanything_sam3_v1_highconf.jsonl"
REQUIRED_BBOX_MANIFESTS=()
preset_args=()

set_sam3_preset() {
    local domain="$1"
    local exp_name="$2"
    local bbox_scale="$3"
    local datasets
    local bbox_manifests

    case "$domain" in
        mixed)
            datasets="opentouch,touchanything"
            bbox_manifests="$OT_BBOX_MANIFEST,$TA_BBOX_MANIFEST"
            REQUIRED_BBOX_MANIFESTS=("$OT_BBOX_MANIFEST" "$TA_BBOX_MANIFEST")
            ;;
        touchanything)
            datasets="touchanything"
            bbox_manifests="$TA_BBOX_MANIFEST"
            REQUIRED_BBOX_MANIFESTS=("$TA_BBOX_MANIFEST")
            ;;
        opentouch)
            datasets="opentouch"
            bbox_manifests="$OT_BBOX_MANIFEST"
            REQUIRED_BBOX_MANIFESTS=("$OT_BBOX_MANIFEST")
            ;;
        *)
            echo "Unsupported SAM3 preset domain: $domain" >&2
            exit 2
            ;;
    esac

    preset_args=(
        --exp_name "$exp_name"
        "${DINO_REZERO_ARGS[@]}"
        "${FULLGRID_CORELOC_ARGS[@]}"
        --bbox_rescale_factor "$bbox_scale"
        "${SAM3_INDEX_ARGS[@]}"
        --datasets "$datasets"
        --expected_datasets "$datasets"
        --bbox_manifests "$bbox_manifests"
    )
}

case "$MODE" in
    dino-rezero-fullgrid32-coreloc-sam3-expanded)
        set_sam3_preset \
            mixed \
            mixed_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3expanded \
            2.0
        ;;
    dino-rezero-fullgrid32-coreloc-sam3-touchanything)
        set_sam3_preset \
            touchanything \
            touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3expanded \
            2.0
        ;;
    dino-rezero-fullgrid32-coreloc-sam3-opentouch)
        set_sam3_preset \
            opentouch \
            opentouch_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3expanded \
            2.0
        ;;
    dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop10)
        set_sam3_preset \
            touchanything \
            touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop10 \
            1.0
        ;;
    dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop12)
        set_sam3_preset \
            touchanything \
            touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12 \
            1.2
        ;;
    dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop14)
        set_sam3_preset \
            touchanything \
            touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop14 \
            1.4
        ;;
    dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop16)
        set_sam3_preset \
            touchanything \
            touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop16 \
            1.6
        ;;
    dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop18)
        set_sam3_preset \
            touchanything \
            touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop18 \
            1.8
        ;;
    dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop20)
        set_sam3_preset \
            touchanything \
            touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop20 \
            2.0
        ;;
    *)
        echo "Unknown or retired mode '$MODE'." >&2
        usage >&2
        exit 2
        ;;
esac

for required_bbox_manifest in "${REQUIRED_BBOX_MANIFESTS[@]}"; do
    if [[ ! -f "$required_bbox_manifest" ]]; then
        echo "Reviewed SAM3 bbox manifest is required before training: $required_bbox_manifest" >&2
        echo "Resolved reconstruction root: $SAM3_RECONSTRUCTION_ROOT" >&2
        echo "Run the SAM3 full reconstruction and association pipeline first, or set" >&2
        echo "SAM3_RECONSTRUCTION_ROOT to the directory containing domain subdirectories." >&2
        exit 2
    fi
done

echo "Using SAM3 reconstruction root: $SAM3_RECONSTRUCTION_ROOT"

# Preset arguments establish the experiment; explicit user options come last.
exec "$SCRIPT_DIR/run_tactile_ft.sh" "${preset_args[@]}" "$@"
