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

Preserved mainline presets:
  dense-v2-mixed                  Final-DINO Dense V2 legacy5 control
  dino-rezero-legacy5-mixed       Multilevel ReZero with legacy5 pooling
  dino-rezero-fullgrid32-mixed    Multilevel ReZero with FullGrid32 pooling
  dino-rezero-strictcontrol       Historical mixed strict control
  dino-rezero-opentouch-only      Historical OpenTouch-only strict control
  dino-rezero-touchanything-only  Historical TouchAnything-only strict control

SAM3 FullGrid32 + CoreLoc presets:
  fullgrid-coreloc-mixed
  fullgrid-coreloc-opentouch
  fullgrid-coreloc-touchanything
  fullgrid-coreloc-ta-crop10|crop12|crop14|crop16|crop18|crop20
  fullgrid-coreloc-ta-res320|res384
  fullgrid-coreloc-ta-capacity50|capacity72
  weight-plateau2-r256
  weight-linear3-r256

Long historical FullGrid/SAM3 preset names remain accepted as aliases.
Explicit options are forwarded last and therefore override preset values.
EOF
}

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
    usage
    exit 0
fi
shift

DINO_COMMON=(
    --visual_backbone dinov3_hplus
    --backbone_feature_layers 8,16,24,32
    --dino_weights "$DINO_WEIGHTS"
)
REZERO_COMMON=(
    "${DINO_COMMON[@]}"
    --tactile_head_type dense_v2_dino_rezero
    --dino_residual_max_scale 0.10
    --dino_residual_rms_budget 0.50
)
CORELOC_COMMON=(
    --pool_layout fullgrid32
    --location_loss_weight 0.001
    --location_gt_volume_thr 1.0
    --location_distribution_power 2.0
    --location_min_gt_peak 0.05
)
SAM3_INDEX=(
    --bbox_source_policy sam3_only
    --index_manifest ""
    --lazy_index_records
    --no-persistent_workers
)

OT_BBOX_MANIFEST="$SAM3_RECONSTRUCTION_ROOT/opentouch/manifests/opentouch_sam3_v1.jsonl"
TA_BBOX_MANIFEST="$SAM3_RECONSTRUCTION_ROOT/touchanything/manifests/touchanything_sam3_v1_highconf.jsonl"
required_manifests=()
preset_args=()

set_plain() {
    local exp_name="$1" datasets="$2" head="$3" pool="$4"
    preset_args=(
        --exp_name "$exp_name"
        "${DINO_COMMON[@]}"
        --tactile_head_type "$head"
        --pool_layout "$pool"
        --datasets "$datasets"
        --expected_datasets "$datasets"
    )
    if [[ "$head" == "dense_v2_dino_rezero" ]]; then
        preset_args+=(--dino_residual_max_scale 0.10 --dino_residual_rms_budget 0.50)
    fi
}

set_sam3() {
    local domain="$1" exp_name="$2" bbox_scale="$3"
    local datasets manifests
    case "$domain" in
        mixed)
            datasets="opentouch,touchanything"
            manifests="$OT_BBOX_MANIFEST,$TA_BBOX_MANIFEST"
            required_manifests=("$OT_BBOX_MANIFEST" "$TA_BBOX_MANIFEST")
            ;;
        opentouch)
            datasets="opentouch"
            manifests="$OT_BBOX_MANIFEST"
            required_manifests=("$OT_BBOX_MANIFEST")
            ;;
        touchanything)
            datasets="touchanything"
            manifests="$TA_BBOX_MANIFEST"
            required_manifests=("$TA_BBOX_MANIFEST")
            ;;
        *) echo "Unsupported domain: $domain" >&2; exit 2 ;;
    esac
    preset_args=(
        --exp_name "$exp_name"
        "${REZERO_COMMON[@]}"
        "${CORELOC_COMMON[@]}"
        --bbox_rescale_factor "$bbox_scale"
        "${SAM3_INDEX[@]}"
        --datasets "$datasets"
        --expected_datasets "$datasets"
        --bbox_manifests "$manifests"
    )
}

set_ta_variant() {
    local exp_name="$1" bbox_scale="$2" resolution="${3:-256x192}"
    set_sam3 touchanything "$exp_name" "$bbox_scale"
    preset_args+=(--input_resolution "$resolution" --accumulate_grad_batches 1)
}

case "$MODE" in
    dense-v2-mixed)
        set_plain mixed_dense_v2_control "opentouch,touchanything" dense_v2 legacy5 ;;
    dino-rezero-legacy5-mixed|dino-rezero-strictcontrol)
        set_plain mixed_dense_v2_dinov3_rezero_strictcontrol "opentouch,touchanything" dense_v2_dino_rezero legacy5 ;;
    dino-rezero-fullgrid32-mixed)
        set_plain mixed_dense_v2_dinov3_rezero_fullgrid32 "opentouch,touchanything" dense_v2_dino_rezero fullgrid32 ;;
    dino-rezero-opentouch-only)
        set_plain opentouch_dense_v2_dinov3_rezero_strictcontrol opentouch dense_v2_dino_rezero legacy5 ;;
    dino-rezero-touchanything-only)
        set_plain touchanything_dense_v2_dinov3_rezero_strictcontrol touchanything dense_v2_dino_rezero legacy5 ;;
    fullgrid-coreloc-mixed|dino-rezero-fullgrid32-coreloc-sam3-expanded)
        set_sam3 mixed mixed_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3expanded 2.0 ;;
    fullgrid-coreloc-opentouch|dino-rezero-fullgrid32-coreloc-sam3-opentouch)
        set_sam3 opentouch opentouch_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3expanded 2.0 ;;
    fullgrid-coreloc-touchanything|dino-rezero-fullgrid32-coreloc-sam3-touchanything)
        set_sam3 touchanything touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3expanded 2.0 ;;
    fullgrid-coreloc-ta-crop10|dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop10)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop10 1.0 ;;
    fullgrid-coreloc-ta-crop12|dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop12)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12 1.2 ;;
    fullgrid-coreloc-ta-crop14|dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop14)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop14 1.4 ;;
    fullgrid-coreloc-ta-crop16|dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop16)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop16 1.6 ;;
    fullgrid-coreloc-ta-crop18|dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop18)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop18 1.8 ;;
    fullgrid-coreloc-ta-crop20|dino-rezero-fullgrid32-coreloc-sam3-touchanything-crop20)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop20 2.0 ;;
    fullgrid-coreloc-ta-res320|dino-rezero-fullgrid32-coreloc-sam3-touchanything-res320)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12_res320 1.2 320x240 ;;
    fullgrid-coreloc-ta-res384|dino-rezero-fullgrid32-coreloc-sam3-touchanything-res384)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12_res384 1.2 384x288 ;;
    fullgrid-coreloc-ta-capacity50|dino-rezero-fullgrid32-coreloc-sam3-touchanything-capacity50)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12_capacity50 1.2
        preset_args+=(--pool_output_channels 50) ;;
    fullgrid-coreloc-ta-capacity72|dino-rezero-fullgrid32-coreloc-sam3-touchanything-capacity72)
        set_ta_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12_capacity72 1.2
        preset_args+=(--pool_output_channels 72) ;;
    weight-plateau2-r256)
        set_ta_variant ta_wplateau2_r256 1.2
        preset_args+=(--pressure_weight_mode plateau --batch_size 128) ;;
    weight-linear3-r256)
        set_ta_variant ta_wlinear3_r256 1.2
        preset_args+=(
            --pressure_weight_mode capped_linear
            --active_pressure_tail_thr 0.70
            --active_pressure_tail_max 3.0
            --batch_size 128
        ) ;;
    *)
        echo "Unknown or retired mode '$MODE'." >&2
        usage >&2
        exit 2 ;;
esac

for manifest in "${required_manifests[@]}"; do
    if [[ ! -f "$manifest" ]]; then
        echo "Reviewed SAM3 bbox manifest is required: $manifest" >&2
        echo "Set SAM3_RECONSTRUCTION_ROOT if the reconstruction lives elsewhere." >&2
        exit 2
    fi
done

exec "$SCRIPT_DIR/run_tactile_ft.sh" "${preset_args[@]}" "$@"
