#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
DINO_WEIGHTS="${DINO_WEIGHTS:-$WORKSPACE_DIR/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}"
SAM3_RECONSTRUCTION_ROOT="${SAM3_RECONSTRUCTION_ROOT:-$WORKSPACE_DIR/sam3_bbox_reconstruction/outputs/full_reconstruction_flow}"
TACTILE_BASE_CHECKPOINT="${TACTILE_BASE_CHECKPOINT:-$SCRIPT_DIR/checkpoints/touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12/best_loss.ckpt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -d /home/ma-user/work/cfzhao ]]; then
    DEFAULT_SURFACE_BASIS_RUNTIME_ROOT=/home/ma-user/work/cfzhao/input_prior_full/surface_basis/stage0_4b_s4
else
    DEFAULT_SURFACE_BASIS_RUNTIME_ROOT="$WORKSPACE_DIR/_DATA/runtime_surface_basis/stage0_4b_s4"
fi
SURFACE_BASIS_RUNTIME_ROOT="${SURFACE_BASIS_RUNTIME_ROOT:-$DEFAULT_SURFACE_BASIS_RUNTIME_ROOT}"
SURFACE_BASIS_AUDIT_DIR="${SURFACE_BASIS_AUDIT_DIR:-}"
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
  fullgrid-coreloc-three-domain   OpenTouch + TouchAnything + EgoTactile gloved official train
  fullgrid-coreloc-four-domain    OT + TA + EgoTactile + AceData train; OT/TA validation only
  fullgrid-coreloc-opentouch
  fullgrid-coreloc-touchanything
  fullgrid-coreloc-ta-crop10|crop12|crop14|crop16|crop18|crop20
  fullgrid-coreloc-ta-res320|res384
  fullgrid-coreloc-ta-capacity50|capacity72
  fg-c32-h512|fg-c64-h512|fg-c128-h512|fg-c256-h512
  fg-c32-h1024|fg-c64-h1024|fg-c128-h1024|fg-c256-h1024
                                  FullGrid channel/decoder-width ablation matrix
  weight-plateau2-r256
  weight-linear3-r256
  local-residual-r256             Frozen crop1.2 base + bounded canonical local residual
  selector-contact-r256           Frozen crop1.2 base + independent contact selector
  selector-ordinal-r256           Frozen crop1.2 base + independent ordinal selector
  selector-grid-r256              Contact-specific selector from frozen ReZero grid
  selector-raw-r256               Contact-specific selector from frozen raw DINO levels
  selector-down-r256              Base-conditioned selector for frozen-base false highs
  selector-down-control-r256      Parameter-matched down selector without base confidence
  surface-s4-k4096-r256           Frozen crop1.2 FullGrid + direct 4096-D surface field
  surface-s4-k5120-r256           Frozen crop1.2 FullGrid + direct 5120-D surface field
  surface-s4-k4096-lrhalf-r256    4096-D surface field with half learning rate
  surface-s4-k5120-lrhalf-r256    5120-D surface field with half learning rate
  surface-s4-k4096-scratch-r256   Fresh ReZero/FullGrid + 4096-D surface field
  surface-nl-k4096-scratch-r256   Fresh ReZero/FullGrid + nonlinear 4096-D surface field

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
OT_HDF5_ROOT="${OPENTOUCH_DATA_ROOT:-/home/ma-user/work/cfzhao/OpenTouch Data/full_dataset}"
TA_HDF5_ROOT="${TOUCHANYTHING_DATA_ROOT:-/home/ma-user/work/cfzhao/EgoTouch/extracted_frames}"
EGO_HDF5_ROOT="${EGOTACTILE_DATA_ROOT:-/home/ma-user/work/cfzhao/EgoTactile/Raw_data/extracted_frames_current}"
ACE_HDF5_ROOT="${ACEDATA_DATA_ROOT:-/home/ma-user/work/hy/acedata-processed-hdf5}"
OT_TRAIN_QUERIES="$OT_HDF5_ROOT/manifests/opentouch_train.queries.jsonl"
OT_VAL_QUERIES="$OT_HDF5_ROOT/manifests/opentouch_val.queries.jsonl"
TA_TRAIN_QUERIES="$TA_HDF5_ROOT/manifests/touchanything_train.queries.jsonl"
TA_VAL_QUERIES="$TA_HDF5_ROOT/manifests/touchanything_val.queries.jsonl"
EGO_GLOVED_OBJECT_TRAIN_QUERIES="$EGO_HDF5_ROOT/manifests/official/gloved_object_held_out/train.queries.jsonl"
ACE_TRAIN_QUERIES="$ACE_HDF5_ROOT/manifests/acedata_train.queries.jsonl"
required_manifests=()
preset_args=()
surface_basis_dim=""
surface_basis_path=""
surface_requires_base=0

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

set_three_domain() {
    required_manifests=(
        "$OT_BBOX_MANIFEST"
        "$TA_BBOX_MANIFEST"
        "$OT_TRAIN_QUERIES"
        "$OT_VAL_QUERIES"
        "$TA_TRAIN_QUERIES"
        "$TA_VAL_QUERIES"
        "$EGO_GLOVED_OBJECT_TRAIN_QUERIES"
    )
    preset_args=(
        --exp_name mixed_ot_ta_ego_fullgrid_coreloc
        "${REZERO_COMMON[@]}"
        "${CORELOC_COMMON[@]}"
        --bbox_rescale_factor 1.2
        "${SAM3_INDEX[@]}"
        --data_backend sequence_hdf5
        --data_dir "$OT_HDF5_ROOT,$TA_HDF5_ROOT,$EGO_HDF5_ROOT"
        --datasets ""
        --expected_datasets opentouch,touchanything,egotactile
        --val_expected_datasets opentouch,touchanything
        --query_manifests "$OT_TRAIN_QUERIES,$TA_TRAIN_QUERIES,$EGO_GLOVED_OBJECT_TRAIN_QUERIES"
        --val_query_manifests "$OT_VAL_QUERIES,$TA_VAL_QUERIES"
        --bbox_manifests "$OT_BBOX_MANIFEST,$TA_BBOX_MANIFEST"
        --batch_size 128
    )
}

set_four_domain() {
    required_manifests=(
        "$OT_BBOX_MANIFEST"
        "$TA_BBOX_MANIFEST"
        "$OT_TRAIN_QUERIES"
        "$OT_VAL_QUERIES"
        "$TA_TRAIN_QUERIES"
        "$TA_VAL_QUERIES"
        "$EGO_GLOVED_OBJECT_TRAIN_QUERIES"
        "$ACE_TRAIN_QUERIES"
    )
    preset_args=(
        --exp_name mixed_ot_ta_ego_ace_fullgrid_coreloc
        "${REZERO_COMMON[@]}"
        "${CORELOC_COMMON[@]}"
        --bbox_rescale_factor 1.2
        "${SAM3_INDEX[@]}"
        --data_backend sequence_hdf5
        --data_dir "$OT_HDF5_ROOT,$TA_HDF5_ROOT,$EGO_HDF5_ROOT,$ACE_HDF5_ROOT"
        --datasets ""
        --expected_datasets opentouch,touchanything,egotactile,acedata
        --val_expected_datasets opentouch,touchanything
        --query_manifests "$OT_TRAIN_QUERIES,$TA_TRAIN_QUERIES,$EGO_GLOVED_OBJECT_TRAIN_QUERIES,$ACE_TRAIN_QUERIES"
        --val_query_manifests "$OT_VAL_QUERIES,$TA_VAL_QUERIES"
        --bbox_manifests "$OT_BBOX_MANIFEST,$TA_BBOX_MANIFEST"
        --batch_size 128
    )
}

set_width_variant() {
    local exp_name="$1" projection_channels="$2" hidden_dim="$3"
    set_ta_variant "$exp_name" 1.2
    preset_args+=(
        --pool_output_channels "$projection_channels"
        --decoder_hidden_dim "$hidden_dim"
    )
}

set_surface_variant() {
    local exp_name="$1" coefficient_dim="$2" base_lr="${3:-5e-5}"
    set_ta_variant "$exp_name" 1.2
    surface_basis_dim="$coefficient_dim"
    surface_basis_path="$SURFACE_BASIS_RUNTIME_ROOT/runtime_basis_k${coefficient_dim}_s4.pt"
    surface_requires_base=1
    preset_args+=(
        --tactile_head_type dense_v2_dino_surface_basis
        --init_tactile_checkpoint "$TACTILE_BASE_CHECKPOINT"
        --surface_basis_path "$surface_basis_path"
        --surface_coefficient_dim "$coefficient_dim"
        --surface_coefficient_architecture linear
        --surface_coefficient_hidden_dim 1024
        --surface_target_support_count 4
        --surface_background_probability 0.001
        --freeze_surface_feature_extractor
        --no-save_contact_best
        --lr "$base_lr"
        --batch_size 128
    )
}

set_surface_scratch_variant() {
    local exp_name="$1" coefficient_dim="$2"
    set_ta_variant "$exp_name" 1.2
    surface_basis_dim="$coefficient_dim"
    surface_basis_path="$SURFACE_BASIS_RUNTIME_ROOT/runtime_basis_k${coefficient_dim}_s4.pt"
    surface_requires_base=0
    preset_args+=(
        --tactile_head_type dense_v2_dino_surface_basis
        --surface_basis_path "$surface_basis_path"
        --surface_coefficient_dim "$coefficient_dim"
        --surface_coefficient_architecture linear
        --surface_coefficient_hidden_dim 1024
        --surface_target_support_count 4
        --surface_background_probability 0.001
        --no-freeze_surface_feature_extractor
        --no-save_contact_best
        --lr 5e-5
        --batch_size 128
    )
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
    fullgrid-coreloc-three-domain)
        set_three_domain ;;
    fullgrid-coreloc-four-domain)
        set_four_domain ;;
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
    fg-c32-h512)
        set_width_variant touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12 32 512 ;;
    fg-c64-h512)
        set_width_variant ta_fg_c64_h512 64 512 ;;
    fg-c128-h512)
        set_width_variant ta_fg_c128_h512 128 512 ;;
    fg-c256-h512)
        set_width_variant ta_fg_c256_h512 256 512 ;;
    fg-c32-h1024)
        set_width_variant ta_fg_c32_h1024 32 1024 ;;
    fg-c64-h1024)
        set_width_variant ta_fg_c64_h1024 64 1024 ;;
    fg-c128-h1024)
        set_width_variant ta_fg_c128_h1024 128 1024 ;;
    fg-c256-h1024)
        set_width_variant ta_fg_c256_h1024 256 1024 ;;
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
    local-residual-r256)
        set_ta_variant ta_localres_r256 1.2
        preset_args+=(
            --tactile_head_type dense_v2_dino_local_residual
            --init_tactile_checkpoint "$TACTILE_BASE_CHECKPOINT"
            --freeze_local_residual_base
            --local_anchor_count 512
            --local_anchor_neighbors 4
            --local_logit_delta_max 6.0
            --local_residual_dropout 0.10
            --batch_size 128
        ) ;;
    selector-contact-r256)
        set_ta_variant ta_selector_contact_r256 1.2
        preset_args+=(
            --tactile_head_type dense_v2_dino_support_selector
            --init_tactile_checkpoint "$TACTILE_BASE_CHECKPOINT"
            --support_selector_mode contact
            --support_selector_no_contact_max 0.02
            --support_selector_contact_min 0.10
            --support_selector_dropout 0.10
            --no-save_contact_best
            --batch_size 128
        ) ;;
    selector-ordinal-r256)
        set_ta_variant ta_selector_ordinal_r256 1.2
        preset_args+=(
            --tactile_head_type dense_v2_dino_support_selector
            --init_tactile_checkpoint "$TACTILE_BASE_CHECKPOINT"
            --support_selector_mode ordinal
            --support_selector_thresholds 0.02,0.05,0.10,0.20,0.50
            --support_selector_no_contact_max 0.02
            --support_selector_contact_min 0.10
            --support_selector_dropout 0.10
            --support_selector_monotonicity_weight 0.10
            --no-save_contact_best
            --batch_size 128
        ) ;;
    selector-grid-r256)
        set_ta_variant ta_selector_grid_r256 1.2
        preset_args+=(
            --tactile_head_type dense_v2_dino_support_selector
            --init_tactile_checkpoint "$TACTILE_BASE_CHECKPOINT"
            --support_selector_mode contact
            --support_selector_no_contact_max 0.02
            --support_selector_contact_min 0.10
            --support_selector_dropout 0.10
            --support_selector_architecture spatial_mlp
            --support_selector_feature_source rezero_grid
            --support_selector_neck_channels 64
            --support_selector_hidden_dim 512
            --no-save_contact_best
            --batch_size 128
        ) ;;
    selector-raw-r256)
        set_ta_variant ta_selector_raw_r256 1.2
        preset_args+=(
            --tactile_head_type dense_v2_dino_support_selector
            --init_tactile_checkpoint "$TACTILE_BASE_CHECKPOINT"
            --support_selector_mode contact
            --support_selector_no_contact_max 0.02
            --support_selector_contact_min 0.10
            --support_selector_dropout 0.10
            --support_selector_architecture spatial_mlp
            --support_selector_feature_source raw_dino
            --support_selector_neck_channels 64
            --support_selector_hidden_dim 512
            --no-save_contact_best
            --batch_size 128
        ) ;;
    selector-down-r256)
        set_ta_variant ta_selector_down_r256 1.2
        preset_args+=(
            --tactile_head_type dense_v2_dino_support_selector
            --init_tactile_checkpoint "$TACTILE_BASE_CHECKPOINT"
            --support_selector_mode down_error
            --support_selector_no_contact_max 0.02
            --support_selector_contact_min 0.10
            --support_selector_dropout 0.10
            --support_selector_architecture spatial_mlp
            --support_selector_feature_source rezero_grid
            --support_selector_neck_channels 64
            --support_selector_hidden_dim 512
            --support_selector_base_conditioning real
            --support_selector_correction_min_precision 0.90
            --no-save_contact_best
            --batch_size 128
        ) ;;
    selector-down-control-r256)
        set_ta_variant ta_selector_down_ctl_r256 1.2
        preset_args+=(
            --tactile_head_type dense_v2_dino_support_selector
            --init_tactile_checkpoint "$TACTILE_BASE_CHECKPOINT"
            --support_selector_mode down_error
            --support_selector_no_contact_max 0.02
            --support_selector_contact_min 0.10
            --support_selector_dropout 0.10
            --support_selector_architecture spatial_mlp
            --support_selector_feature_source rezero_grid
            --support_selector_neck_channels 64
            --support_selector_hidden_dim 512
            --support_selector_base_conditioning constant_control
            --support_selector_correction_min_precision 0.90
            --no-save_contact_best
            --batch_size 128
        ) ;;
    surface-s4-k4096-r256)
        set_surface_variant ta_surface_s4_k4096_r256 4096 ;;
    surface-s4-k5120-r256)
        set_surface_variant ta_surface_s4_k5120_r256 5120 ;;
    surface-s4-k4096-lrhalf-r256)
        set_surface_variant ta_surface_s4_k4096_lrhalf_r256 4096 2.5e-5 ;;
    surface-s4-k5120-lrhalf-r256)
        set_surface_variant ta_surface_s4_k5120_lrhalf_r256 5120 2.5e-5 ;;
    surface-s4-k4096-scratch-r256)
        set_surface_scratch_variant ta_surface_s4_k4096_scratch_r256 4096 ;;
    surface-nl-k4096-scratch-r256)
        set_surface_scratch_variant ta_surface_nl_k4096_r256 4096
        preset_args+=(
            --surface_coefficient_architecture nonlinear
            --surface_coefficient_hidden_dim 1024
        ) ;;
    *)
        echo "Unknown or retired mode '$MODE'." >&2
        usage >&2
        exit 2 ;;
esac

if [[ -n "$surface_basis_dim" ]]; then
    if [[ "$surface_requires_base" == 1 && ! -f "$TACTILE_BASE_CHECKPOINT" ]]; then
        echo "Frozen tactile base checkpoint is required: $TACTILE_BASE_CHECKPOINT" >&2
        echo "Set TACTILE_BASE_CHECKPOINT to the crop1.2 loss-best checkpoint." >&2
        exit 2
    fi
    if [[ -z "$SURFACE_BASIS_AUDIT_DIR" ]]; then
        if [[ -d "$SCRIPT_DIR/reports/stage0_4b_basis_density/support_4" ]]; then
            SURFACE_BASIS_AUDIT_DIR="$SCRIPT_DIR/reports/stage0_4b_basis_density/support_4"
        elif [[ -d "$SCRIPT_DIR/stage0_4b_basis_density/support_4" ]]; then
            SURFACE_BASIS_AUDIT_DIR="$SCRIPT_DIR/stage0_4b_basis_density/support_4"
        else
            echo "Could not find the Stage 0.4b support-4 audit." >&2
            echo "Set SURFACE_BASIS_AUDIT_DIR explicitly." >&2
            exit 2
        fi
    fi
    mkdir -p "$SURFACE_BASIS_RUNTIME_ROOT"
    "$PYTHON_BIN" "$SCRIPT_DIR/export_surface_basis_runtime.py" \
        --audit-dir "$SURFACE_BASIS_AUDIT_DIR" \
        --coefficient-dim "$surface_basis_dim" \
        --output "$surface_basis_path"
fi

for manifest in "${required_manifests[@]}"; do
    if [[ ! -f "$manifest" ]]; then
        echo "Required training data manifest is missing: $manifest" >&2
        echo "Set the corresponding data-root environment variable if it lives elsewhere." >&2
        exit 2
    fi
done

if [[ "$MODE" =~ ^(local-residual-r256|selector-contact-r256|selector-ordinal-r256|selector-grid-r256|selector-raw-r256|selector-down-r256|selector-down-control-r256|surface-s4-k4096-r256|surface-s4-k5120-r256|surface-s4-k4096-lrhalf-r256|surface-s4-k5120-lrhalf-r256)$ && ! -f "$TACTILE_BASE_CHECKPOINT" ]]; then
    echo "Frozen tactile base checkpoint is required: $TACTILE_BASE_CHECKPOINT" >&2
    echo "Set TACTILE_BASE_CHECKPOINT to the crop1.2 loss-best checkpoint." >&2
    exit 2
fi

exec "$SCRIPT_DIR/run_tactile_ft.sh" "${preset_args[@]}" "$@"
