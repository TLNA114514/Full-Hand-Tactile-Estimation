#!/usr/bin/env bash
set -euo pipefail

# Keep long orchestration stable if a shared-filesystem sync updates this checkout.
if [[ "${SAM3_RUN_SNAPSHOT:-0}" != "1" ]]; then
  SOURCE_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
  SOURCE_ROOT="$(cd "$(dirname "${SOURCE_SCRIPT}")" && pwd)"
  SNAPSHOT="$(mktemp "${TMPDIR:-/tmp}/sam3_bbox_run.XXXXXX.sh")"
  cp -- "${SOURCE_SCRIPT}" "${SNAPSHOT}"
  chmod 700 "${SNAPSHOT}"
  export SAM3_RUN_SNAPSHOT=1
  export SAM3_RUN_ROOT="${SOURCE_ROOT}"
  export SAM3_RUN_SNAPSHOT_PATH="${SNAPSHOT}"
  exec /usr/bin/env bash "${SNAPSHOT}" "$@"
fi

ROOT_DIR="${SAM3_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SNAPSHOT_PATH="${SAM3_RUN_SNAPSHOT_PATH:-}"

cleanup_launcher_snapshot() {
  if [[ -n "${SNAPSHOT_PATH}" && -f "${SNAPSHOT_PATH}" ]]; then
    rm -f -- "${SNAPSHOT_PATH}"
  fi
}
trap cleanup_launcher_snapshot EXIT

source "${ROOT_DIR}/shell_utils.sh"

usage() {
  cat <<'EOF'
Usage:
  run.sh pilot [run_pilot.py options]
  run.sh domain opentouch|touchanything|both [pilot options]
  run.sh full opentouch|touchanything|both [--apply|--run-and-apply] [options]
  run.sh apply [apply_bbox_manifest.py options]

The legacy run_pilot.sh, run_domain_pilot.sh, and run_full_reconstruction.sh
remain as compatibility wrappers around these commands.
EOF
}

resolve_sam3_python() {
  local env_name="${SAM3_BBOX_ENV:-sam3bbox}"
  local conda_bin env_prefix
  conda_bin="$(resolve_conda_executable "${CONDA_BIN:-}" || true)"
  if [[ -z "${conda_bin}" ]]; then
    echo "Conda was not found. Run ${ROOT_DIR}/install_env.sh first." >&2
    return 1
  fi
  env_prefix="$(resolve_conda_env_prefix "${conda_bin}" "${env_name}" || true)"
  if [[ -z "${env_prefix}" ]]; then
    echo "Conda environment ${env_name} was not found or has no bin/python." >&2
    echo "Run ${ROOT_DIR}/install_env.sh first." >&2
    return 1
  fi
  printf '%s\n' "${env_prefix}/bin/python"
}

command_pilot() {
  local python
  python="$(resolve_sam3_python)"
  "${python}" -u "${ROOT_DIR}/run_pilot.py" "$@"
}

command_apply() {
  local python
  python="$(resolve_sam3_python)"
  "${python}" -u "${ROOT_DIR}/apply_bbox_manifest.py" "$@"
}

command_domain() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: run.sh domain opentouch|touchanything|both [pilot options]" >&2
    return 2
  fi

  local mode="$1"
  shift
  local datasets default_output
  case "${mode}" in
    opentouch|ot)
      datasets="opentouch"
      default_output="${ROOT_DIR}/outputs/pilot_opentouch"
      ;;
    touchanything|ta)
      datasets="touchanything"
      default_output="${ROOT_DIR}/outputs/pilot_touchanything"
      ;;
    both)
      datasets="opentouch,touchanything"
      default_output="${ROOT_DIR}/outputs/pilot_opentouch_touchanything"
      ;;
    *)
      echo "Unknown mode ${mode}; choose opentouch, touchanything, or both." >&2
      return 2
      ;;
  esac

  local output_dir="${default_output}"
  local manifest_only=0
  local all_sequences=0
  local association_previews="auto"
  local compact_manifests="auto"
  local -a args=("$@")
  local -a run_args=()
  local i=0 arg
  while ((i < ${#args[@]})); do
    arg="${args[$i]}"
    case "${arg}" in
      --datasets|--datasets=*)
        echo "run.sh domain fixes --datasets from its domain argument." >&2
        return 2
        ;;
      --manifest-only)
        manifest_only=1
        run_args+=("${arg}")
        ;;
      --all-sequences)
        all_sequences=1
        run_args+=("${arg}")
        ;;
      --association-previews)
        association_previews="yes"
        ;;
      --no-association-previews)
        association_previews="no"
        ;;
      --compact-manifests)
        compact_manifests="yes"
        ;;
      --no-compact-manifests)
        compact_manifests="no"
        ;;
      --output-dir)
        if ((i + 1 >= ${#args[@]})); then
          echo "--output-dir requires a value." >&2
          return 2
        fi
        output_dir="${args[$((i + 1))]}"
        run_args+=("${arg}" "${output_dir}")
        ((i += 1))
        ;;
      --output-dir=*)
        output_dir="${arg#--output-dir=}"
        run_args+=("${arg}")
        ;;
      *)
        run_args+=("${arg}")
        ;;
    esac
    ((i += 1))
  done

  if [[ "${association_previews}" == "auto" ]]; then
    [[ "${all_sequences}" -eq 1 ]] && association_previews="no" || association_previews="yes"
  fi
  if [[ "${compact_manifests}" == "auto" ]]; then
    [[ "${all_sequences}" -eq 1 ]] && compact_manifests="yes" || compact_manifests="no"
  fi

  local track_status
  set +e
  command_pilot --datasets "${datasets}" --output-dir "${output_dir}" "${run_args[@]}"
  track_status=$?
  set -e

  if [[ "${track_status}" -eq 130 ]]; then
    return 130
  fi
  if [[ "${track_status}" -ne 0 ]] && \
     { [[ ! -f "${output_dir}/pilot_manifest.jsonl" ]] || [[ ! -d "${output_dir}/results" ]]; }; then
    echo "Tracking failed before a usable manifest/results tree was created; skipping post-processing." >&2
    return "${track_status}"
  fi
  if [[ "${manifest_only}" -eq 1 ]]; then
    echo "Manifest only: ${output_dir}/pilot_manifest.jsonl"
    return 0
  fi

  local python report_dir postprocess_status=0
  python="$(resolve_sam3_python)"
  report_dir="${output_dir}/reports/track_quality"
  "${python}" -u "${ROOT_DIR}/evaluate_tracks.py" \
    --pilot-dir "${output_dir}" \
    --datasets "${datasets}" \
    --output-dir "${report_dir}" || postprocess_status=$?

  local association_preview_flag="--association-previews"
  local compact_manifest_flag="--no-compact-manifests"
  [[ "${association_previews}" == "no" ]] && association_preview_flag="--no-association-previews"
  [[ "${compact_manifests}" == "yes" ]] && compact_manifest_flag="--compact-manifests"

  "${python}" -u "${ROOT_DIR}/associate_tracks.py" \
    --pilot-dir "${output_dir}" \
    --touchanything-association screen_order \
    "${association_preview_flag}" \
    "${compact_manifest_flag}" \
    --output-dir "${output_dir}/manifests" || postprocess_status=$?

  echo "Pilot: ${output_dir}"
  echo "Quality report: ${report_dir}"
  echo "BBox manifests: ${output_dir}/manifests"
  if [[ "${datasets}" == *touchanything* && "${association_previews}" == "yes" ]]; then
    echo "Association video gallery: ${output_dir}/association_index.html"
  fi
  if [[ "${track_status}" -ne 0 ]]; then
    echo "Tracking had failures (status ${track_status}); reports cover completed jobs." >&2
    return "${track_status}"
  fi
  return "${postprocess_status}"
}

command_full() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: run.sh full opentouch|touchanything|both [--apply|--run-and-apply] [options]" >&2
    return 2
  fi

  local mode="$1"
  shift
  case "${mode}" in
    opentouch|ot) mode="opentouch" ;;
    touchanything|ta) mode="touchanything" ;;
    both) ;;
    *)
      echo "Unknown mode ${mode}; choose opentouch, touchanything, or both." >&2
      return 2
      ;;
  esac

  local output_root="${ROOT_DIR}/outputs/full_reconstruction_flow"
  local gpus="0,1,2,3,4,5,6,7"
  local checkpoint="${SAM3_CHECKPOINT:-/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/sam3/sam3.pt}"
  local opentouch_workers=2
  local touchanything_workers=1
  local opentouch_chunk_frames=64
  local opentouch_chunk_overlap=8
  local touchanything_chunk_frames=0
  local touchanything_chunk_overlap=0
  local cpu_threads_per_worker=1
  local writeback_preflight_workers=64
  local writeback_apply_workers=32
  local writeback_fsync=0
  local apply_mode="none"
  local -a extra_args=()

  while (($#)); do
    case "$1" in
      --apply)
        apply_mode="only"
        shift
        ;;
      --run-and-apply)
        apply_mode="after"
        shift
        ;;
      --output-root)
        output_root="${2:?--output-root requires a path}"
        shift 2
        ;;
      --gpus)
        gpus="${2:?--gpus requires a list}"
        shift 2
        ;;
      --checkpoint)
        checkpoint="${2:?--checkpoint requires a path}"
        shift 2
        ;;
      --opentouch-workers-per-gpu)
        opentouch_workers="${2:?option requires an integer}"
        shift 2
        ;;
      --touchanything-workers-per-gpu)
        touchanything_workers="${2:?option requires an integer}"
        shift 2
        ;;
      --opentouch-chunk-frames)
        opentouch_chunk_frames="${2:?option requires an integer}"
        shift 2
        ;;
      --opentouch-chunk-overlap)
        opentouch_chunk_overlap="${2:?option requires an integer}"
        shift 2
        ;;
      --touchanything-chunk-frames)
        touchanything_chunk_frames="${2:?option requires an integer}"
        shift 2
        ;;
      --touchanything-chunk-overlap)
        touchanything_chunk_overlap="${2:?option requires an integer}"
        shift 2
        ;;
      --cpu-threads-per-worker)
        cpu_threads_per_worker="${2:?option requires an integer}"
        shift 2
        ;;
      --writeback-workers)
        writeback_preflight_workers="${2:?option requires an integer}"
        writeback_apply_workers="${2:?option requires an integer}"
        shift 2
        ;;
      --writeback-preflight-workers)
        writeback_preflight_workers="${2:?option requires an integer}"
        shift 2
        ;;
      --writeback-apply-workers)
        writeback_apply_workers="${2:?option requires an integer}"
        shift 2
        ;;
      --writeback-fsync-each-file)
        writeback_fsync=1
        shift
        ;;
      --output-dir|--workers-per-gpu|--all-sequences)
        echo "$1 is managed by run.sh full; use its domain-specific options." >&2
        return 2
        ;;
      *)
        extra_args+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${apply_mode}" != "only" && ! -f "${checkpoint}" ]]; then
    echo "SAM3 checkpoint not found: ${checkpoint}" >&2
    return 1
  fi
  mkdir -p "${output_root}"

  run_full_domain() {
    local domain="$1"
    local workers="$2"
    local output_dir="${output_root}/${domain}"
    local manifest chunk_frames chunk_overlap chunk_cache_flag continuous_state_memory
    if [[ "${domain}" == "opentouch" ]]; then
      manifest="${output_dir}/manifests/opentouch_sam3_v1.jsonl"
      chunk_frames="${opentouch_chunk_frames}"
      chunk_overlap="${opentouch_chunk_overlap}"
      chunk_cache_flag="--cache-staged-chunks"
      continuous_state_memory="native"
    else
      manifest="${output_dir}/manifests/touchanything_sam3_v1_highconf.jsonl"
      chunk_frames="${touchanything_chunk_frames}"
      chunk_overlap="${touchanything_chunk_overlap}"
      chunk_cache_flag="--no-cache-staged-chunks"
      continuous_state_memory="bounded"
    fi

    if [[ "${apply_mode}" != "only" ]]; then
      echo "[full] ${domain}: output=${output_dir}, workers/GPU=${workers}"
      command_domain "${domain}" \
        --all-sequences \
        --output-dir "${output_dir}" \
        --gpus "${gpus}" \
        --workers-per-gpu "${workers}" \
        --cpu-threads-per-worker "${cpu_threads_per_worker}" \
        --checkpoint "${checkpoint}" \
        --video-chunk-frames "${chunk_frames}" \
        --video-chunk-overlap "${chunk_overlap}" \
        --chunk-carry-sessions 2 \
        --chunk-fragment-reentry \
        --continuous-state-memory "${continuous_state_memory}" \
        --continuous-state-retain-frames 64 \
        --continuous-state-log-interval 256 \
        --continuous-input-cache-frames 4 \
        --offload-state-to-cpu always \
        --chunk-staging-root /dev/shm/sam3_bbox_chunks \
        "${chunk_cache_flag}" \
        --no-mask-previews \
        --no-input-rgb-samples \
        --no-association-previews \
        --compact-manifests \
        "${extra_args[@]}"
    fi

    if [[ "${apply_mode}" == "none" ]]; then
      return 0
    fi
    if [[ ! -f "${manifest}" ]]; then
      echo "[full] Cannot apply ${domain}: manifest not found: ${manifest}" >&2
      echo "[full] Run reconstruction first, then retry with --apply." >&2
      return 1
    fi

    echo "[apply] Applying reviewed/high-confidence bbox manifest: ${manifest}"
    local -a writeback_args=(
      --manifest "${manifest}"
      --output-dir "${output_dir}/manifests/writeback"
      --preflight-workers "${writeback_preflight_workers}"
      --apply-workers "${writeback_apply_workers}"
      --allow-missing-samples
      --apply
    )
    if [[ "${writeback_fsync}" -eq 1 ]]; then
      writeback_args+=(--fsync-each-file)
    fi
    command_apply "${writeback_args[@]}"
  }

  case "${mode}" in
    opentouch) run_full_domain opentouch "${opentouch_workers}" ;;
    touchanything) run_full_domain touchanything "${touchanything_workers}" ;;
    both)
      run_full_domain opentouch "${opentouch_workers}"
      run_full_domain touchanything "${touchanything_workers}"
      ;;
  esac
  echo "[full] Complete. Reports and manifests are under ${output_root}."
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

COMMAND="$1"
shift
case "${COMMAND}" in
  pilot) command_pilot "$@" ;;
  domain) command_domain "$@" ;;
  full) command_full "$@" ;;
  apply) command_apply "$@" ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown command: ${COMMAND}" >&2
    usage >&2
    exit 2
    ;;
esac
