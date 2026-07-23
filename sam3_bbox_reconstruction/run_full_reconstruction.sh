#!/usr/bin/env bash
set -euo pipefail

# Bash reads scripts incrementally. A long reconstruction can therefore fail at
# the very end if a shared-filesystem sync replaces this launcher while a child
# process is running. Execute a private snapshot so the orchestration remains
# stable even when the workspace is updated concurrently.
if [[ "${SAM3_FULL_RECONSTRUCTION_SNAPSHOT:-0}" != "1" ]]; then
  SOURCE_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
  SOURCE_ROOT="$(cd "$(dirname "${SOURCE_SCRIPT}")" && pwd)"
  SNAPSHOT="$(mktemp "${TMPDIR:-/tmp}/sam3_full_reconstruction.XXXXXX.sh")"
  cp -- "${SOURCE_SCRIPT}" "${SNAPSHOT}"
  chmod 700 "${SNAPSHOT}"
  export SAM3_FULL_RECONSTRUCTION_SNAPSHOT=1
  export SAM3_FULL_RECONSTRUCTION_ROOT="${SOURCE_ROOT}"
  export SAM3_FULL_RECONSTRUCTION_SNAPSHOT_PATH="${SNAPSHOT}"
  exec /usr/bin/env bash "${SNAPSHOT}" "$@"
fi

ROOT_DIR="${SAM3_FULL_RECONSTRUCTION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SNAPSHOT_PATH="${SAM3_FULL_RECONSTRUCTION_SNAPSHOT_PATH:-}"
cleanup_launcher_snapshot() {
  if [[ -n "${SNAPSHOT_PATH}" && -f "${SNAPSHOT_PATH}" ]]; then
    rm -f -- "${SNAPSHOT_PATH}"
  fi
}
trap cleanup_launcher_snapshot EXIT

source "${ROOT_DIR}/shell_utils.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 opentouch|touchanything|both [--apply|--run-and-apply] [options]" >&2
  exit 2
fi

MODE="$1"
shift
case "${MODE}" in
  opentouch|ot) MODE="opentouch" ;;
  touchanything|ta) MODE="touchanything" ;;
  both) ;;
  *) echo "Unknown mode ${MODE}; choose opentouch, touchanything, or both." >&2; exit 2 ;;
esac

OUTPUT_ROOT="${ROOT_DIR}/outputs/full_reconstruction"
GPUS="0,1,2,3,4,5,6,7"
CHECKPOINT="${SAM3_CHECKPOINT:-/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/sam3/sam3.pt}"
OPENTOUCH_WORKERS=2
TOUCHANYTHING_WORKERS=1
OPENTOUCH_CHUNK_FRAMES=64
OPENTOUCH_CHUNK_OVERLAP=8
TOUCHANYTHING_CHUNK_FRAMES=0
TOUCHANYTHING_CHUNK_OVERLAP=0
CPU_THREADS_PER_WORKER=1
WRITEBACK_PREFLIGHT_WORKERS=64
WRITEBACK_APPLY_WORKERS=32
WRITEBACK_FSYNC=0
APPLY_MODE="none"
EXTRA_ARGS=()

while (($#)); do
  case "$1" in
    --apply)
      APPLY_MODE="only"
      shift
      ;;
    --run-and-apply)
      APPLY_MODE="after"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?--output-root requires a path}"
      shift 2
      ;;
    --gpus)
      GPUS="${2:?--gpus requires a list}"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="${2:?--checkpoint requires a path}"
      shift 2
      ;;
    --opentouch-workers-per-gpu)
      OPENTOUCH_WORKERS="${2:?option requires an integer}"
      shift 2
      ;;
    --touchanything-workers-per-gpu)
      TOUCHANYTHING_WORKERS="${2:?option requires an integer}"
      shift 2
      ;;
    --opentouch-chunk-frames)
      OPENTOUCH_CHUNK_FRAMES="${2:?option requires an integer}"
      shift 2
      ;;
    --opentouch-chunk-overlap)
      OPENTOUCH_CHUNK_OVERLAP="${2:?option requires an integer}"
      shift 2
      ;;
    --touchanything-chunk-frames)
      TOUCHANYTHING_CHUNK_FRAMES="${2:?option requires an integer}"
      shift 2
      ;;
    --touchanything-chunk-overlap)
      TOUCHANYTHING_CHUNK_OVERLAP="${2:?option requires an integer}"
      shift 2
      ;;
    --cpu-threads-per-worker)
      CPU_THREADS_PER_WORKER="${2:?option requires an integer}"
      shift 2
      ;;
    --writeback-workers)
      WRITEBACK_PREFLIGHT_WORKERS="${2:?option requires an integer}"
      WRITEBACK_APPLY_WORKERS="${2:?option requires an integer}"
      shift 2
      ;;
    --writeback-preflight-workers)
      WRITEBACK_PREFLIGHT_WORKERS="${2:?option requires an integer}"
      shift 2
      ;;
    --writeback-apply-workers)
      WRITEBACK_APPLY_WORKERS="${2:?option requires an integer}"
      shift 2
      ;;
    --writeback-fsync-each-file)
      WRITEBACK_FSYNC=1
      shift
      ;;
    --output-dir|--workers-per-gpu|--all-sequences)
      echo "$1 is managed by run_full_reconstruction.sh; use its domain-specific options." >&2
      exit 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${APPLY_MODE}" != "only" && ! -f "${CHECKPOINT}" ]]; then
  echo "SAM3 checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

ENV_NAME="${SAM3_BBOX_ENV:-sam3bbox}"
CONDA_BIN="$(resolve_conda_executable "${CONDA_BIN:-}" || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "Conda was not found; run install_env.sh first." >&2
  exit 1
fi
ENV_PREFIX="$(resolve_conda_env_prefix "${CONDA_BIN}" "${ENV_NAME}" || true)"
if [[ -z "${ENV_PREFIX}" ]]; then
  echo "Conda environment ${ENV_NAME} was not found; run install_env.sh first." >&2
  exit 1
fi
PYTHON="${ENV_PREFIX}/bin/python"

run_domain() {
  local domain="$1"
  local workers="$2"
  local output_dir="${OUTPUT_ROOT}/${domain}"
  local manifest
  local chunk_frames
  local chunk_overlap
  local chunk_cache_flag
  local continuous_state_memory
  if [[ "${domain}" == "opentouch" ]]; then
    manifest="${output_dir}/manifests/opentouch_sam3_v1.jsonl"
    chunk_frames="${OPENTOUCH_CHUNK_FRAMES}"
    chunk_overlap="${OPENTOUCH_CHUNK_OVERLAP}"
    # OpenTouch reuses staged frames across several semantic verifier passes.
    chunk_cache_flag="--cache-staged-chunks"
    continuous_state_memory="native"
  else
    manifest="${output_dir}/manifests/touchanything_sam3_v1_highconf.jsonl"
    chunk_frames="${TOUCHANYTHING_CHUNK_FRAMES}"
    chunk_overlap="${TOUCHANYTHING_CHUNK_OVERLAP}"
    # TouchAnything runs one uninterrupted SAM session. Its input decoder and
    # historical inference state are rolling bounded, so RAM does not grow with T.
    chunk_cache_flag="--no-cache-staged-chunks"
    continuous_state_memory="bounded"
  fi

  if [[ "${APPLY_MODE}" != "only" ]]; then
    echo "[full] ${domain}: output=${output_dir}, workers/GPU=${workers}"
    "${ROOT_DIR}/run_domain_pilot.sh" "${domain}" \
      --all-sequences \
      --output-dir "${output_dir}" \
      --gpus "${GPUS}" \
      --workers-per-gpu "${workers}" \
      --cpu-threads-per-worker "${CPU_THREADS_PER_WORKER}" \
      --checkpoint "${CHECKPOINT}" \
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
      "${EXTRA_ARGS[@]}"
  fi

  if [[ "${APPLY_MODE}" == "none" ]]; then
    return
  fi
  if [[ ! -f "${manifest}" ]]; then
    echo "[full] Cannot apply ${domain}: manifest not found: ${manifest}" >&2
    echo "[full] Run reconstruction without --apply first, then retry --apply." >&2
    return 1
  fi
  echo "[apply-only] Applying reviewed/high-confidence bbox manifest: ${manifest}"
  local writeback_args=(
    --manifest "${manifest}"
    --output-dir "${output_dir}/manifests/writeback"
    --preflight-workers "${WRITEBACK_PREFLIGHT_WORKERS}"
    --apply-workers "${WRITEBACK_APPLY_WORKERS}"
    --allow-missing-samples
    --apply
  )
  if [[ "${WRITEBACK_FSYNC}" -eq 1 ]]; then
    writeback_args+=(--fsync-each-file)
  fi
  "${PYTHON}" -u "${ROOT_DIR}/apply_bbox_manifest.py" \
    "${writeback_args[@]}"
}

case "${MODE}" in
  opentouch) run_domain opentouch "${OPENTOUCH_WORKERS}" ;;
  touchanything) run_domain touchanything "${TOUCHANYTHING_WORKERS}" ;;
  both)
    run_domain opentouch "${OPENTOUCH_WORKERS}"
    run_domain touchanything "${TOUCHANYTHING_WORKERS}"
    ;;
esac

echo "[full] Complete. Reports and manifests are under ${OUTPUT_ROOT}."
