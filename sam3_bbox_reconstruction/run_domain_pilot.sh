#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/shell_utils.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 opentouch|touchanything|both [run_pilot options]" >&2
  exit 2
fi

MODE="$1"
shift
case "${MODE}" in
  opentouch|ot)
    DATASETS="opentouch"
    DEFAULT_OUTPUT="${ROOT_DIR}/outputs/pilot_opentouch"
    ;;
  touchanything|ta)
    DATASETS="touchanything"
    DEFAULT_OUTPUT="${ROOT_DIR}/outputs/pilot_touchanything"
    ;;
  both)
    DATASETS="opentouch,touchanything"
    DEFAULT_OUTPUT="${ROOT_DIR}/outputs/pilot_opentouch_touchanything"
    ;;
  *)
    echo "Unknown mode ${MODE}; choose opentouch, touchanything, or both." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${DEFAULT_OUTPUT}"
ARGS=("$@")
RUN_ARGS=()
MANIFEST_ONLY=0
ALL_SEQUENCES=0
ASSOCIATION_PREVIEWS="auto"
COMPACT_MANIFESTS="auto"
i=0
while ((i < ${#ARGS[@]})); do
  arg="${ARGS[$i]}"
  case "${arg}" in
    --datasets|--datasets=*)
      echo "run_domain_pilot.sh fixes --datasets from its first mode argument." >&2
      exit 2
      ;;
    --manifest-only)
      MANIFEST_ONLY=1
      RUN_ARGS+=("${arg}")
      ;;
    --all-sequences)
      ALL_SEQUENCES=1
      RUN_ARGS+=("${arg}")
      ;;
    --association-previews)
      ASSOCIATION_PREVIEWS="yes"
      ;;
    --no-association-previews)
      ASSOCIATION_PREVIEWS="no"
      ;;
    --compact-manifests)
      COMPACT_MANIFESTS="yes"
      ;;
    --no-compact-manifests)
      COMPACT_MANIFESTS="no"
      ;;
    --output-dir)
      if ((i + 1 >= ${#ARGS[@]})); then
        echo "--output-dir requires a value." >&2
        exit 2
      fi
      OUTPUT_DIR="${ARGS[$((i + 1))]}"
      RUN_ARGS+=("${arg}" "${OUTPUT_DIR}")
      ((i += 1))
      ;;
    --output-dir=*)
      OUTPUT_DIR="${arg#--output-dir=}"
      RUN_ARGS+=("${arg}")
      ;;
    *)
      RUN_ARGS+=("${arg}")
      ;;
  esac
  ((i += 1))
done

if [[ "${ASSOCIATION_PREVIEWS}" == "auto" ]]; then
  if [[ "${ALL_SEQUENCES}" -eq 1 ]]; then
    ASSOCIATION_PREVIEWS="no"
  else
    ASSOCIATION_PREVIEWS="yes"
  fi
fi
if [[ "${COMPACT_MANIFESTS}" == "auto" ]]; then
  if [[ "${ALL_SEQUENCES}" -eq 1 ]]; then
    COMPACT_MANIFESTS="yes"
  else
    COMPACT_MANIFESTS="no"
  fi
fi

set +e
"${ROOT_DIR}/run_pilot.sh" \
  --datasets "${DATASETS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${RUN_ARGS[@]}"
TRACK_STATUS=$?
set -e

if [[ "${TRACK_STATUS}" -eq 130 ]]; then
  exit 130
fi

if [[ "${TRACK_STATUS}" -ne 0 ]] && \
   { [[ ! -f "${OUTPUT_DIR}/pilot_manifest.jsonl" ]] || [[ ! -d "${OUTPUT_DIR}/results" ]]; }; then
  echo "Tracking failed before a usable manifest/results tree was created; skipping post-processing." >&2
  exit "${TRACK_STATUS}"
fi

if [[ "${MANIFEST_ONLY}" -eq 1 ]]; then
  echo "Manifest only: ${OUTPUT_DIR}/pilot_manifest.jsonl"
  exit 0
fi

ENV_NAME="${SAM3_BBOX_ENV:-sam3bbox}"
CONDA_BIN_OVERRIDE="${CONDA_BIN:-}"
CONDA_BIN="$(resolve_conda_executable "${CONDA_BIN_OVERRIDE}" || true)"
ENV_PREFIX="$(resolve_conda_env_prefix "${CONDA_BIN}" "${ENV_NAME}" || true)"
if [[ -z "${ENV_PREFIX}" ]]; then
  echo "Conda environment ${ENV_NAME} was not found after pilot completion." >&2
  exit 1
fi

REPORT_DIR="${OUTPUT_DIR}/reports/track_quality"
POSTPROCESS_STATUS=0
"${ENV_PREFIX}/bin/python" -u "${ROOT_DIR}/evaluate_tracks.py" \
  --pilot-dir "${OUTPUT_DIR}" \
  --datasets "${DATASETS}" \
  --output-dir "${REPORT_DIR}" || POSTPROCESS_STATUS=$?

ASSOCIATION_PREVIEW_FLAG="--association-previews"
if [[ "${ASSOCIATION_PREVIEWS}" == "no" ]]; then
  ASSOCIATION_PREVIEW_FLAG="--no-association-previews"
fi
COMPACT_MANIFEST_FLAG="--no-compact-manifests"
if [[ "${COMPACT_MANIFESTS}" == "yes" ]]; then
  COMPACT_MANIFEST_FLAG="--compact-manifests"
fi

"${ENV_PREFIX}/bin/python" -u "${ROOT_DIR}/associate_tracks.py" \
  --pilot-dir "${OUTPUT_DIR}" \
  --touchanything-association screen_order \
  "${ASSOCIATION_PREVIEW_FLAG}" \
  "${COMPACT_MANIFEST_FLAG}" \
  --output-dir "${OUTPUT_DIR}/manifests" || POSTPROCESS_STATUS=$?

echo "Pilot: ${OUTPUT_DIR}"
echo "Quality report: ${REPORT_DIR}"
echo "BBox manifests: ${OUTPUT_DIR}/manifests"
if [[ "${DATASETS}" == *touchanything* && "${ASSOCIATION_PREVIEWS}" == "yes" ]]; then
  echo "Association video gallery: ${OUTPUT_DIR}/association_index.html"
fi

if [[ "${TRACK_STATUS}" -ne 0 ]]; then
  echo "Tracking had failures (status ${TRACK_STATUS}); reports and association previews above cover completed jobs." >&2
  exit "${TRACK_STATUS}"
fi
exit "${POSTPROCESS_STATUS}"
