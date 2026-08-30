#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-build}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON="${TACTILE_PYTHON:-/home/ma-user/work/cfzhao/tactile/bin/python}"
MANIFEST="${ACEDATA_SAM3_MANIFEST:-/home/ma-user/work/hy/acedata-sam3-reconstruction-v1/jobs/acedata_gloved_jobs.jsonl}"
PROCESSED_ROOT="${ACEDATA_PROCESSED_ROOT:-/home/ma-user/work/hy/acedata-processed}"
OUTPUT_ROOT="${ACEDATA_DATA_ROOT:-/home/ma-user/work/hy/acedata-processed-hdf5}"
WORKERS="${ACEDATA_HDF5_WORKERS:-8}"
IMAGE_SOURCE="${ACEDATA_IMAGE_SOURCE:-video}"

run_builder() {
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -u "${ROOT_DIR}/preprocess/acedata/build_sequence_hdf5.py" \
    "$@" \
    --manifest "${MANIFEST}" \
    --processed-root "${PROCESSED_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
}

case "${MODE}" in
  build)
    mkdir -p "${OUTPUT_ROOT}"
    exec 9>"${OUTPUT_ROOT}/.build.lock"
    if ! flock -n 9; then
      echo "Another AceData HDF5 build is active: ${OUTPUT_ROOT}/.build.lock" >&2
      exit 1
    fi
    run_builder build --workers "${WORKERS}" --mode resume \
      --image-source "${IMAGE_SOURCE}" "$@"
    ;;
  overwrite)
    mkdir -p "${OUTPUT_ROOT}"
    exec 9>"${OUTPUT_ROOT}/.build.lock"
    if ! flock -n 9; then
      echo "Another AceData HDF5 build is active: ${OUTPUT_ROOT}/.build.lock" >&2
      exit 1
    fi
    run_builder build --workers "${WORKERS}" --mode overwrite \
      --image-source "${IMAGE_SOURCE}" "$@"
    ;;
  verify)
    run_builder build --workers "${WORKERS}" --mode verify \
      --image-source "${IMAGE_SOURCE}" "$@"
    ;;
  manifest)
    run_builder publish --image-source "${IMAGE_SOURCE}" "$@"
    ;;
  status)
    run_builder status "$@"
    ;;
  pilot)
    run_builder build --workers 1 --mode resume --max-sequences 1 --no-publish \
      --image-source "${IMAGE_SOURCE}" "$@"
    ;;
  -h|--help|help)
    cat <<EOF
Usage: $0 build|overwrite|verify|manifest|status|pilot [options]

Defaults:
  source:    ${PROCESSED_ROOT}
  manifest:  ${MANIFEST}
  output:    ${OUTPUT_ROOT}
  workers:   ${WORKERS}
  RGB source: ${IMAGE_SOURCE}

The published dataset is train-only. Missing sensor or SAM3 queries are
excluded; no AceData validation/test manifest is generated.
EOF
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
