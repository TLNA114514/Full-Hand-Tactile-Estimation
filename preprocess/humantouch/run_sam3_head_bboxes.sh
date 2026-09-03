#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

RAW_ROOT="${HUMANTOUCH_RAW_ROOT:-/home/ma-user/work/hy/humantouch}"
PROCESSED_ROOT="${HUMANTOUCH_PROCESSED_ROOT:-/home/ma-user/work/hy/humantouch-processed}"
RUN_ROOT="${HUMANTOUCH_SAM3_RUN_ROOT:-/home/ma-user/work/hy/humantouch-sam3-head-v1}"
OUTPUT_BBOX_ROOT="${HUMANTOUCH_SAM3_BBOX_ROOT:-${PROCESSED_ROOT}/bboxes_sam3_head_gloved_screen_order_v1}"
SAM_PYTHON="${SAM3_PYTHON:-/home/ma-user/anaconda3/envs/sam3bbox/bin/python}"
TACTILE_PYTHON="${TACTILE_PYTHON:-/home/ma-user/work/cfzhao/tactile/bin/python}"
SAM_CHECKPOINT="${SAM3_CHECKPOINT:-${ROOT_DIR}/_DATA/sam3/sam3.pt}"
GPUS="${SAM3_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${SAM3_WORKERS_PER_GPU:-1}"
TRACK_RETRIES="${SAM3_TRACK_RETRIES:-3}"

JOBS_DIR="${RUN_ROOT}/jobs"
MANIFEST="${JOBS_DIR}/humantouch_head_gloved_jobs.jsonl"
TRACKING_ROOT="${RUN_ROOT}/tracking"
AUDIT_DIR="${RUN_ROOT}/audit"

usage() {
  cat <<EOF
Usage: $0 build|track|audit|materialize|all|status [SAM3 tracking options]

Defaults:
  raw:       ${RAW_ROOT}
  processed: ${PROCESSED_ROOT}
  run:       ${RUN_ROOT}
  bbox:      ${OUTPUT_BBOX_ROOT}
  GPUs:      ${GPUS}

Only observation.images.cam_head MP4 files are discovered. Wrist videos are
rejected by both manifest construction and materialization. This pipeline does
not create HDF5 files or replace an active bbox path.
EOF
}

if [[ "${MODE}" != "status" && "${MODE}" != "help" && "${MODE}" != "-h" && "${MODE}" != "--help" ]]; then
  mkdir -p "${RUN_ROOT}"
  exec 9>"${RUN_ROOT}/pipeline.lock"
  if ! flock -n 9; then
    echo "Another HumanTouch SAM3 pipeline is active: ${RUN_ROOT}/pipeline.lock" >&2
    exit 1
  fi
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

run_adapter() {
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TACTILE_PYTHON}" -u \
    "${ROOT_DIR}/preprocess/humantouch/build_sam3_head_bboxes.py" "$@"
}

build_manifest() {
  mkdir -p "${JOBS_DIR}"
  run_adapter build-manifest \
    --raw-root "${RAW_ROOT}" \
    --processed-root "${PROCESSED_ROOT}" \
    --output "${MANIFEST}"
}

track_all() {
  require_file "${MANIFEST}"
  require_file "${SAM_CHECKPOINT}"
  mkdir -p "${TRACKING_ROOT}"
  local attempt status=1
  for ((attempt = 1; attempt <= TRACK_RETRIES; attempt++)); do
    echo "[HumanTouch SAM3] tracking attempt ${attempt}/${TRACK_RETRIES}"
    set +e
    "${SAM_PYTHON}" -u "${ROOT_DIR}/sam3_bbox_reconstruction/run_pilot.py" \
      --input-manifest "${MANIFEST}" \
      --output-dir "${TRACKING_ROOT}" \
      --all-sequences \
      --gpus "${GPUS}" \
      --workers-per-gpu "${WORKERS_PER_GPU}" \
      --cpu-threads-per-worker 1 \
      --checkpoint "${SAM_CHECKPOINT}" \
      --prompt-preset gloved \
      --max-objects 2 \
      --sam-candidate-capacity 4 \
      --humantouch-semantic-verification-mode off \
      --video-chunk-frames 0 \
      --video-chunk-overlap 0 \
      --continuous-state-memory bounded \
      --continuous-state-retain-frames 64 \
      --continuous-state-log-interval 256 \
      --continuous-input-cache-frames 4 \
      --offload-state-to-cpu always \
      --no-cache-staged-chunks \
      --no-mask-previews \
      --no-input-rgb-samples \
      "$@"
    status=$?
    set -e
    if [[ "${status}" -eq 0 || "${status}" -eq 130 ]]; then
      return "${status}"
    fi
    if ((attempt < TRACK_RETRIES)); then
      echo "[HumanTouch SAM3] retrying unfinished episodes in 15 seconds..." >&2
      sleep 15
    fi
  done
  return "${status}"
}

audit_all() {
  require_file "${MANIFEST}"
  run_adapter audit \
    --manifest "${MANIFEST}" \
    --tracking-root "${TRACKING_ROOT}" \
    --output-dir "${AUDIT_DIR}"
}

materialize_all() {
  require_file "${MANIFEST}"
  run_adapter materialize \
    --manifest "${MANIFEST}" \
    --tracking-root "${TRACKING_ROOT}" \
    --output-bbox-root "${OUTPUT_BBOX_ROOT}" \
    --audit-dir "${AUDIT_DIR}"
}

show_status() {
  echo "Run root: ${RUN_ROOT}"
  echo "Manifest: ${MANIFEST}"
  echo "Tracking root: ${TRACKING_ROOT}"
  echo "SAM3 bbox root: ${OUTPUT_BBOX_ROOT}"
  local expected=0 completed=0 materialized=0
  [[ -f "${MANIFEST}" ]] && expected="$(wc -l < "${MANIFEST}")"
  if [[ -d "${TRACKING_ROOT}/results/humantouch/all" ]]; then
    completed="$(find "${TRACKING_ROOT}/results/humantouch/all" -mindepth 2 -maxdepth 2 -name summary.json -type f -print 2>/dev/null | wc -l)"
  fi
  if [[ -d "${OUTPUT_BBOX_ROOT}" ]]; then
    materialized="$(find "${OUTPUT_BBOX_ROOT}" -mindepth 2 -maxdepth 2 -name 'episode_*.json' -type f -print 2>/dev/null | wc -l)"
  fi
  echo "Tracker summaries: ${completed}/${expected}"
  echo "Materialized bbox files: ${materialized}/${expected}"
  if [[ -f "${MANIFEST%.jsonl}.summary.json" ]]; then
    echo "Manifest summary: ${MANIFEST%.jsonl}.summary.json"
  fi
  if [[ -f "${AUDIT_DIR}/summary.json" ]]; then
    echo "Audit summary: ${AUDIT_DIR}/summary.json"
  fi
  if [[ -f "${OUTPUT_BBOX_ROOT}/.sam3_bbox_status.json" ]]; then
    echo "BBox status: ${OUTPUT_BBOX_ROOT}/.sam3_bbox_status.json"
  fi
}

case "${MODE}" in
  build) build_manifest ;;
  track) track_all "$@" ;;
  audit) audit_all ;;
  materialize) materialize_all ;;
  all)
    build_manifest
    track_status=0
    track_all "$@" || track_status=$?
    if [[ "${track_status}" -eq 130 ]]; then
      exit 130
    fi
    materialize_all
    exit "${track_status}"
    ;;
  status) show_status ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
