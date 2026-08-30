#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

RAW_ROOT="${ACEDATA_RAW_ROOT:-/home/ma-user/work/hy/acedata-tactile}"
PROCESSED_ROOT="${ACEDATA_PROCESSED_ROOT:-/home/ma-user/work/hy/acedata-processed}"
RUN_ROOT="${ACEDATA_SAM3_RUN_ROOT:-/home/ma-user/work/hy/acedata-sam3-reconstruction-v1}"
OUTPUT_BBOX_ROOT="${ACEDATA_SAM3_BBOX_ROOT:-${PROCESSED_ROOT}/bboxes_sam3_gloved_screen_order_v1}"
BACKUP_BBOX_ROOT="${ACEDATA_OLD_BBOX_ROOT:-${PROCESSED_ROOT}/bboxes_vitdet_kcf_v1}"
SAM_PYTHON="${SAM3_PYTHON:-/home/ma-user/anaconda3/envs/sam3bbox/bin/python}"
TACTILE_PYTHON="${TACTILE_PYTHON:-/home/ma-user/work/cfzhao/tactile/bin/python}"
SAM_CHECKPOINT="${SAM3_CHECKPOINT:-${ROOT_DIR}/_DATA/sam3/sam3.pt}"
GPUS="${SAM3_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${SAM3_WORKERS_PER_GPU:-1}"
EXPECTED_CLIPS="${ACEDATA_EXPECTED_CLIPS:-494}"
TRACK_RETRIES="${SAM3_TRACK_RETRIES:-3}"

JOBS_DIR="${RUN_ROOT}/jobs"
MANIFEST="${JOBS_DIR}/acedata_gloved_jobs.jsonl"
TRACKING_ROOT="${RUN_ROOT}/tracking"
AUDIT_DIR="${RUN_ROOT}/audit"

usage() {
  cat <<EOF
Usage: $0 build|track|audit|materialize|activate|all|status

Defaults:
  raw:       ${RAW_ROOT}
  processed: ${PROCESSED_ROOT}
  run:       ${RUN_ROOT}
  output:    ${OUTPUT_BBOX_ROOT}
  GPUs:      ${GPUS}

All videos use the gloved prompt. The first reliable screen-left/right tracks
are assigned to left/right. Missing or ambiguous frames remain without boxes.
EOF
}

if [[ "${MODE}" != "status" && "${MODE}" != "help" && "${MODE}" != "-h" && "${MODE}" != "--help" ]]; then
  mkdir -p "${RUN_ROOT}"
  exec 9>"${RUN_ROOT}/pipeline.lock"
  if ! flock -n 9; then
    echo "Another AceData SAM3 pipeline is active: ${RUN_ROOT}/pipeline.lock" >&2
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
    "${TACTILE_PYTHON}" -u "${ROOT_DIR}/preprocess/acedata/rebuild_sam3_bboxes.py" "$@"
}

build_manifest() {
  mkdir -p "${JOBS_DIR}"
  run_adapter build-manifest \
    --raw-root "${RAW_ROOT}" \
    --processed-root "${PROCESSED_ROOT}" \
    --output "${MANIFEST}" \
    --expected-clips "${EXPECTED_CLIPS}"
}

track_all() {
  require_file "${MANIFEST}"
  require_file "${SAM_CHECKPOINT}"
  mkdir -p "${TRACKING_ROOT}"
  local attempt status=1
  for ((attempt = 1; attempt <= TRACK_RETRIES; attempt++)); do
    echo "[AceData SAM3] tracking attempt ${attempt}/${TRACK_RETRIES}"
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
      --acedata-semantic-verification-mode off \
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
    if [[ "${status}" -eq 0 ]]; then
      return 0
    fi
    if [[ "${status}" -eq 130 ]]; then
      return 130
    fi
    if ((attempt < TRACK_RETRIES)); then
      echo "[AceData SAM3] retrying unfinished clips in 15 seconds..." >&2
      sleep 15
    fi
  done
  return "${status}"
}

audit_all() {
  run_adapter audit \
    --manifest "${MANIFEST}" \
    --tracking-root "${TRACKING_ROOT}" \
    --output-dir "${AUDIT_DIR}" \
    --expected-clips "${EXPECTED_CLIPS}"
}

materialize_all() {
  run_adapter materialize \
    --manifest "${MANIFEST}" \
    --tracking-root "${TRACKING_ROOT}" \
    --output-bbox-root "${OUTPUT_BBOX_ROOT}" \
    --audit-dir "${AUDIT_DIR}" \
    --expected-clips "${EXPECTED_CLIPS}"
}

activate_all() {
  run_adapter activate \
    --processed-root "${PROCESSED_ROOT}" \
    --output-bbox-root "${OUTPUT_BBOX_ROOT}" \
    --backup-bbox-root "${BACKUP_BBOX_ROOT}" \
    --expected-clips "${EXPECTED_CLIPS}"
}

show_status() {
  echo "Run root: ${RUN_ROOT}"
  echo "Manifest: ${MANIFEST}"
  echo "Tracking root: ${TRACKING_ROOT}"
  echo "SAM3 bbox root: ${OUTPUT_BBOX_ROOT}"
  local completed=0
  if [[ -d "${TRACKING_ROOT}/results/acedata/all" ]]; then
    completed="$(find "${TRACKING_ROOT}/results/acedata/all" -mindepth 2 -maxdepth 2 -name summary.json -type f -print 2>/dev/null | wc -l)"
  fi
  echo "Tracker summaries: ${completed}/${EXPECTED_CLIPS}"
  [[ -f "${AUDIT_DIR}/summary.json" ]] && echo "Audit: ${AUDIT_DIR}/summary.json"
  [[ -f "${OUTPUT_BBOX_ROOT}/.sam3_bbox_complete.json" ]] && echo "Materialization: complete"
  if [[ -L "${PROCESSED_ROOT}/bboxes" ]]; then
    echo "Active bbox target: $(readlink -f "${PROCESSED_ROOT}/bboxes")"
  else
    echo "Active bbox target: legacy directory"
  fi
}

case "${MODE}" in
  build) build_manifest ;;
  track) track_all "$@" ;;
  audit) audit_all ;;
  materialize) materialize_all ;;
  activate) activate_all ;;
  all)
    build_manifest
    track_all "$@"
    audit_all
    materialize_all
    activate_all
    ;;
  status) show_status ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
