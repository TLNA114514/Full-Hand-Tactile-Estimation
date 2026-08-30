#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

RAW_ROOT="${EGOTACTILE_RAW_ROOT:-/home/ma-user/work/cfzhao/EgoTactile/Raw_data}"
SOURCE_HDF5_ROOT="${EGOTACTILE_SOURCE_HDF5_ROOT:-${RAW_ROOT}/extracted_frames}"
OUTPUT_HDF5_ROOT="${EGOTACTILE_SAM3_HDF5_ROOT:-${RAW_ROOT}/extracted_frames_sam3}"
ACTIVE_LINK="${EGOTACTILE_ACTIVE_HDF5_ROOT:-${RAW_ROOT}/extracted_frames_current}"
RUN_ROOT="${EGOTACTILE_SAM3_RUN_ROOT:-/home/ma-user/work/cfzhao/EgoTactile/sam3_bbox_reconstruction_v1}"
SAM_PYTHON="${SAM3_PYTHON:-/home/ma-user/anaconda3/envs/sam3bbox/bin/python}"
TACTILE_PYTHON="${TACTILE_PYTHON:-/home/ma-user/work/cfzhao/tactile/bin/python}"
SAM_CHECKPOINT="${SAM3_CHECKPOINT:-${ROOT_DIR}/_DATA/sam3/sam3.pt}"
GPUS="${SAM3_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${SAM3_WORKERS_PER_GPU:-1}"
MIGRATION_WORKERS="${EGOTACTILE_MIGRATION_WORKERS:-8}"
EXPECTED_CLIPS="${EGOTACTILE_EXPECTED_CLIPS:-767}"
TRACK_RETRIES="${SAM3_TRACK_RETRIES:-3}"

JOBS_DIR="${RUN_ROOT}/jobs"
BARE_MANIFEST="${JOBS_DIR}/egotactile_bare_hand_jobs.jsonl"
GLOVED_MANIFEST="${JOBS_DIR}/egotactile_gloved_hand_jobs.jsonl"
BARE_TRACKING="${RUN_ROOT}/tracking/bare_hand"
GLOVED_TRACKING="${RUN_ROOT}/tracking/gloved_hand"
AUDIT_DIR="${RUN_ROOT}/audit"

if [[ "${MODE}" != "status" && "${MODE}" != "help" && "${MODE}" != "-h" && "${MODE}" != "--help" ]]; then
  mkdir -p "${RUN_ROOT}"
  exec 9>"${RUN_ROOT}/pipeline.lock"
  if ! flock -n 9; then
    echo "Another EgoTactile SAM3 pipeline is already active: ${RUN_ROOT}/pipeline.lock" >&2
    exit 1
  fi
fi

usage() {
  cat <<EOF
Usage: $0 build|track|audit|materialize|official-splits|activate|all|status

Environment overrides:
  SAM3_GPUS=${GPUS}
  SAM3_WORKERS_PER_GPU=${WORKERS_PER_GPU}
  EGOTACTILE_SAM3_RUN_ROOT=${RUN_ROOT}
  EGOTACTILE_SAM3_HDF5_ROOT=${OUTPUT_HDF5_ROOT}
  EGOTACTILE_ACTIVE_HDF5_ROOT=${ACTIVE_LINK}
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

build_manifests() {
  mkdir -p "${JOBS_DIR}"
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TACTILE_PYTHON}" -u "${ROOT_DIR}/preprocess/egotactile/rebuild_sam3_bboxes.py" \
      build-manifests \
      --raw-root "${RAW_ROOT}" \
      --hdf5-root "${SOURCE_HDF5_ROOT}" \
      --output-dir "${JOBS_DIR}" \
      --expected-clips "${EXPECTED_CLIPS}"
}

run_tracking_manifest() {
  local manifest="$1"
  local output_dir="$2"
  local preset="$3"
  shift 3
  require_file "${manifest}"
  mkdir -p "${output_dir}"
  local attempt status=1
  for ((attempt = 1; attempt <= TRACK_RETRIES; attempt++)); do
    echo "[EgoTactile SAM3] ${preset} attempt ${attempt}/${TRACK_RETRIES}"
    set +e
    "${SAM_PYTHON}" -u "${ROOT_DIR}/sam3_bbox_reconstruction/run_pilot.py" \
      --input-manifest "${manifest}" \
      --output-dir "${output_dir}" \
      --gpus "${GPUS}" \
      --workers-per-gpu "${WORKERS_PER_GPU}" \
      --cpu-threads-per-worker 1 \
      --checkpoint "${SAM_CHECKPOINT}" \
      --prompt-preset "${preset}" \
      --max-objects 1 \
      --sam-candidate-capacity 4 \
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
      echo "[EgoTactile SAM3] retrying unfinished jobs in 15 seconds..." >&2
      sleep 15
    fi
  done
  return "${status}"
}

track_all() {
  require_file "${SAM_CHECKPOINT}"
  run_tracking_manifest "${BARE_MANIFEST}" "${BARE_TRACKING}" bare
  run_tracking_manifest "${GLOVED_MANIFEST}" "${GLOVED_TRACKING}" gloved
}

audit_all() {
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TACTILE_PYTHON}" -u "${ROOT_DIR}/preprocess/egotactile/rebuild_sam3_bboxes.py" \
      audit \
      --manifests "${BARE_MANIFEST}" "${GLOVED_MANIFEST}" \
      --tracking-roots "${BARE_TRACKING}" "${GLOVED_TRACKING}" \
      --output-dir "${AUDIT_DIR}" \
      --expected-clips "${EXPECTED_CLIPS}"
}

materialize_all() {
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TACTILE_PYTHON}" -u "${ROOT_DIR}/preprocess/egotactile/rebuild_sam3_bboxes.py" \
      materialize \
      --manifests "${BARE_MANIFEST}" "${GLOVED_MANIFEST}" \
      --tracking-roots "${BARE_TRACKING}" "${GLOVED_TRACKING}" \
      --source-hdf5-root "${SOURCE_HDF5_ROOT}" \
      --output-root "${OUTPUT_HDF5_ROOT}" \
      --audit-dir "${AUDIT_DIR}" \
      --workers "${MIGRATION_WORKERS}" \
      --expected-clips "${EXPECTED_CLIPS}"
}

build_official_splits() {
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TACTILE_PYTHON}" -u "${ROOT_DIR}/preprocess/egotactile/rebuild_sam3_bboxes.py" \
      official-splits \
      --output-root "${OUTPUT_HDF5_ROOT}"
}

activate_all() {
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TACTILE_PYTHON}" -u "${ROOT_DIR}/preprocess/egotactile/rebuild_sam3_bboxes.py" \
      activate \
      --output-root "${OUTPUT_HDF5_ROOT}" \
      --active-link "${ACTIVE_LINK}" \
      --expected-clips "${EXPECTED_CLIPS}"
}

show_status() {
  echo "Run root: ${RUN_ROOT}"
  echo "Source HDF5: ${SOURCE_HDF5_ROOT}"
  echo "SAM3 HDF5: ${OUTPUT_HDF5_ROOT}"
  echo "Active path: ${ACTIVE_LINK}"
  for split in bare_hand gloved_hand; do
    local root="${RUN_ROOT}/tracking/${split}/results/egotactile/${split}"
    local completed=0
    if [[ -d "${root}" ]]; then
      completed="$(find "${root}" -mindepth 2 -maxdepth 2 -name summary.json -type f -print 2>/dev/null | wc -l)"
    fi
    echo "${split}: ${completed} tracker summaries"
  done
  if [[ -f "${AUDIT_DIR}/summary.json" ]]; then
    echo "Audit: ${AUDIT_DIR}/summary.json"
  fi
  if [[ -f "${OUTPUT_HDF5_ROOT}/.sam3_bbox_complete.json" ]]; then
    echo "HDF5 migration: complete"
  else
    echo "HDF5 migration: incomplete/not started"
  fi
  if [[ -L "${ACTIVE_LINK}" ]]; then
    echo "Active target: $(readlink -f "${ACTIVE_LINK}")"
  fi
  if [[ -f "${OUTPUT_HDF5_ROOT}/manifests/official/index.json" ]]; then
    echo "Official splits: ${OUTPUT_HDF5_ROOT}/manifests/official/index.json"
  else
    echo "Official splits: missing"
  fi
}

case "${MODE}" in
  build) build_manifests ;;
  track) track_all ;;
  audit) audit_all ;;
  materialize) materialize_all ;;
  official-splits) build_official_splits ;;
  activate) activate_all ;;
  all)
    build_manifests
    track_all
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
