#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/sam3_bbox_reconstruction/shell_utils.sh"

VIDEO_PATH=""
TACTILE_CHECKPOINT=""
OUT_DIR="${ROOT_DIR}/demo_output"
GPU="0"
GPUS=""
HAND="right"
DINO_WEIGHTS="${ROOT_DIR}/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-}"
SAM3_PROMPT_PRESET="gloved"
SAM3_PROMPT=""
PROMPT_FRAME="auto"
MEDIAPIPE_PROMPT_SAMPLES="48"
EXPECTED_HANDS="1"
SAM3_TRACK_ID=""
LEFT_SAM3_TRACK_ID=""
RIGHT_SAM3_TRACK_ID=""
VIDEO_ROTATION="auto"
DISPLAY_FLOOR="0.05"
TEMPORAL_ALPHA="0.4"
TACTILE_RENDER_SIZE="720x1280"
COMBINED_LAYOUT="horizontal"
SKIP_FRAME_EXTRACTION=0
FORCE_SAM=0

usage() {
  cat <<'EOF'
Usage: ./hamer_tactile_ft/run_demo_sam3_bbox.sh --video_path VIDEO --checkpoint CKPT [options]

Required:
  --video_path PATH             Source video
  --checkpoint PATH             Compact tactile checkpoint

Options:
  --out_dir PATH                Demo output root (default: ./demo_output)
  --gpu ID                      SAM3 GPU and single-hand tactile GPU (default: 0)
  --gpus LEFT_GPU,RIGHT_GPU     Dual-hand tactile GPUs, ordered left then right
  --hand left|right|both        Tactile target handedness (default: right)
  --dino_weights PATH           Local DINOv3 H+/16 weights
  --sam3_checkpoint PATH        Local SAM3 checkpoint
  --prompt_preset gloved|bare   SAM3 text-prompt preset (default: gloved)
  --prompt TEXT                 Override the preset's primary SAM prompt
  --prompt_frame auto|INDEX     SAM anchor frame; auto optionally uses MediaPipe (default: auto)
  --mediapipe_prompt_samples N  Frames sampled only to choose the SAM anchor (default: 48)
  --expected_hands 1|2          Maximum accepted anonymous SAM tracks (default: 1)
  --sam3_track_id ID            Required for tactile inference if two tracks survive
  --left_sam3_track_id ID       SAM3 track bound to the left hand in --hand both mode
  --right_sam3_track_id ID      SAM3 track bound to the right hand in --hand both mode
  --video_rotation auto|0|90|180|270
  --display_floor FLOAT
  --temporal_alpha FLOAT
  --tactile_render_size WxH
  --combined_layout horizontal|vertical|auto
  --skip_frame_extraction       Reuse existing orientation-correct RGB frames/video
  --force_sam                   Rerun SAM3 even when the same completed output exists

Environment:
  TACTILE_ENV=tactile           Conda environment containing the tactile stack
  SAM3_BBOX_ENV=sam3bbox        Conda environment created by install_env.sh
  CONDA_BIN=/path/to/conda      Optional Conda executable override
EOF
}

while (($#)); do
  case "$1" in
    --video_path|--video-path) VIDEO_PATH="${2:?option requires a path}"; shift 2 ;;
    --checkpoint) TACTILE_CHECKPOINT="${2:?option requires a path}"; shift 2 ;;
    --out_dir|--out-dir) OUT_DIR="${2:?option requires a path}"; shift 2 ;;
    --gpu) GPU="${2:?option requires an ID}"; shift 2 ;;
    --gpus) GPUS="${2:?option requires LEFT_GPU,RIGHT_GPU}"; shift 2 ;;
    --hand) HAND="${2:?option requires left, right, or both}"; shift 2 ;;
    --dino_weights|--dino-weights) DINO_WEIGHTS="${2:?option requires a path}"; shift 2 ;;
    --sam3_checkpoint|--sam3-checkpoint) SAM3_CHECKPOINT="${2:?option requires a path}"; shift 2 ;;
    --prompt_preset|--prompt-preset) SAM3_PROMPT_PRESET="${2:?option requires gloved or bare}"; shift 2 ;;
    --prompt) SAM3_PROMPT="${2:?option requires text}"; shift 2 ;;
    --prompt_frame|--prompt-frame) PROMPT_FRAME="${2:?option requires auto or an integer}"; shift 2 ;;
    --mediapipe_prompt_samples|--mediapipe-prompt-samples) MEDIAPIPE_PROMPT_SAMPLES="${2:?option requires an integer}"; shift 2 ;;
    --expected_hands|--expected-hands) EXPECTED_HANDS="${2:?option requires 1 or 2}"; shift 2 ;;
    --sam3_track_id|--sam3-track-id) SAM3_TRACK_ID="${2:?option requires an integer}"; shift 2 ;;
    --left_sam3_track_id|--left-sam3-track-id) LEFT_SAM3_TRACK_ID="${2:?option requires an integer}"; shift 2 ;;
    --right_sam3_track_id|--right-sam3-track-id) RIGHT_SAM3_TRACK_ID="${2:?option requires an integer}"; shift 2 ;;
    --video_rotation|--video-rotation) VIDEO_ROTATION="${2:?option requires a value}"; shift 2 ;;
    --display_floor|--display-floor) DISPLAY_FLOOR="${2:?option requires a value}"; shift 2 ;;
    --temporal_alpha|--temporal-alpha) TEMPORAL_ALPHA="${2:?option requires a value}"; shift 2 ;;
    --tactile_render_size|--tactile-render-size) TACTILE_RENDER_SIZE="${2:?option requires WxH}"; shift 2 ;;
    --combined_layout|--combined-layout) COMBINED_LAYOUT="${2:?option requires a value}"; shift 2 ;;
    --skip_frame_extraction|--skip-frame-extraction) SKIP_FRAME_EXTRACTION=1; shift ;;
    --force_sam|--force-sam) FORCE_SAM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${VIDEO_PATH}" ]] || { echo "--video_path is required" >&2; exit 2; }
[[ -n "${TACTILE_CHECKPOINT}" ]] || { echo "--checkpoint is required" >&2; exit 2; }
[[ "${HAND}" == "left" || "${HAND}" == "right" || "${HAND}" == "both" ]] || {
  echo "--hand must be left, right, or both" >&2
  exit 2
}
[[ "${EXPECTED_HANDS}" == "1" || "${EXPECTED_HANDS}" == "2" ]] || { echo "--expected_hands must be 1 or 2" >&2; exit 2; }
[[ "${SAM3_PROMPT_PRESET}" == "gloved" || "${SAM3_PROMPT_PRESET}" == "bare" ]] || {
  echo "--prompt_preset must be gloved or bare" >&2
  exit 2
}
[[ "${PROMPT_FRAME}" == "auto" || "${PROMPT_FRAME}" =~ ^[0-9]+$ ]] || {
  echo "--prompt_frame must be auto or a non-negative integer" >&2
  exit 2
}
[[ "${MEDIAPIPE_PROMPT_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--mediapipe_prompt_samples must be a positive integer" >&2
  exit 2
}

LEFT_GPU=""
RIGHT_GPU=""
if [[ "${HAND}" == "both" ]]; then
  [[ -n "${GPUS}" ]] || {
    echo "--hand both requires --gpus LEFT_GPU,RIGHT_GPU" >&2
    exit 2
  }
  IFS=',' read -r LEFT_GPU RIGHT_GPU EXTRA_GPU <<< "${GPUS}"
  LEFT_GPU="${LEFT_GPU//[[:space:]]/}"
  RIGHT_GPU="${RIGHT_GPU//[[:space:]]/}"
  [[ -n "${LEFT_GPU}" && -n "${RIGHT_GPU}" && -z "${EXTRA_GPU:-}" ]] || {
    echo "--gpus must contain exactly two IDs ordered as LEFT_GPU,RIGHT_GPU" >&2
    exit 2
  }
  [[ "${LEFT_GPU}" != "${RIGHT_GPU}" ]] || {
    echo "Dual-hand parallel inference requires two distinct GPU IDs" >&2
    exit 2
  }
  [[ "${LEFT_GPU}" =~ ^[0-9]+$ && "${RIGHT_GPU}" =~ ^[0-9]+$ ]] || {
    echo "--gpus values must be non-negative integer GPU IDs" >&2
    exit 2
  }
  [[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "--gpu must be a non-negative SAM3 GPU ID" >&2; exit 2; }
  if [[ -n "${LEFT_SAM3_TRACK_ID}" || -n "${RIGHT_SAM3_TRACK_ID}" ]]; then
    [[ "${LEFT_SAM3_TRACK_ID}" =~ ^[0-9]+$ && "${RIGHT_SAM3_TRACK_ID}" =~ ^[0-9]+$ ]] || {
      echo "Both --left_sam3_track_id and --right_sam3_track_id must be non-negative integers" >&2
      exit 2
    }
    [[ "${LEFT_SAM3_TRACK_ID}" != "${RIGHT_SAM3_TRACK_ID}" ]] || {
      echo "Left and right hands must use different SAM3 track IDs" >&2
      exit 2
    }
  fi
  [[ -z "${SAM3_TRACK_ID}" ]] || {
    echo "Use --left_sam3_track_id and --right_sam3_track_id with --hand both" >&2
    exit 2
  }
  EXPECTED_HANDS="2"
else
  [[ -z "${GPUS}" ]] || {
    echo "--gpus is only valid with --hand both; use --gpu for one hand" >&2
    exit 2
  }
  [[ -z "${LEFT_SAM3_TRACK_ID}" && -z "${RIGHT_SAM3_TRACK_ID}" ]] || {
    echo "Left/right track options require --hand both" >&2
    exit 2
  }
  [[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "--gpu must be a non-negative integer" >&2; exit 2; }
  [[ -z "${SAM3_TRACK_ID}" || "${SAM3_TRACK_ID}" =~ ^[0-9]+$ ]] || {
    echo "--sam3_track_id must be a non-negative integer" >&2
    exit 2
  }
fi

[[ -f "${VIDEO_PATH}" ]] || { echo "Video not found: ${VIDEO_PATH}" >&2; exit 1; }
[[ -f "${TACTILE_CHECKPOINT}" ]] || { echo "Tactile checkpoint not found: ${TACTILE_CHECKPOINT}" >&2; exit 1; }
[[ -f "${DINO_WEIGHTS}" ]] || { echo "DINO weights not found: ${DINO_WEIGHTS}" >&2; exit 1; }
VIDEO_PATH="$(readlink -f "${VIDEO_PATH}")"
TACTILE_CHECKPOINT="$(readlink -f "${TACTILE_CHECKPOINT}")"
DINO_WEIGHTS="$(readlink -f "${DINO_WEIGHTS}")"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(readlink -f "${OUT_DIR}")"

if [[ -z "${SAM3_CHECKPOINT}" ]]; then
  for candidate in \
    "${ROOT_DIR}/_DATA/sam3/sam3.pt" \
    "${ROOT_DIR}/_DATA/sam3.pt"; do
    if [[ -f "${candidate}" ]]; then
      SAM3_CHECKPOINT="${candidate}"
      break
    fi
  done
fi
[[ -n "${SAM3_CHECKPOINT}" ]] || {
  echo "SAM3 checkpoint was not found; pass --sam3_checkpoint PATH" >&2
  exit 1
}
[[ -f "${SAM3_CHECKPOINT}" ]] || { echo "SAM3 checkpoint not found: ${SAM3_CHECKPOINT}" >&2; exit 1; }
SAM3_CHECKPOINT="$(readlink -f "${SAM3_CHECKPOINT}")"

CONDA_EXECUTABLE="$(resolve_conda_executable "${CONDA_BIN:-}" || true)"
[[ -n "${CONDA_EXECUTABLE}" ]] || { echo "Conda not found; set CONDA_BIN" >&2; exit 1; }
TACTILE_ENV_NAME="${TACTILE_ENV:-tactile}"
SAM3_ENV_NAME="${SAM3_BBOX_ENV:-sam3bbox}"
TACTILE_PREFIX="$(resolve_conda_env_prefix "${CONDA_EXECUTABLE}" "${TACTILE_ENV_NAME}" || true)"
SAM3_PREFIX="$(resolve_conda_env_prefix "${CONDA_EXECUTABLE}" "${SAM3_ENV_NAME}" || true)"
[[ -n "${TACTILE_PREFIX}" ]] || { echo "Conda env not found: ${TACTILE_ENV_NAME}" >&2; exit 1; }
[[ -n "${SAM3_PREFIX}" ]] || {
  echo "Conda env not found: ${SAM3_ENV_NAME}; run sam3_bbox_reconstruction/install_env.sh first" >&2
  exit 1
}
TACTILE_PYTHON="${TACTILE_PREFIX}/bin/python"
SAM3_PYTHON="${SAM3_PREFIX}/bin/python"

VIDEO_NAME="$(basename "${VIDEO_PATH}")"
DEMO_ID="${VIDEO_NAME%.*}"
DEMO_DIR="${OUT_DIR}/${DEMO_ID}"
RGB_DIR="${DEMO_DIR}/rgb"
RGB_VIDEO="${DEMO_DIR}/rgb.mp4"
SAM3_OUTPUT_DIR="${DEMO_DIR}/sam3_bbox"
SAM3_JSONL="${SAM3_OUTPUT_DIR}/bboxes.jsonl"
PREPARE_HAND="${HAND}"
if [[ "${HAND}" == "both" ]]; then
  # Frame extraction is hand-independent; demo_tactile_video only accepts a
  # concrete side for ordinary videos once model inference is requested.
  PREPARE_HAND="right"
fi

if [[ "${SKIP_FRAME_EXTRACTION}" -eq 0 ]]; then
  echo "[1/3] Preparing orientation-correct RGB frames in tactile env: ${TACTILE_ENV_NAME}"
  "${TACTILE_PYTHON}" -u "${SCRIPT_DIR}/demo_tactile_video.py" \
    --video_path "${VIDEO_PATH}" \
    --out_dir "${OUT_DIR}" \
    --gpu "${GPU}" \
    --hand "${PREPARE_HAND}" \
    --video_rotation "${VIDEO_ROTATION}" \
    --prepare_frames_only
else
  echo "[1/3] Reusing RGB frames: ${RGB_DIR}"
  [[ -d "${RGB_DIR}" && -f "${RGB_VIDEO}" ]] || {
    echo "Existing RGB frames/rgb.mp4 are required by --skip_frame_extraction" >&2
    exit 1
  }
fi

RESOLVED_PROMPT_FRAME="${PROMPT_FRAME}"
if [[ "${PROMPT_FRAME}" == "auto" ]]; then
  PROMPT_FRAME_HELPER="${SCRIPT_DIR}/select_mediapipe_prompt_frame.py"
  RESOLVED_PROMPT_FRAME=""
  for prompt_python in "${TACTILE_PYTHON}" "${SAM3_PYTHON}"; do
    if candidate_frame="$("${prompt_python}" -u "${PROMPT_FRAME_HELPER}" \
      --video "${RGB_VIDEO}" \
      --samples "${MEDIAPIPE_PROMPT_SAMPLES}" \
      --max_num_hands "${EXPECTED_HANDS}")"; then
      if [[ "${candidate_frame}" =~ ^[0-9]+$ ]]; then
        RESOLVED_PROMPT_FRAME="${candidate_frame}"
        break
      fi
    fi
  done
  if [[ -z "${RESOLVED_PROMPT_FRAME}" ]]; then
    RESOLVED_PROMPT_FRAME="0"
    echo "MediaPipe auxiliary unavailable or found no hand; using SAM prompt frame 0." >&2
  fi
fi
echo "SAM prompt frame: ${RESOLVED_PROMPT_FRAME} (requested: ${PROMPT_FRAME})"

echo "[2/3] Tracking hand bbox in SAM3 env: ${SAM3_ENV_NAME}"
SAM3_ARGS=(
  --resource "${RGB_VIDEO}"
  --output-dir "${SAM3_OUTPUT_DIR}"
  --dataset generic
  --expected-gloved-hands "${EXPECTED_HANDS}"
  --sam-version sam3
  --checkpoint "${SAM3_CHECKPOINT}"
  --prompt-preset "${SAM3_PROMPT_PRESET}"
  --prompt-frame "${RESOLVED_PROMPT_FRAME}"
  --video-chunk-frames 0
  --video-chunk-overlap 0
  --continuous-state-memory bounded
  --continuous-state-retain-frames 64
  --continuous-state-log-interval 256
  --continuous-input-cache-frames 4
  --offload-state-to-cpu always
)
if [[ -n "${SAM3_PROMPT}" ]]; then
  SAM3_ARGS+=(--prompt "${SAM3_PROMPT}")
fi
if [[ "${SAM3_PROMPT_PRESET}" == "bare" ]]; then
  SAM3_ARGS+=(
    --bare-verification-mode filter
    --glove-verification-prompts auto
    --bare-verification-prompts auto
    --min-glove-verifier-fraction 0.10
    --max-bare-evidence-fraction 0.0
    --bare-rejection-policy bare_only
  )
fi
if [[ "${FORCE_SAM}" -eq 1 || "${SKIP_FRAME_EXTRACTION}" -eq 0 ]]; then
  SAM3_ARGS+=(--overwrite)
fi
CUDA_VISIBLE_DEVICES="${GPU}" "${SAM3_PYTHON}" -u \
  "${ROOT_DIR}/sam3_bbox_reconstruction/track_video.py" "${SAM3_ARGS[@]}"
[[ -f "${SAM3_JSONL}" ]] || { echo "SAM3 output missing: ${SAM3_JSONL}" >&2; exit 1; }

mapfile -t TRACK_SUMMARY < <(
  "${SAM3_PYTHON}" - "${SAM3_JSONL}" <<'PY'
import collections
import json
import sys

counts = collections.Counter()
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        for track in json.loads(line).get("tracks", []):
            counts[int(track["track_id"])] += 1
for track_id, frame_count in sorted(counts.items()):
    print(f"{track_id} {frame_count}")
PY
)

echo "Accepted SAM3 tracks (track_id frame_count):"
if [[ "${#TRACK_SUMMARY[@]}" -eq 0 ]]; then
  echo "  none"
else
  printf '  %s\n' "${TRACK_SUMMARY[@]}"
fi

track_exists() {
  local requested_id="$1"
  local row
  for row in "${TRACK_SUMMARY[@]}"; do
    if [[ "${row%% *}" == "${requested_id}" ]]; then
      return 0
    fi
  done
  return 1
}

COMMON_TACTILE_ARGS=(
  --checkpoint "${TACTILE_CHECKPOINT}"
  --video_path "${VIDEO_PATH}"
  --out_dir "${OUT_DIR}"
  --dino_weights "${DINO_WEIGHTS}"
  --skip_frame_extraction
  --bbox_source sam3
  --sam3_bbox_jsonl "${SAM3_JSONL}"
  --missing_bbox_policy zero
  --display_floor "${DISPLAY_FLOOR}"
  --temporal_alpha "${TEMPORAL_ALPHA}"
  --canonical_view palm
  --tactile_render_size "${TACTILE_RENDER_SIZE}"
  --video_rotation "${VIDEO_ROTATION}"
  --combined_layout "${COMBINED_LAYOUT}"
)

if [[ "${HAND}" != "both" ]]; then
  echo "[3/3] Running ${HAND}-hand tactile inference on GPU ${GPU}"
  TACTILE_ARGS=(
    "${COMMON_TACTILE_ARGS[@]}"
    --gpu "${GPU}"
    --hand "${HAND}"
    --panel_label "${HAND} PRED"
  )
  if [[ -n "${SAM3_TRACK_ID}" ]]; then
    track_exists "${SAM3_TRACK_ID}" || {
      echo "SAM3 track ${SAM3_TRACK_ID} is unavailable; see the track summary above" >&2
      exit 2
    }
    TACTILE_ARGS+=(--sam3_track_id "${SAM3_TRACK_ID}")
  fi
  "${TACTILE_PYTHON}" -u "${SCRIPT_DIR}/demo_tactile_video.py" "${TACTILE_ARGS[@]}"

  echo "SAM3 preview : ${SAM3_OUTPUT_DIR}/preview.mp4"
  echo "BBox audit  : ${DEMO_DIR}/bbox.mp4"
  echo "Tactile demo: ${DEMO_DIR}/combined.mp4"
  exit 0
fi

if [[ -z "${LEFT_SAM3_TRACK_ID}" || -z "${RIGHT_SAM3_TRACK_ID}" ]]; then
  cat >&2 <<EOF
Dual-hand inference needs an explicit anonymous-track assignment.
Inspect ${SAM3_OUTPUT_DIR}/preview.mp4, then rerun with:
  --left_sam3_track_id LEFT_ID --right_sam3_track_id RIGHT_ID --skip_frame_extraction
The completed SAM3 output will be reused unless --force_sam is supplied.
EOF
  exit 2
fi
track_exists "${LEFT_SAM3_TRACK_ID}" || {
  echo "Left SAM3 track ${LEFT_SAM3_TRACK_ID} is unavailable; see the track summary above" >&2
  exit 2
}
track_exists "${RIGHT_SAM3_TRACK_ID}" || {
  echo "Right SAM3 track ${RIGHT_SAM3_TRACK_ID} is unavailable; see the track summary above" >&2
  exit 2
}

LEFT_DIR="${DEMO_DIR}/left"
RIGHT_DIR="${DEMO_DIR}/right"
mkdir -p "${LEFT_DIR}" "${RIGHT_DIR}"
LEFT_LOG="${LEFT_DIR}/inference.log"
RIGHT_LOG="${RIGHT_DIR}/inference.log"
LEFT_PID=""
RIGHT_PID=""

terminate_dual_workers() {
  local pid
  for pid in "${LEFT_PID}" "${RIGHT_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap 'terminate_dual_workers; exit 130' INT TERM

echo "[3/3] Running dual-hand tactile inference in parallel:"
echo "  left : GPU ${LEFT_GPU}, SAM3 track ${LEFT_SAM3_TRACK_ID}, log ${LEFT_LOG}"
echo "  right: GPU ${RIGHT_GPU}, SAM3 track ${RIGHT_SAM3_TRACK_ID}, log ${RIGHT_LOG}"

"${TACTILE_PYTHON}" -u "${SCRIPT_DIR}/demo_tactile_video.py" \
  "${COMMON_TACTILE_ARGS[@]}" \
  --gpu "${LEFT_GPU}" \
  --hand left \
  --sam3_track_id "${LEFT_SAM3_TRACK_ID}" \
  --shared_rgb_dir "${RGB_DIR}" \
  --output_subdir "${DEMO_ID}/left" \
  --panel_label "LEFT PRED" \
  >"${LEFT_LOG}" 2>&1 &
LEFT_PID=$!

"${TACTILE_PYTHON}" -u "${SCRIPT_DIR}/demo_tactile_video.py" \
  "${COMMON_TACTILE_ARGS[@]}" \
  --gpu "${RIGHT_GPU}" \
  --hand right \
  --sam3_track_id "${RIGHT_SAM3_TRACK_ID}" \
  --shared_rgb_dir "${RGB_DIR}" \
  --output_subdir "${DEMO_ID}/right" \
  --panel_label "RIGHT PRED" \
  >"${RIGHT_LOG}" 2>&1 &
RIGHT_PID=$!

DUAL_START_SECONDS=${SECONDS}
DUAL_LAST_REPORT_SECONDS=${SECONDS}
while kill -0 "${LEFT_PID}" 2>/dev/null || kill -0 "${RIGHT_PID}" 2>/dev/null; do
  sleep 2
  if (( SECONDS - DUAL_LAST_REPORT_SECONDS < 30 )); then
    continue
  fi
  DUAL_LAST_REPORT_SECONDS=${SECONDS}
  LEFT_STATE="done"
  RIGHT_STATE="done"
  kill -0 "${LEFT_PID}" 2>/dev/null && LEFT_STATE="running"
  kill -0 "${RIGHT_PID}" 2>/dev/null && RIGHT_STATE="running"
  LEFT_PROGRESS="$(tail -c 512 "${LEFT_LOG}" 2>/dev/null | tr '\r' '\n' | tail -n 1 || true)"
  RIGHT_PROGRESS="$(tail -c 512 "${RIGHT_LOG}" 2>/dev/null | tr '\r' '\n' | tail -n 1 || true)"
  echo "[dual +$((SECONDS - DUAL_START_SECONDS))s] left=${LEFT_STATE}: ${LEFT_PROGRESS}"
  echo "[dual +$((SECONDS - DUAL_START_SECONDS))s] right=${RIGHT_STATE}: ${RIGHT_PROGRESS}"
done

LEFT_RC=0
RIGHT_RC=0
wait "${LEFT_PID}" || LEFT_RC=$?
wait "${RIGHT_PID}" || RIGHT_RC=$?
trap - INT TERM
if [[ "${LEFT_RC}" -ne 0 || "${RIGHT_RC}" -ne 0 ]]; then
  echo "Dual-hand inference failed: left_rc=${LEFT_RC}, right_rc=${RIGHT_RC}" >&2
  echo "Left log tail:" >&2
  tail -n 30 "${LEFT_LOG}" >&2 || true
  echo "Right log tail:" >&2
  tail -n 30 "${RIGHT_LOG}" >&2 || true
  exit 1
fi

echo "Combining left/right outputs vertically..."
"${TACTILE_PYTHON}" -u "${ROOT_DIR}/opentouch/preprocess/concat_videos.py" \
  "${LEFT_DIR}/combined.mp4" "${RIGHT_DIR}/combined.mp4" \
  --output "${DEMO_DIR}/combined_both.mp4" \
  --layout vertical \
  --preset fast
"${TACTILE_PYTHON}" -u "${ROOT_DIR}/opentouch/preprocess/concat_videos.py" \
  "${LEFT_DIR}/bbox.mp4" "${RIGHT_DIR}/bbox.mp4" \
  --output "${DEMO_DIR}/bbox_both.mp4" \
  --layout vertical \
  --preset fast

echo "SAM3 preview     : ${SAM3_OUTPUT_DIR}/preview.mp4"
echo "Left result      : ${LEFT_DIR}/combined.mp4"
echo "Right result     : ${RIGHT_DIR}/combined.mp4"
echo "Dual bbox audit  : ${DEMO_DIR}/bbox_both.mp4"
echo "Dual tactile demo: ${DEMO_DIR}/combined_both.mp4"
