#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/shell_utils.sh"
ENV_NAME="${SAM3_BBOX_ENV:-sam3bbox}"
PROFILE="compat-cu124"
MIRROR="tsinghua"
PYTORCH_INDEX_URL_OVERRIDE=""
PYTORCH_FIND_LINKS_OVERRIDE=""
SAM3_ROOT="${SAM3_ROOT:-${ROOT_DIR}/third_party/sam3}"
SAM3_REF="${SAM3_REF:-main}"
CHECKPOINT=""
MODEL_VERSION="sam3"
UPDATE_SOURCE=0

usage() {
  cat <<'EOF'
Usage: ./sam3_bbox_reconstruction/install_env.sh [options]

Options:
  --profile compat-cu124|official  Installation profile (default: compat-cu124)
  --mirror tsinghua|huawei|aliyun|none
                                   Package mirror preset (default: tsinghua)
  --pytorch-index-url URL         Override the CUDA PyTorch wheel index
  --pytorch-find-links URL        Override the flat CUDA wheel directory
  --env-name NAME                  Conda environment name (default: sam3bbox)
  --sam3-root PATH                SAM3 source checkout
  --sam3-ref REF                  Git branch/tag/commit for a fresh checkout
  --update-source                 Fetch and checkout --sam3-ref in an existing checkout
  --checkpoint PATH               Also instantiate a model with this local checkpoint
  --model-version sam3|sam3.1     Version used by the optional model build check
  -h, --help                      Show this help

Profiles:
  compat-cu124  Python 3.12 + PyTorch 2.6/cu124. This is an intentionally
                conservative compatibility attempt for CUDA-12.4-limited hosts.
                It disables FA3/compile in the supplied runners and is not an
                officially supported SAM3.1 combination.
  official      Current upstream-style PyTorch/cu128 environment. Do not use it
                when nvidia-smi reports a maximum CUDA version below the wheel's
                driver requirement.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --mirror) MIRROR="$2"; shift 2 ;;
    --pytorch-index-url) PYTORCH_INDEX_URL_OVERRIDE="$2"; shift 2 ;;
    --pytorch-find-links) PYTORCH_FIND_LINKS_OVERRIDE="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --sam3-root) SAM3_ROOT="$2"; shift 2 ;;
    --sam3-ref) SAM3_REF="$2"; shift 2 ;;
    --update-source) UPDATE_SOURCE=1; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --model-version) MODEL_VERSION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${PROFILE}" in
  compat-cu124|official) ;;
  *) echo "Unsupported profile: ${PROFILE}" >&2; exit 2 ;;
esac
case "${MIRROR}" in
  tsinghua|huawei|aliyun|none) ;;
  *) echo "Unsupported mirror: ${MIRROR}" >&2; exit 2 ;;
esac
case "${MODEL_VERSION}" in
  sam3|sam3.1) ;;
  *) echo "Unsupported model version: ${MODEL_VERSION}" >&2; exit 2 ;;
esac

CONDA_BIN_OVERRIDE="${CONDA_BIN:-}"
CONDA_BIN="$(resolve_conda_executable "${CONDA_BIN_OVERRIDE}" || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "Conda was not found. Set CONDA_BIN or initialize conda first." >&2
  exit 1
fi
echo "[setup] Conda executable: ${CONDA_BIN}"

CONDA_CHANNEL_ARGS=()
PIP_INDEX_ARGS=()
case "${MIRROR}" in
  tsinghua)
    CONDA_MIRROR_URL="${CONDA_MIRROR_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}"
    PIP_MIRROR_URL="${PIP_MIRROR_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    PIP_INDEX_ARGS=(--index-url "${PIP_MIRROR_URL}")
    ;;
  huawei)
    # Huawei Cloud provides a regular PyPI mirror but no verified pip-compatible
    # CUDA wheel index. Conda uses Tsinghua and CUDA wheels use flat find-links.
    CONDA_MIRROR_URL="${CONDA_MIRROR_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}"
    PIP_MIRROR_URL="${PIP_MIRROR_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
    PIP_INDEX_ARGS=(--index-url "${PIP_MIRROR_URL}")
    ;;
  aliyun)
    # Aliyun no longer exposes the Anaconda channels used here. Keep Aliyun
    # for PyPI/PyTorch wheels and use a working Conda mirror.
    CONDA_MIRROR_URL="${CONDA_MIRROR_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}"
    PIP_MIRROR_URL="${PIP_MIRROR_URL:-https://mirrors.aliyun.com/pypi/simple}"
    PIP_INDEX_ARGS=(--index-url "${PIP_MIRROR_URL}")
    ;;
  none)
    PIP_MIRROR_URL=""
    ;;
esac
if [[ "${MIRROR}" != "none" ]]; then
  if command -v curl >/dev/null 2>&1; then
    CONDA_PROBE_URL="${CONDA_MIRROR_URL}/cloud/conda-forge/noarch/current_repodata.json"
    if ! curl -fsSIL --connect-timeout 5 --max-time 15 "${CONDA_PROBE_URL}" >/dev/null; then
      CONDA_FALLBACK_URL="https://mirrors.sustech.edu.cn/anaconda"
      CONDA_FALLBACK_PROBE="${CONDA_FALLBACK_URL}/cloud/conda-forge/noarch/current_repodata.json"
      echo "[setup] Conda mirror is unavailable: ${CONDA_MIRROR_URL}" >&2
      echo "[setup] Trying fallback mirror: ${CONDA_FALLBACK_URL}" >&2
      if curl -fsSIL --connect-timeout 5 --max-time 15 "${CONDA_FALLBACK_PROBE}" >/dev/null; then
        CONDA_MIRROR_URL="${CONDA_FALLBACK_URL}"
      else
        echo "No configured Conda mirror is currently reachable." >&2
        echo "Set CONDA_MIRROR_URL to a working Anaconda mirror and retry." >&2
        exit 1
      fi
    fi
  fi
  CONDA_CHANNEL_ARGS=(
    --override-channels
    -c "${CONDA_MIRROR_URL}/cloud/conda-forge"
    -c "${CONDA_MIRROR_URL}/pkgs/main"
  )
  echo "[setup] Conda mirror: ${CONDA_MIRROR_URL}"
  echo "[setup] PyPI mirror:  ${PIP_MIRROR_URL}"
fi

if [[ "${PROFILE}" == "compat-cu124" ]]; then
  PYTORCH_CUDA_TAG="cu124"
  PYTORCH_PACKAGES=(
    "torch==2.6.0+cu124"
    "torchvision==0.21.0+cu124"
    "torchaudio==2.6.0+cu124"
  )
else
  PYTORCH_CUDA_TAG="cu128"
  PYTORCH_PACKAGES=(torch==2.10.0 torchvision torchaudio)
fi

if [[ -n "${PYTORCH_INDEX_URL_OVERRIDE}" && -n "${PYTORCH_FIND_LINKS_OVERRIDE}" ]]; then
  echo "Use only one of --pytorch-index-url and --pytorch-find-links." >&2
  exit 2
elif [[ -n "${PYTORCH_FIND_LINKS_OVERRIDE}" ]]; then
  PYTORCH_SOURCE_URL="${PYTORCH_FIND_LINKS_OVERRIDE%/}"
  PYTORCH_SOURCE_ARGS=("${PIP_INDEX_ARGS[@]}" --find-links "${PYTORCH_SOURCE_URL}")
  PYTORCH_SOURCE_KIND="find-links"
elif [[ -n "${PYTORCH_INDEX_URL_OVERRIDE}" ]]; then
  PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL_OVERRIDE%/}"
  PYTORCH_SOURCE_URL="${PYTORCH_INDEX_URL}"
  PYTORCH_SOURCE_ARGS=(--index-url "${PYTORCH_INDEX_URL}")
  PYTORCH_SOURCE_KIND="index"
elif [[ "${MIRROR}" == "none" ]]; then
  PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${PYTORCH_CUDA_TAG}"
  PYTORCH_SOURCE_URL="${PYTORCH_INDEX_URL}"
  PYTORCH_SOURCE_ARGS=(--index-url "${PYTORCH_INDEX_URL}")
  PYTORCH_SOURCE_KIND="index"
else
  # Aliyun exposes a flat wheel listing, not a PEP 503 package index. Passing
  # it as --index-url makes pip request a nonexistent /torch/ child path.
  PYTORCH_SOURCE_URL="https://mirrors.aliyun.com/pytorch-wheels/${PYTORCH_CUDA_TAG}"
  PYTORCH_SOURCE_ARGS=("${PIP_INDEX_ARGS[@]}" --find-links "${PYTORCH_SOURCE_URL}")
  PYTORCH_SOURCE_KIND="find-links"
fi
echo "[setup] PyTorch wheel source (${PYTORCH_SOURCE_KIND}): ${PYTORCH_SOURCE_URL}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

sam3_source_is_installable() {
  local source_root="$1"
  [[ -f "${source_root}/pyproject.toml" && -d "${source_root}/sam3" ]]
}

clone_sam3_source() {
  local destination="$1"
  git clone --branch "${SAM3_REF}" --depth 1 \
    https://github.com/facebookresearch/sam3.git "${destination}"
  if ! sam3_source_is_installable "${destination}"; then
    echo "Fresh SAM3 checkout is missing pyproject.toml or sam3/." >&2
    echo "Checkout path: ${destination}" >&2
    return 1
  fi
}

SAM3_PARENT="$(dirname "${SAM3_ROOT}")"
mkdir -p "${SAM3_PARENT}"
if sam3_source_is_installable "${SAM3_ROOT}"; then
  echo "[setup] Using installable SAM3 source: ${SAM3_ROOT}"
  if [[ "${UPDATE_SOURCE}" -eq 1 ]]; then
    if [[ ! -d "${SAM3_ROOT}/.git" ]]; then
      echo "--update-source requires a git checkout: ${SAM3_ROOT}" >&2
      exit 1
    fi
    git -C "${SAM3_ROOT}" fetch --tags origin
    git -C "${SAM3_ROOT}" checkout "${SAM3_REF}"
  fi
elif [[ -e "${SAM3_ROOT}" ]]; then
  # A previous interrupted/sparse/wrong clone can contain .git while lacking
  # package metadata. Build a verified replacement first and preserve the old
  # directory for inspection instead of deleting it.
  SAM3_REPAIR_ROOT="${SAM3_ROOT}.repair.$$"
  SAM3_BACKUP_ROOT="${SAM3_ROOT}.invalid.$(date -u +%Y%m%dT%H%M%SZ).$$"
  echo "[setup] Existing SAM3 source is not installable: ${SAM3_ROOT}" >&2
  echo "[setup] Cloning a verified replacement into ${SAM3_REPAIR_ROOT}" >&2
  clone_sam3_source "${SAM3_REPAIR_ROOT}"
  mv "${SAM3_ROOT}" "${SAM3_BACKUP_ROOT}"
  mv "${SAM3_REPAIR_ROOT}" "${SAM3_ROOT}"
  echo "[setup] Preserved invalid checkout at: ${SAM3_BACKUP_ROOT}"
else
  echo "[setup] Cloning SAM3 (${SAM3_REF}) into ${SAM3_ROOT}"
  clone_sam3_source "${SAM3_ROOT}"
fi

if ! sam3_source_is_installable "${SAM3_ROOT}"; then
  echo "SAM3 source validation failed: ${SAM3_ROOT}" >&2
  exit 1
fi

CONDA_ENV_PACKAGES=(
  python=3.12
  pip
  "numpy>=1.26,<2"
  "h5py>=3.11,<4"
)

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] Creating conda environment ${ENV_NAME}"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" \
    "${CONDA_CHANNEL_ARGS[@]}" "${CONDA_ENV_PACKAGES[@]}"
else
  echo "[setup] Updating existing conda environment ${ENV_NAME}"
  "${CONDA_BIN}" install -y -n "${ENV_NAME}" \
    "${CONDA_CHANNEL_ARGS[@]}" "${CONDA_ENV_PACKAGES[@]}"
fi

ENV_PREFIX="$(resolve_conda_env_prefix "${CONDA_BIN}" "${ENV_NAME}" || true)"
if [[ -z "${ENV_PREFIX}" ]]; then
  echo "Could not resolve the prefix for conda environment ${ENV_NAME}." >&2
  exit 1
fi
ENV_PYTHON="${ENV_PREFIX}/bin/python"
echo "[setup] Environment prefix: ${ENV_PREFIX}"

"${ENV_PYTHON}" -m pip install "${PIP_INDEX_ARGS[@]}" --upgrade pip wheel "setuptools<81"
if [[ "${MIRROR}" != "none" ]]; then
  "${ENV_PYTHON}" -m pip config --site set global.index-url "${PIP_MIRROR_URL}"
fi

"${ENV_PYTHON}" -m pip install \
  "${PYTORCH_SOURCE_ARGS[@]}" \
  --timeout 120 \
  --retries 10 \
  "${PYTORCH_PACKAGES[@]}"

"${ENV_PYTHON}" -m pip install "${PIP_INDEX_ARGS[@]}" -e "${SAM3_ROOT}"
"${ENV_PYTHON}" -m pip install "${PIP_INDEX_ARGS[@]}" \
  "opencv-python<4.13" pillow tqdm psutil orjson \
  pycocotools einops imageio-ffmpeg

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[setup] Authenticating Hugging Face from HF_TOKEN"
  "${ENV_PREFIX}/bin/hf" auth login --token "${HF_TOKEN}" --add-to-git-credential || true
fi

DOCTOR_ARGS=(
  --profile "${PROFILE}"
  --sam3-root "${SAM3_ROOT}"
  --model-version "${MODEL_VERSION}"
)
if [[ -n "${CHECKPOINT}" ]]; then
  DOCTOR_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
"${ENV_PYTHON}" "${ROOT_DIR}/doctor.py" "${DOCTOR_ARGS[@]}"

cat <<EOF

Environment ready: ${ENV_NAME}
SAM3 source:       ${SAM3_ROOT}
Profile:           ${PROFILE}
Mirror:            ${MIRROR}
PyTorch source:    ${PYTORCH_SOURCE_URL} (${PYTORCH_SOURCE_KIND})

For a CUDA-12.4-limited host, start with original SAM3 video tracking:
  ${ROOT_DIR}/run_domain_pilot.sh opentouch --gpus 0
  ${ROOT_DIR}/run_domain_pilot.sh touchanything --gpus 0

SAM3.1 can be tried explicitly after its checkpoint is available:
  ${ROOT_DIR}/run_domain_pilot.sh touchanything --sam-version sam3.1 --gpus 0

Dataset paths and local checkpoint search paths are built into defaults.py;
all of them remain overridable by CLI or environment variables.
EOF
