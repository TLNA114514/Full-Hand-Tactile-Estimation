#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/shell_utils.sh"
ENV_NAME="${SAM3_BBOX_ENV:-sam3bbox}"
CONDA_BIN_OVERRIDE="${CONDA_BIN:-}"
CONDA_BIN="$(resolve_conda_executable "${CONDA_BIN_OVERRIDE}" || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "Conda was not found. Run ${ROOT_DIR}/install_env.sh first." >&2
  exit 1
fi

ENV_PREFIX="$(resolve_conda_env_prefix "${CONDA_BIN}" "${ENV_NAME}" || true)"
if [[ -z "${ENV_PREFIX}" ]]; then
  echo "Conda environment ${ENV_NAME} was not found or has no bin/python." >&2
  echo "Run ${ROOT_DIR}/install_env.sh first." >&2
  exit 1
fi

exec "${ENV_PREFIX}/bin/python" -u "${ROOT_DIR}/run_pilot.py" "$@"
