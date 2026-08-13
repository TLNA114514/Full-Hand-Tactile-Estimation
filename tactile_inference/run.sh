#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${TACTILE_PYTHON:-python}"
CONFIG="${TACTILE_INFERENCE_CONFIG:-${SCRIPT_DIR}/config.server.json}"

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/infer.py" --config "${CONFIG}" "$@"
