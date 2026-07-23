#!/usr/bin/env bash

# Resolve ModelArts/Conda variants without treating a searchable bin directory
# as the conda executable itself.
resolve_conda_executable() {
  local override="${1:-}"
  local raw candidate prefix_root path_conda
  local -a candidates=()

  [[ -n "${override}" ]] && candidates+=("${override}")
  [[ -n "${CONDA_EXE:-}" ]] && candidates+=("${CONDA_EXE}")

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/conda")
    if [[ "${CONDA_PREFIX}" == */envs/* ]]; then
      prefix_root="${CONDA_PREFIX%%/envs/*}"
      candidates+=("${prefix_root}/bin/conda")
    fi
  fi
  if [[ -n "${CONDA_PYTHON_EXE:-}" ]]; then
    candidates+=("$(dirname "${CONDA_PYTHON_EXE}")/conda")
  fi

  path_conda="$(type -P conda 2>/dev/null || true)"
  [[ -n "${path_conda}" ]] && candidates+=("${path_conda}")
  candidates+=(
    "/home/ma-user/anaconda3/bin/conda"
    "/opt/conda/bin/conda"
    "/opt/miniforge3/bin/conda"
  )

  for raw in "${candidates[@]}"; do
    [[ -n "${raw}" ]] || continue
    candidate="${raw}"
    if [[ -d "${candidate}" ]]; then
      candidate="${candidate%/}/conda"
    fi
    if [[ -f "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

resolve_conda_env_prefix() {
  local conda_executable="${1:?conda executable is required}"
  local env_name="${2:?environment name is required}"
  local prefix
  prefix="$(
    "${conda_executable}" env list 2>/dev/null |
      awk -v wanted="${env_name}" '$1 == wanted { print $NF; exit }'
  )"
  if [[ -n "${prefix}" && -d "${prefix}" && -x "${prefix}/bin/python" ]]; then
    printf '%s\n' "${prefix}"
    return 0
  fi
  return 1
}
