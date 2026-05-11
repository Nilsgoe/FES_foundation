#!/bin/bash
set -euo pipefail

activate_named_env() {
  local env_name="$1"
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    return
  fi
  if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    return
  fi
  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    return
  fi
  echo "ERROR: conda was not found; cannot activate ${env_name}" >&2
  return 1
}

activate_env() {
  local env_spec="$1"
  if [ -d "${env_spec}" ] && [ -f "${env_spec}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${env_spec}/bin/activate"
  else
    activate_named_env "${env_spec}"
  fi
}
