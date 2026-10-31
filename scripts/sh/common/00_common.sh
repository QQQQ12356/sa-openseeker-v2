#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$PROJECT_ROOT"

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=1

resolve_python() {
  local env_name=${CONDA_ENV:-search-comp-qwen35-cu126}
  if [[ -n ${PYTHON:-} && -x ${PYTHON} ]]; then
    printf '%s\n' "$PYTHON"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    local candidate
    candidate="$(conda info --base)/envs/${env_name}/bin/python"
    if [[ -x $candidate ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi
  if [[ -x "$HOME/miniconda3/envs/${env_name}/bin/python" ]]; then
    printf '%s\n' "$HOME/miniconda3/envs/${env_name}/bin/python"
    return
  fi
  command -v python
}

new_log_path() {
  local name=$1
  mkdir -p outputs/logs
  printf 'outputs/logs/%s_%s.log\n' "$name" "$(date '+%Y%m%d_%H%M%S')"
}

run_logged() {
  local log_path=$1
  shift
  echo "[run] command: $*"
  echo "[run] log: $log_path"
  "$@" 2>&1 | tee "$log_path"
}
