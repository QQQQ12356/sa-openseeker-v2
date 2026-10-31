#!/usr/bin/env bash
# 在目标训练服务器检查 CUDA、Beacon 与 CPUAdam；包含小张量反向传播。
set -euo pipefail
PYTHON=""
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
"$PY" scripts/py/03_check_environment.py
