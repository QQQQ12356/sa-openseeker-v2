#!/usr/bin/env bash
# audit 不调用模型；preflight 只调用裁判正/反例。详见 docs/benchmark/54_stable_judging.md。
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
ENV_FILE="${ENV_FILE:-}" # audit 不需要；preflight 与 47 使用同一私有配置
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
if [[ -n "$ENV_FILE" ]]; then source "$ENV_FILE"; fi
PY=$(resolve_python)
exec "$PY" scripts/py/54_check_judge.py "$@"
