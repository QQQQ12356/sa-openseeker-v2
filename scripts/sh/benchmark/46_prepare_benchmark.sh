#!/usr/bin/env bash
# 下载固定版本官方代码和 BrowseComp；不加载模型、不请求裁判。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
exec "$PY" scripts/py/47_benchmark.py --prepare-only "$@"
