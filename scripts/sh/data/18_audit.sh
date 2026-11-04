#!/usr/bin/env bash
set -euo pipefail
INPUTS=("${INPUT:-outputs/data/raw/polar/openseeker_v1_data.jsonl}") # 多文件编辑此数组；或直接用 Python --input
SOURCE="${SOURCE:-openseeker}" # openseeker / research
OUTPUT="${OUTPUT:-outputs/reports/raw_audit.json}"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
exec "$PY" scripts/py/18_audit.py --source "$SOURCE" --input "${INPUTS[@]}" --output "$OUTPUT" "$@"
