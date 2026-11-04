#!/usr/bin/env bash
set -euo pipefail
PREPARED="${PREPARED:-outputs/data/openseeker/polar_correct_qwen35_16k}"
OUTPUT="${OUTPUT:-outputs/reports/quality}" # 新报告目录
MODEL="${MODEL:-}" # 留空仅在 manifest 恰有一个 tokenizer 时自动选择
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
COMMAND=("$PY" scripts/py/45_report.py --prepared "$PREPARED" --output "$OUTPUT")
[[ -z "$MODEL" ]] || COMMAND+=(--model "$MODEL")
exec "${COMMAND[@]}" "$@"
