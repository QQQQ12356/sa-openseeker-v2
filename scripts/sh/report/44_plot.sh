#!/usr/bin/env bash
set -euo pipefail
MEASUREMENTS="${MEASUREMENTS:-outputs/data/openseeker/polar_correct_qwen35_16k/measurements.jsonl}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}" # 与 measurements 的 tokenizer 键严格相同
OUTPUT="${OUTPUT:-outputs/reports/length_distribution.svg}"
POPULATION="${POPULATION:-all}" # all / retained / rejected；使用原 prepare 的 rejection
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
exec "$PY" scripts/py/44_plot.py --measurements "$MEASUREMENTS" --model "$MODEL" --population "$POPULATION" --output "$OUTPUT" "$@"
