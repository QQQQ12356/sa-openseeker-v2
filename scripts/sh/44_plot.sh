#!/usr/bin/env bash
set -euo pipefail
MEASUREMENTS="outputs/data/openseeker/polar_correct_qwen35_16k/measurements.jsonl"
MODEL="Qwen/Qwen3.5-4B"
OUTPUT="outputs/reports/length_distribution.svg"
source "$(dirname "${BASH_SOURCE[0]}")/00_common.sh"
PY=$(resolve_python)
"$PY" scripts/py/44_plot.py --measurements "$MEASUREMENTS" --model "$MODEL" --output "$OUTPUT"
