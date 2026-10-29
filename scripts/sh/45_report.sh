#!/usr/bin/env bash
set -euo pipefail
PREPARED="outputs/data/openseeker/polar_correct_qwen35_16k"
OUTPUT="outputs/reports/quality"
source "$(dirname "${BASH_SOURCE[0]}")/00_common.sh"
PY=$(resolve_python)
"$PY" scripts/py/45_report.py --prepared "$PREPARED" --output "$OUTPUT"
