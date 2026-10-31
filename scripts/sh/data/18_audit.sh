#!/usr/bin/env bash
set -euo pipefail
INPUTS=("outputs/data/raw/polar/openseeker_v1_data.jsonl")
OUTPUT="outputs/reports/raw_audit.json"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
"$PY" scripts/py/18_audit.py --input "${INPUTS[@]}" --output "$OUTPUT"
