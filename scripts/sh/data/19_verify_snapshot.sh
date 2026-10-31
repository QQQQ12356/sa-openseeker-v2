#!/usr/bin/env bash
# 对比官方快照和可读 JSONL 的逐行 JSON 对象。
set -euo pipefail
REVISION="5970e581dc1548f37d846a1955cb8b651d2b657e"
ORIGINAL="outputs/data/raw/polar/$REVISION/openseeker_v1_data.jsonl"
CONVERTED="outputs/data/raw/polar/openseeker_v1_data.jsonl"
OUTPUT="outputs/reports/snapshot_verification.json"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
"$PY" scripts/py/19_verify_snapshot.py --original "$ORIGINAL" --converted "$CONVERTED" \
  --revision "$REVISION" --output "$OUTPUT"
