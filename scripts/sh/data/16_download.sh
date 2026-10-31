#!/usr/bin/env bash
# 当前 Polar 已迁移，无需重新下载；新服务器可填写固定 revision 再运行。
set -euo pipefail
SOURCE="polar"
REVISION="main"
OUTPUT="outputs/data/raw"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
"$PY" scripts/py/16_download.py --source "$SOURCE" --revision "$REVISION" --output "$OUTPUT"
