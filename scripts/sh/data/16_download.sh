#!/usr/bin/env bash
# OpenSeeker 下载；解析 revision 为实际 commit，原始文件和导出清单一起保存。
set -euo pipefail
SOURCE="${SOURCE:-polar}" # polar / zephyr；Research 使用 08_download_research.sh
REVISION="${REVISION:-main}" # 正式实验推荐固定 commit；main 会被解析并记录为 commit
OUTPUT="${OUTPUT:-outputs/data/raw}" # 实际目录为 OUTPUT/SOURCE/<commit>
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
exec "$PY" scripts/py/16_download.py --source "$SOURCE" --revision "$REVISION" --output "$OUTPUT" "$@"
