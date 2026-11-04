#!/usr/bin/env bash
# Research 全量快照下载和 Parquet→JSONL 导出；只在 Linux 联网准备阶段运行。
# 说明：docs/data/71_research_experiment_guide.md。
set -euo pipefail
REVISION="${REVISION:-main}" # 正式实验填固定 commit；main 的解析结果写入 manifest
OUTPUT="${OUTPUT:-outputs/data/raw}" # OUTPUT/research/<commit>
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
exec "$PY" scripts/py/16_download.py --source research --revision "$REVISION" --output "$OUTPUT" "$@"
