#!/usr/bin/env bash
# 从已实测的 normalized.jsonl 精细筛选，不重新分词、不截断。修改变量后一键执行。
set -euo pipefail
PREPARED="outputs/data/openseeker/polar_correct_qwen35_16k"
OUTPUT="outputs/data/openseeker/polar_memory_probe"
TOKENIZER="Qwen/Qwen3.5-4B"
MIN_LENGTH=64512
MAX_LENGTH=65536
EXACT_LENGTH=""                        # 非空时只保留恰好此长度；仍须处于 MIN/MAX 范围
CORRECTNESS="Correct"                  # Correct / Incorrect / any
OBSERVATIONS="with"                    # any / with / without
SPLIT_MODE="train"                   # grouped：正常实验；train：全部写入训练集，仅供显存测试
source "$(dirname "${BASH_SOURCE[0]}")/00_common.sh"
PY=$(resolve_python)
ARGS=()
if [[ -n "$EXACT_LENGTH" ]]; then ARGS+=(--exact-length "$EXACT_LENGTH"); fi
"$PY" scripts/py/11_select.py --prepared "$PREPARED" --output "$OUTPUT" \
  --tokenizer "$TOKENIZER" --min-length "$MIN_LENGTH" --max-length "$MAX_LENGTH" \
  --correctness "$CORRECTNESS" --observations "$OBSERVATIONS" --split-mode "$SPLIT_MODE" "${ARGS[@]}"
