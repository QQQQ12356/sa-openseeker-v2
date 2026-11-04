#!/usr/bin/env bash
# RESEARCH 本地导出预处理；变量与产物详见 docs/data/09_research_sft.md。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
INPUT="${INPUT:-outputs/data/openseeker_one.jsonl}" # 只读本地 JSONL，不下载
DATASET_DIR="${DATASET_DIR:-outputs/data/research/prepared_16k}" # 新目录
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}" # 本地 tokenizer 身份须与训练相同
MAX_LENGTH="${MAX_LENGTH:-16384}" # 完整序列上限，整条过滤，不截断
WORKERS="${WORKERS:-4}" # CPU 分词进程数
SEED="${SEED:-42}" # 按问题分组划分种子
VAL_FRACTION="${VAL_FRACTION:-0.02}" # 可设 0
TEST_FRACTION="${TEST_FRACTION:-0.02}" # 可设 0；不是独立 benchmark
OBSERVATIONS="${OBSERVATIONS:-with}" # with / without / any
(( $# == 0 )) || { echo '请使用环境变量配置，不接受位置参数' >&2; exit 2; }
exec "$PY" scripts/py/10_prepare.py --source research --input "$INPUT" \
  --output "$DATASET_DIR" --models "$MODEL_PATH" --offline \
  --max-length "$MAX_LENGTH" --workers "$WORKERS" --seed "$SEED" \
  --correctness any --observations "$OBSERVATIONS" \
  --val-fraction "$VAL_FRACTION" --test-fraction "$TEST_FRACTION"
