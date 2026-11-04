#!/usr/bin/env bash
# BrowseComp 独立评分：EM、F1、在线 LLM judge；不加载待测模型。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
DATA="${DATA:-outputs/data/benchmarks/browsecomp.jsonl}" # 与生成相同的题目和顺序
PREDICTIONS="${PREDICTIONS:?请设置测试生成的 result_tool200.jsonl 路径}"
OUTPUT_DIR="${OUTPUT_DIR:-}" # 必须为新目录；留空自动生成
LIMIT="${LIMIT:--1}" # 与生成相同；-1 全量
ENV_FILE="${ENV_FILE:-}" # SCORER_URLS / SCORER_API_KEY / SCORER_MODEL_NAME
JUDGE_WORKERS="${JUDGE_WORKERS:-1}" # 裁判请求并发
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
[[ -z "$ENV_FILE" ]] || source "$ENV_FILE"
PY=$(resolve_python)
COMMAND=("$PY" scripts/py/47_benchmark.py --mode native --data "$DATA"
  --benchmark BrowseComp --predictions "$PREDICTIONS" --limit "$LIMIT" --judge-workers "$JUDGE_WORKERS")
[[ -z "$OUTPUT_DIR" ]] || COMMAND+=(--output-dir "$OUTPUT_DIR")
[[ "${DRY_RUN:-0}" != 1 ]] || COMMAND+=(--dry-run)
source scripts/sh/benchmark/54_judge_options.sh
add_judge_options online
exec "${COMMAND[@]}" "$@"
