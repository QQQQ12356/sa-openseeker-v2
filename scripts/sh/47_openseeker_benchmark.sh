#!/usr/bin/env bash
# 修改默认值或设置同名环境变量，然后一键执行；额外 CLI 参数放在命令末尾。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen3.5}"
MODE="${MODE:-official-v2}"           # official-v1 / official-v2 / native（Beacon）
RUN_DIR="${RUN_DIR:-}"                # native 自动读取训练结果
MODEL_PATH="${MODEL_PATH:-}"          # native 可显式指定已合并模型
KIND="${KIND:-}"                      # native: plain / beacon；默认读 RUN_DIR
DATA="${DATA:-outputs/data/benchmarks/browsecomp.jsonl}"
BENCHMARK="${BENCHMARK:-BrowseComp}"
OUTPUT_DIR="${OUTPUT_DIR:-}"          # 默认新目录；已有目录拒绝覆盖
PREDICTIONS="${PREDICTIONS:-}"          # 可只评分已有 query/answer/final_response
WORKERS="${WORKERS:-60}"
JUDGE_WORKERS="${JUDGE_WORKERS:-20}"
LIMIT="${LIMIT:--1}"                  # -1 全量；正数用于 smoke，不能当全量结果
MAX_TOKENS="${MAX_TOKENS:-32768}"
TOOL_COUNT="${TOOL_COUNT:-200}"        # 官方模式的工具调用上限
MAX_CONTEXT="${MAX_CONTEXT:-131072}"  # 以下三项仅用于 native
MAX_TURNS="${MAX_TURNS:-200}"
MAX_TOOL_TOKENS="${MAX_TOOL_TOKENS:-32768}"
DRY_RUN="${DRY_RUN:-0}"
ENV_FILE="${ENV_FILE:-}"              # 可选私有 shell 文件，不在日志中打印密钥

source "$(dirname "${BASH_SOURCE[0]}")/00_common.sh"
if [[ -n "$ENV_FILE" ]]; then source "$ENV_FILE"; fi
PY=$(resolve_python)
COMMAND=("$PY" scripts/py/47_benchmark.py --mode "$MODE" --data "$DATA" --benchmark "$BENCHMARK"
  --workers "$WORKERS" --judge-workers "$JUDGE_WORKERS" --limit "$LIMIT"
  --max-tokens "$MAX_TOKENS" --tool-count "$TOOL_COUNT" --max-context "$MAX_CONTEXT"
  --max-turns "$MAX_TURNS" --max-tool-tokens "$MAX_TOOL_TOKENS")
if [[ -n "$RUN_DIR" ]]; then COMMAND+=(--run-dir "$RUN_DIR"); fi
if [[ -n "$MODEL_PATH" ]]; then COMMAND+=(--model "$MODEL_PATH"); fi
if [[ -n "$KIND" ]]; then COMMAND+=(--kind "$KIND"); fi
if [[ -n "$OUTPUT_DIR" ]]; then COMMAND+=(--output-dir "$OUTPUT_DIR"); fi
if [[ -n "$PREDICTIONS" ]]; then COMMAND+=(--predictions "$PREDICTIONS"); fi
if [[ "$DRY_RUN" == 1 ]]; then COMMAND+=(--dry-run); fi
exec "${COMMAND[@]}" "$@"
