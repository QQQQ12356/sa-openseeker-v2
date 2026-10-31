#!/usr/bin/env bash
# 原始 Qwen3.5 在线 BrowseComp；通过独立基座加载器与官方 v1 在线工具评测。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
ACTION="${ACTION:-run}"                  # prepare / run / score
BASE_CHECKPOINT="${BASE_CHECKPOINT:-Qwen/Qwen3.5-4B}" # run 需要本地完整原始权重目录
RUN_DIR="${RUN_DIR:-}"                   # 可选：仅读取原训练 system/schema，不选择训练模型
PROMPT_DATA="${PROMPT_DATA:-}"            # 训练 system/schema JSONL；优先于 RUN_DIR
DATA="${DATA:-outputs/data/benchmarks/browsecomp.jsonl}"
BENCHMARK="${BENCHMARK:-BrowseComp}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PREDICTIONS="${PREDICTIONS:-}"            # score 输入
ENV_FILE="${ENV_FILE:-}"                  # 私有在线服务配置
WORKERS="${WORKERS:-60}"                 # 保留的官方参数；native 不做题目并行
JUDGE_WORKERS="${JUDGE_WORKERS:-20}"
LIMIT="${LIMIT:--1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TOOL_COUNT="${TOOL_COUNT:-200}"           # 官方裁判文件名；native 循环由 MAX_TURNS 控制
MAX_CONTEXT="${MAX_CONTEXT:-131072}"
MAX_TURNS="${MAX_TURNS:-200}"
MAX_TOOL_TOKENS="${MAX_TOOL_TOKENS:-32768}"
DECODING="${DECODING:-greedy}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
GENERATION_SEED="${GENERATION_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
if [[ -n "$ENV_FILE" ]]; then source "$ENV_FILE"; fi
PY=$(resolve_python)
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo 'DRY_RUN 必须是 0 或 1' >&2; exit 2; }
case "$ACTION" in
  prepare)
    PREPARE=("$PY" scripts/py/47_benchmark.py --prepare-only --data "$DATA")
    [[ "$DRY_RUN" != 1 ]] || PREPARE+=(--dry-run)
    exec "${PREPARE[@]}" "$@" ;;
  run) [[ -z "$PREDICTIONS" ]] || { echo 'run 请清除 PREDICTIONS' >&2; exit 2; } ;;
  score) [[ -n "$PREDICTIONS" ]] || { echo 'score 需要 PREDICTIONS' >&2; exit 2; } ;;
  *) echo 'ACTION 必须是 prepare、run 或 score' >&2; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-outputs/results/qwen35_base/browsecomp/${ACTION}_$(date -u '+%Y%m%dT%H%M%S%NZ')}"
if [[ "$ACTION" == run && "$DRY_RUN" != 1 && ! -d "$BASE_CHECKPOINT" ]]; then
  echo '在线 benchmark 需要本地原始模型目录：BASE_CHECKPOINT=/path/to/Qwen3.5-4B' >&2; exit 2
fi
COMMAND=("$PY" scripts/py/47_benchmark.py --mode native --kind plain --model-origin base-qwen35
  --model "$BASE_CHECKPOINT" --data "$DATA" --benchmark "$BENCHMARK"
  --workers "$WORKERS" --judge-workers "$JUDGE_WORKERS" --limit "$LIMIT"
  --max-tokens "$MAX_TOKENS" --tool-count "$TOOL_COUNT" --max-context "$MAX_CONTEXT"
  --max-turns "$MAX_TURNS" --max-tool-tokens "$MAX_TOOL_TOKENS"
  --decoding "$DECODING" --repetition-penalty "$REPETITION_PENALTY"
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --generation-seed "$GENERATION_SEED")
[[ -z "$RUN_DIR" ]] || COMMAND+=(--run-dir "$RUN_DIR")
[[ -z "$PROMPT_DATA" ]] || COMMAND+=(--prompt-data "$PROMPT_DATA")
[[ -z "$OUTPUT_DIR" ]] || COMMAND+=(--output-dir "$OUTPUT_DIR")
[[ -z "$PREDICTIONS" ]] || COMMAND+=(--predictions "$PREDICTIONS")
[[ "$DRY_RUN" != 1 ]] || COMMAND+=(--dry-run)
exec "${COMMAND[@]}" "$@"
