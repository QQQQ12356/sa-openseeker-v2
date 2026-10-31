#!/usr/bin/env bash
# 用法：bash scripts/sh/report/42_summarize_results.sh OUTPUT.json RESULT1.jsonl [RESULT2.jsonl ...]
set -euo pipefail
source "$(dirname "$0")/../common/00_common.sh"

RESULT_PATHS=("${PREDICTIONS:-outputs/results/openseeker/predictions.jsonl}")
OUTPUT_PATH="${OUTPUT_PATH:-outputs/results/summary.json}"
# 兼容顶部配置、同名环境变量与注释中的位置参数接口。
if [[ $# -gt 0 ]]; then
  if [[ $# -lt 2 ]]; then echo '需要 OUTPUT.json 和至少一个 RESULT.jsonl' >&2; exit 2; fi
  OUTPUT_PATH=$1
  shift
  RESULT_PATHS=("$@")
fi
PYTHON_BIN=$(resolve_python)
ARGS=()
for result_path in "${RESULT_PATHS[@]}"; do
  ARGS+=(--result_path "$result_path")
done

COMMAND=("$PYTHON_BIN" -u -m search_comp.evaluation.statistics "${ARGS[@]}" --output_path "$OUTPUT_PATH")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ -e "$OUTPUT_PATH" ]]; then echo "输出已存在，请设置新的 OUTPUT_PATH: $OUTPUT_PATH" >&2; exit 2; fi
LOG_PATH=${LOG_PATH:-$(new_log_path summarize_results)}
mkdir -p "$(dirname "$LOG_PATH")"
run_logged "$LOG_PATH" "${COMMAND[@]}"
