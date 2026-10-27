#!/usr/bin/env bash
# 用法：bash scripts/sh/42_summarize_results.sh OUTPUT.json RESULT1.jsonl [RESULT2.jsonl ...]
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

RESULT_PATHS=("outputs/results/openseeker/predictions.jsonl")
OUTPUT_PATH="outputs/results/summary.json"
PYTHON_BIN=$(resolve_python)
LOG_PATH=${LOG_PATH:-$(new_log_path summarize_results)}
ARGS=()
for result_path in "${RESULT_PATHS[@]}"; do
  ARGS+=(--result_path "$result_path")
done

run_logged "$LOG_PATH" "$PYTHON_BIN" -u -m search_comp.evaluation.statistics \
  "${ARGS[@]}" --output_path "$OUTPUT_PATH"
