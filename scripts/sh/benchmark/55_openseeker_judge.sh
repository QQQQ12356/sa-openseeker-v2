#!/usr/bin/env bash
# Source after generation; JUDGE_MODEL enables optional local semantic scoring.
run_openseeker_judge() {
  [[ -n "${JUDGE_MODEL:-}" ]] || return 0
  local judge_output="${JUDGE_OUTPUT_DIR:-${OUTPUT%.jsonl}.judge}"
  local -a judge_command=("$PY" -m search_comp.workflows.judge_openseeker
    --predictions "$OUTPUT" --data "$DATA" --limit "$LIMIT"
    --output-dir "$judge_output" --judge-model "$JUDGE_MODEL"
    --judge-attempts "${JUDGE_ATTEMPTS:-3}"
    --judge-max-tokens "${JUDGE_MAX_TOKENS:-4096}"
    --judge-max-context "${JUDGE_MAX_CONTEXT:-32768}")
  if [[ "${JUDGE_ENABLE_THINKING:-0}" == 1 ]]; then judge_command+=(--judge-enable-thinking); fi
  if [[ "${SKIP_JUDGE_PREFLIGHT:-0}" == 1 ]]; then judge_command+=(--skip-judge-preflight); fi
  "${judge_command[@]}"
}
