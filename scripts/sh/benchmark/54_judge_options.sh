#!/usr/bin/env bash
# 共享评分/工具恢复参数映射，供 47/56/72/73 入口 source；详见 54_stable_judging.md。
add_judge_options() {
  local backend=$1 flag
  JUDGE_ATTEMPTS="${JUDGE_ATTEMPTS:-3}"       # 每题最大总尝试次数；有效负判不重试
  JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-0}" # 0 关闭裁判思考
  SKIP_JUDGE_PREFLIGHT="${SKIP_JUDGE_PREFLIGHT:-0}"   # 1 跳过正/负例校准，需记录原因
  TOOL_CALL_RETRIES="${TOOL_CALL_RETRIES:-1}" # native 每题格式修正总额度；0 严格失败
  for flag in JUDGE_ENABLE_THINKING SKIP_JUDGE_PREFLIGHT; do
    [[ ${!flag} == 0 || ${!flag} == 1 ]] || { echo "$flag 必须为 0 或 1" >&2; exit 2; }
  done
  COMMAND+=(--judge-attempts "$JUDGE_ATTEMPTS" --tool-call-retries "$TOOL_CALL_RETRIES")
  if [[ "$JUDGE_ENABLE_THINKING" == 1 ]]; then COMMAND+=(--judge-enable-thinking); fi
  if [[ "$SKIP_JUDGE_PREFLIGHT" == 1 ]]; then COMMAND+=(--skip-judge-preflight); fi
  if [[ "$backend" == online ]]; then
    JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}" # 裁判输出预算，不影响待测模型
    JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"       # 单次 HTTP 超时秒数
    TOOL_POLICY="${TOOL_POLICY:-stable}"       # native visit：stable / official
    COMMAND+=(--judge-max-tokens "$JUDGE_MAX_TOKENS" --judge-timeout "$JUDGE_TIMEOUT" --tool-policy "$TOOL_POLICY")
  elif [[ "${BENCHMARK:-${TARGET:-}}" == hotpotqa ]]; then
    # 三种模型共用；默认增加语义裁判，同时保留官方 EM/F1。
    # ACTION=run 只生成，ACTION=score/all 才加载裁判。
    LLM_JUDGE="${LLM_JUDGE:-1}"           # 0 恢复仅官方评分（兼容旧实验）
    [[ "$LLM_JUDGE" == 0 || "$LLM_JUDGE" == 1 ]] || { echo 'LLM_JUDGE 必须为 0 或 1' >&2; exit 2; }
    if [[ "$LLM_JUDGE" == 1 ]]; then
      COMMAND+=(--llm-judge)
      [[ -z "${JUDGE_MODEL:-}" ]] || COMMAND+=(--judge-model "$JUDGE_MODEL")
      COMMAND+=(--judge-max-tokens "${JUDGE_MAX_TOKENS:-4096}"
        --judge-max-context "${JUDGE_MAX_CONTEXT:-32768}" --judge-temperature "${JUDGE_TEMPERATURE:-0}"
        --judge-top-p "${JUDGE_TOP_P:-0.8}" --judge-top-k "${JUDGE_TOP_K:-20}" --seed "${SEED:-42}")
    fi
  fi
}
