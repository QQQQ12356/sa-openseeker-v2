#!/usr/bin/env bash
# 部署在线 benchmark 的摘要/裁判模型；详见 docs/benchmark/49_local_aux_models.md。
# 默认共用一个 Qwen3-32B 服务；前台运行，Ctrl+C 停止，不启动训练或评测。
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-search-comp-sglang}" # 与 48 共用独立服务环境
ENV_FILE="${ENV_FILE:-}"                     # 可选私有配置，与 47 使用同一文件
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
if [[ -n "$ENV_FILE" ]]; then source "$ENV_FILE"; fi
source scripts/sh/benchmark/49_benchmark_env.example.sh
AUX_ROLE="${AUX_ROLE:-shared}"               # shared / summary / scorer
AUX_MODEL_PATH="${AUX_MODEL_PATH:-Qwen/Qwen3-32B}" # Hub ID 或本地完整权重目录
AUX_HOST="${AUX_HOST:-127.0.0.1}"             # 跨节点监听用 0.0.0.0
AUX_PORT="${AUX_PORT:-8890}"                  # 与客户端 URL 同步修改
AUX_TP_SIZE="${AUX_TP_SIZE:-1}"               # 单节点张量并行卡数
AUX_CONTEXT_LENGTH="${AUX_CONTEXT_LENGTH:-32768}" # 输入与生成总预算
AUX_MEM_FRACTION="${AUX_MEM_FRACTION:-0.80}"   # SGLang 静态内存池比例
AUX_MAX_RUNNING_REQUESTS="${AUX_MAX_RUNNING_REQUESTS:-4}" # 服务并发上限
AUX_ENABLE_THINKING="${AUX_ENABLE_THINKING:-0}" # 辅助模型默认关闭思考；请求可显式覆盖
DRY_RUN="${DRY_RUN:-0}"                       # 1 仅打印命令，不加载模型
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "$AUX_ROLE" in
  shared)
    [[ "$SUMMARY_MODEL_NAME" == "$SCORER_MODEL_NAME" && "$SUMMARY_API_URL" == "$SCORER_URLS" && "$SUMMARY_API_KEY" == "$SCORER_API_KEY" ]] || {
      echo 'shared 要求 SUMMARY 与 SCORER 的模型名、URL、key 相同；不同配置请分别用 summary/scorer 启动。' >&2; exit 2;
    }
    MODEL_NAME="$SUMMARY_MODEL_NAME"; API_KEY="$SUMMARY_API_KEY" ;;
  summary) MODEL_NAME="$SUMMARY_MODEL_NAME"; API_KEY="$SUMMARY_API_KEY" ;;
  scorer) MODEL_NAME="$SCORER_MODEL_NAME"; API_KEY="$SCORER_API_KEY" ;;
  *) echo 'AUX_ROLE 必须为 shared、summary 或 scorer' >&2; exit 2 ;;
esac
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo 'DRY_RUN 必须为 0 或 1' >&2; exit 2; }
for value in "$AUX_PORT" "$AUX_TP_SIZE" "$AUX_CONTEXT_LENGTH" "$AUX_MAX_RUNNING_REQUESTS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo '端口、TP、上下文和并发必须为正整数' >&2; exit 2; }
done
(( AUX_PORT <= 65535 )) || { echo '端口不能超过 65535' >&2; exit 2; }
[[ "$AUX_ENABLE_THINKING" == 0 || "$AUX_ENABLE_THINKING" == 1 ]] || { echo 'AUX_ENABLE_THINKING 必须为 0 或 1' >&2; exit 2; }
THINKING_JSON='{"enable_thinking":false}'
if [[ "$AUX_ENABLE_THINKING" == 1 ]]; then THINKING_JSON='{"enable_thinking":true}'; fi
PY=$(resolve_python)
COMMAND=("$PY" -m sglang.launch_server --model-path "$AUX_MODEL_PATH"
  --served-model-name "$MODEL_NAME" --host "$AUX_HOST" --port "$AUX_PORT"
  --tp-size "$AUX_TP_SIZE" --context-length "$AUX_CONTEXT_LENGTH"
  --mem-fraction-static "$AUX_MEM_FRACTION" --max-running-requests "$AUX_MAX_RUNNING_REQUESTS"
  --reasoning-parser qwen3 --default-chat-template-kwargs "$THINKING_JSON")
# EMPTY 是客户端占位值，服务端不启用认证；真实 key 传给 SGLang，但不打印。
PUBLIC_COMMAND=("${COMMAND[@]}")
if [[ "$API_KEY" != EMPTY ]]; then COMMAND+=(--api-key "$API_KEY"); fi
printf '[serve] role=%s model=%s GPU=%s\n' "$AUX_ROLE" "$AUX_MODEL_PATH" "$CUDA_VISIBLE_DEVICES"
printf '[serve] command:'
# 额外参数可能含密钥，因此只输出已知非敏感参数。
printf ' %q' "${PUBLIC_COMMAND[@]}"
printf '\n[serve] 额外参数数量：%s；认证：%s（key 不显示）\n' "$#" "$([[ "$API_KEY" == EMPTY ]] && echo off || echo on)"
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
exec "${COMMAND[@]}" "$@"
