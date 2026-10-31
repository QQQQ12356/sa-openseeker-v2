#!/usr/bin/env bash
# 在独立服务终端启动已合并 Plain 模型；完整步骤见 docs/plain/48_sglang_deployment.md。
# 服务环境：environment.serve-sglang.yml；Qwen3.5 纯文本架构须先按文档核验。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-sglang}"  # 独立服务环境；显式 PYTHON 优先
RUN_DIR="${RUN_DIR:-}"                # 训练实验根目录，自动读取最终模型路径
MODEL_PATH="${MODEL_PATH:-}"          # 本地完整 checkpoint，优先于 RUN_DIR 的模型记录
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HOST="${HOST:-127.0.0.1}"             # 跨节点监听设 0.0.0.0；客户端填实际节点名/IP
TP_SIZE="${TP_SIZE:-1}"              # 张量并行卡数，不是按题并行 worker 数
PORT="${PORT:-30018}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
DRY_RUN="${DRY_RUN:-0}"               # 1 只检查本地模型路径并打印启动命令
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo 'DRY_RUN 必须是 0 或 1' >&2; exit 2; }
PY=$(resolve_python)
COMMAND=("$PY" scripts/py/48_serve_checkpoint.py --host "$HOST" --port "$PORT" --tp-size "$TP_SIZE" --context-length "$CONTEXT_LENGTH")
if [[ -n "$RUN_DIR" ]]; then COMMAND+=(--run-dir "$RUN_DIR"); fi
if [[ -n "$MODEL_PATH" ]]; then COMMAND+=(--model "$MODEL_PATH"); fi
if [[ "$DRY_RUN" == 1 ]]; then COMMAND+=(--dry-run); fi
exec "${COMMAND[@]}" "$@"
