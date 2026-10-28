#!/usr/bin/env bash
# 在单独终端启动已训练 Plain 模型；Beacon 使用 47 的 native 模式。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen3.5}"
RUN_DIR="${RUN_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-30018}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
DRY_RUN="${DRY_RUN:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/00_common.sh"
PY=$(resolve_python)
COMMAND=("$PY" scripts/py/48_serve_checkpoint.py --port "$PORT" --tp-size "$TP_SIZE" --context-length "$CONTEXT_LENGTH")
if [[ -n "$RUN_DIR" ]]; then COMMAND+=(--run-dir "$RUN_DIR"); fi
if [[ -n "$MODEL_PATH" ]]; then COMMAND+=(--model "$MODEL_PATH"); fi
if [[ "$DRY_RUN" == 1 ]]; then COMMAND+=(--dry-run); fi
exec "${COMMAND[@]}" "$@"
