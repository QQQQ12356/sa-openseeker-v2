#!/usr/bin/env bash
# 原始 Qwen3.5-4B 基线：复用训练模型的 OpenSeeker 协议与 benchmark。
# TARGET=openseeker|browsecomp|browsecomp-plus|hotpotqa；ACTION 依目标而定。
# 完整参数和数据准备见 docs/22_qwen35_baseline_and_adapter.md。
set -euo pipefail
TARGET="${TARGET:-openseeker}"
ACTION="${ACTION:-run}"
# 本地快照路径或 HF 模型 ID。离线基准和在线 BrowseComp 要求本地完整目录。
BASE_CHECKPOINT="${BASE_CHECKPOINT:-Qwen/Qwen3.5-4B}"
# 避免误将上次训练模型路径作为原始模型基线；以 BASE_CHECKPOINT 为唯一模型输入。
export MODEL_PATH="$BASE_CHECKPOINT" TARGET ACTION KIND=plain
unset BASE_MODEL_PATH
if [[ "$ACTION" == run || "$ACTION" == all ]]; then
  if [[ -f "$BASE_CHECKPOINT/adapter_config.json" ]]; then
    echo 'BASE_CHECKPOINT 必须是原始完整基座，不能是 LoRA adapter' >&2; exit 2
  fi
  if [[ "$TARGET" != openseeker && ${DRY_RUN:-0} != 1 && ! -d "$BASE_CHECKPOINT" ]]; then
    echo '此 benchmark 要求本地完整基座目录；设置 BASE_CHECKPOINT=/path/to/Qwen3.5-4B' >&2; exit 2
  fi
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/../plain/27_plain_workflow.sh" "$@"
