#!/usr/bin/env bash
# 正常训练自动合并；此入口用于合并失败恢复或 checkpoint-N，默认 CPU。
# bash scripts/sh/plain/29_plain_merge_adapter.sh "$RUN_DIR/models/model"
# bash scripts/sh/plain/29_plain_merge_adapter.sh "$RUN_DIR/models/model/checkpoint-200" --output_path /new/output
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  exec "$PY" -m search_comp.trainer.merge_lora --help
fi
if [[ $# -lt 1 ]]; then
  echo '用法: 29_plain_merge_adapter.sh 模型产物目录或adapter目录 [--output_path 新目录]' >&2
  exit 2
fi
TARGET=$1
shift
if [[ -f "$TARGET/adapter_config.json" ]]; then
  ARGS=(--adapter_path "$TARGET")
else
  ARGS=(--exp_dir "$TARGET")
fi
exec "$PY" -u -m search_comp.trainer.merge_lora "${ARGS[@]}" --device-map "${DEVICE_MAP:-cpu}" "$@"
