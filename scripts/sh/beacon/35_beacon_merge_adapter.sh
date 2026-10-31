#!/usr/bin/env bash
# 把训练产出的 final_adapter 合并为可推理的 final（Beacon LoRA）。
# 训练收尾会自动做这件事；本脚本用于重跑、恢复中断的 run、或换输出目录。
#
# 用法:
#   scripts/sh/beacon/35_beacon_merge_adapter.sh outputs/models/<exp_name>
#   scripts/sh/beacon/35_beacon_merge_adapter.sh outputs/models/<exp_name>/final_adapter
#   scripts/sh/beacon/35_beacon_merge_adapter.sh outputs/models/<exp_name>/checkpoint-500 --output_path <dir>
#
# 环境变量: DEVICE_MAP（默认 cpu）、LOG_PATH、CONDA_ENV（见 common.sh）
set -euo pipefail
source "$(dirname "$0")/../common/00_common.sh"
PYTHON_BIN=$(resolve_python)

TARGET="outputs/experiments/填写实验名/models/model"
if [[ $# -gt 0 ]]; then TARGET=$1; shift; fi
DEVICE_MAP=${DEVICE_MAP:-cpu}

# 指向实验目录 vs 指向 adapter 目录：后者靠同级的 resolved_config.json 补基座与 beacon 配置。
if [ -f "$TARGET/resolved_config.json" ]; then
  ARGS=(--exp_dir "$TARGET")
elif [ -f "$(dirname "$TARGET")/resolved_config.json" ]; then
  ARGS=(--exp_dir "$(dirname "$TARGET")" --adapter_path "$TARGET")
else
  echo "[merge-adapter] 警告: $TARGET 附近没有 resolved_config.json，" >&2
  echo "  需要自行提供 --base_model_path 与 --beacon_config_json" >&2
  ARGS=(--adapter_path "$TARGET")
fi

LOG_PATH=${LOG_PATH:-$(new_log_path merge_adapter)}
# 附加参数放在最后，用户显式给的 --device-map 优先于 DEVICE_MAP
run_logged "$LOG_PATH" \
    "$PYTHON_BIN" -u -m search_comp.trainer.merge_adapter "${ARGS[@]}" --device-map "$DEVICE_MAP" "$@"
echo "[merge-adapter] 完成"
