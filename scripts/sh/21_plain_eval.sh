#!/usr/bin/env bash
# Plain 评测一键入口；全部后端、数据/预算参数共用 31_beacon_eval.sh。
# KIND 默认 plain；RUN_DIR/MODEL_PATH/DATA/BACKEND 等均可用同名环境变量覆盖。
# 例：RUN_DIR=/path/to/run DRY_RUN=1 bash scripts/sh/21_plain_eval.sh
# 参数说明：docs/40_evaluation.md 和 docs/45_unified_agent_flow.md。
set -euo pipefail
export KIND="${KIND:-plain}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/31_beacon_eval.sh" "$@"
