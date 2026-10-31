#!/usr/bin/env bash
# Plain 评测一键入口；全部后端、数据/预算参数共用 31_beacon_eval.sh。
# KIND 默认 plain；RUN_DIR/MODEL_PATH/DATA/BACKEND 等均可用同名环境变量覆盖。
# MODEL_PATH 可指向 Plain LoRA adapter；BASE_MODEL_PATH 可指定原训练基座，
# TOKENIZER_PATH 可指定训练 tokenizer；留空时自动读取 adapter 和训练配置。
# 例：RUN_DIR=/path/to/run DRY_RUN=1 bash scripts/sh/plain/21_plain_eval.sh
# 参数说明：docs/benchmark/40_evaluation.md 和 docs/common/45_unified_agent_flow.md。
set -euo pipefail
export KIND="${KIND:-plain}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/../beacon/31_beacon_eval.sh" "$@"
