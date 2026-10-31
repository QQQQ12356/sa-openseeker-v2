#!/usr/bin/env bash
# 四卡 A800 Plain；模型、数据和训练参数统一在 20 中修改。
# ACTION=plan bash scripts/sh/plain/28_plain_4xa800.sh 只生成配置。
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-zero2}"
export ACTION="${ACTION:-train}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/20_plain_train.sh"
