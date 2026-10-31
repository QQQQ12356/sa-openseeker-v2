#!/usr/bin/env bash
# 四卡 A800 OpenSeeker Beacon；模型、数据及训练参数统一在 30 中修改。
# ACTION=plan bash scripts/sh/beacon/38_beacon_4xa800.sh 只归档配置，不加载模型或训练。
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-zero2}"
export ACTION="${ACTION:-train}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/30_beacon_train.sh"
