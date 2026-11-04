#!/usr/bin/env bash
# 复用自包含实验入口；调用者先设置 TRAIN_KIND=plain / beacon。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_common.sh"
export DATASET_DIR="${DATASET_DIR:-outputs/data/browsecomp-qwen35-sft/prepared_16k}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export ACTION="${ACTION:-plan}" # plan 默认只归档配置；train 才训练
[[ "$ACTION" == plan || "$ACTION" == train ]] || { echo 'ACTION 必须为 plan 或 train' >&2; exit 2; }
if [[ ! -f "$DATASET_DIR/manifest.json" ]]; then
  bash scripts/sh/data/14_prepare_browsecomp.sh
fi
case "$TRAIN_KIND" in
  plain) bash scripts/sh/plain/20_plain_train.sh "$@" ;;
  beacon) bash scripts/sh/beacon/30_beacon_train.sh "$@" ;;
  *) echo 'TRAIN_KIND 必须为 plain 或 beacon' >&2; exit 2 ;;
esac
