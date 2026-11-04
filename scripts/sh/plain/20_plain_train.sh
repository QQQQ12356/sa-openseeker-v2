#!/usr/bin/env bash
# 兼容旧命令；参数请修改 10_train_openseeker.sh。Research 使用 11_train_research.sh。
set -euo pipefail
[[ "${DATASET_NAME:-openseeker}" == openseeker ]] || { echo 'Research 请使用 11_train_research.sh' >&2; exit 2; }
exec bash "$(dirname "${BASH_SOURCE[0]}")/10_train_openseeker.sh" "$@"
