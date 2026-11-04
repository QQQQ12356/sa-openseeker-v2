#!/usr/bin/env bash
# 一键预处理/计划/训练。默认 ACTION=plan；参数见 docs/data/14_browsecomp_sft.md。
set -euo pipefail
TRAIN_KIND=plain
source "$(dirname "${BASH_SOURCE[0]}")/../common/06_browsecomp_train.sh"
