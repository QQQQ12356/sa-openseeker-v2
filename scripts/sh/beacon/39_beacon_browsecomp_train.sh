#!/usr/bin/env bash
# 与 Plain 共用同一 prepared 数据；仅训练器和压缩配置不同。
set -euo pipefail
TRAIN_KIND=beacon
source "$(dirname "${BASH_SOURCE[0]}")/../common/06_browsecomp_train.sh"
