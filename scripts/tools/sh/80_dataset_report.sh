#!/usr/bin/env bash
# 独立入口：只修改本文件参数；对应 py 文件负责执行。完整说明：docs/60_parameters.md。
set -euo pipefail

# Python 解释器；先激活 Linux 环境，或填写其 bin/python 绝对路径。
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux；Windows 仅修改代码。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# ===== 可编辑参数：该字段控制 ===== 可编辑参数；约束及推荐值见 docs/60_parameters.md 对应参数表。
# ENVIRONMENT.HF_ENDPOINT：Hugging Face 来源；可改为 https://hf-mirror.com。
export HF_ENDPOINT=https://huggingface.co

# ENVIRONMENT.HF_HUB_OFFLINE：作用及修改约束见 docs/60_parameters.md。
export HF_HUB_OFFLINE=1

# DATASET_DIR：必改为已准备版本
export SA_DATASET_DIR=artifacts/datasets/openseeker/prepared/qwen35_L16384_s42

# SPLIT：train / validation / test / rejected / all
export SA_SPLIT=train

# LIMIT：-1 全量，正数取源顺序前 N 条；小样本不能当全量成绩。
export SA_LIMIT=100

# OFFSET：跳过的记录条数
export SA_OFFSET=0

# TITLE：HTML 标题
export SA_TITLE='训练数据轨迹检查'

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=tools/80_dataset_report
export SA_LAUNCHER="$SCRIPT_DIR/80_dataset_report.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/80_dataset_report.py"
