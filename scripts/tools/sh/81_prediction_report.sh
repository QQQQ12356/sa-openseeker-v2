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

# METHOD：plain / beacon / baseline；必须匹配测试清单
export SA_METHOD=plain

# BENCHMARK：browsecomp / browsecomp-plus / hotpotqa；必须匹配测试清单
export SA_BENCHMARK=hotpotqa

# EVALUATION_DIR：作用及修改约束见 docs/60_parameters.md。
export SA_EVALUATION_DIR=artifacts/experiments/plain/openseeker/REPLACE_WITH_RUN/evaluation/hotpotqa/REPLACE_WITH_EVALUATION

# TITLE：HTML 标题
export SA_TITLE='测试轨迹检查'

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=tools/81_prediction_report
export SA_LAUNCHER="$SCRIPT_DIR/81_prediction_report.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/81_prediction_report.py"
