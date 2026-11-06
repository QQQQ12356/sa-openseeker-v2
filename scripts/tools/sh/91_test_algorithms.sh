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
# TEST_PATH：目录或单个测试文件
export SA_TEST_PATH=tests

# TOKENIZER_PATH：可选本地 tokenizer 路径；未填且缓存缺失时跳过依赖真实 tokenizer 的兼容测试
export SA_TOKENIZER_PATH=''

# VERBOSE：true 显示逐条测试名称
export SA_VERBOSE=false

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=tools/91_test_algorithms
export SA_LAUNCHER="$SCRIPT_DIR/91_test_algorithms.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/91_test_algorithms.py"
