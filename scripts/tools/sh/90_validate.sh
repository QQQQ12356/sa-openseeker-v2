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
# RUN_UNIT_TESTS：false 仅运行静态检查
export SA_RUN_UNIT_TESTS=true

# TEST_PATTERNS：作用及修改约束见 docs/60_parameters.md。
export SA_TEST_PATTERNS='["test_experiment_contracts.py", "test_research_sft.py", "test_training_benchmark_matrix.py", "test_hotpot_judge.py", "test_shell_config.py"]'

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=tools/90_validate
export SA_LAUNCHER="$SCRIPT_DIR/90_validate.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/90_validate.py"
