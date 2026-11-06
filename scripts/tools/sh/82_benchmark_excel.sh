#!/usr/bin/env bash
# Benchmark 评分结果导出 Excel；参数说明：docs/82_benchmark_excel.md。
set -euo pipefail

# 使用已激活的 Linux Python，或填写 /path/to/venv/bin/python。
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux；Windows 仅修改代码。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# ===== 可编辑参数；路径相对项目根目录解析 =====
# INPUTS：严格 JSON 字符串列表；支持实验根目录、测试目录、评分目录或 summary.metrics.json。
# 目录递归查找；默认汇总全部方法和 benchmark。多个输入重叠时按真实路径去重。
export SA_INPUTS='["artifacts/experiments"]'

# OUTPUT_DIR：导出操作的父目录，必须位于本项目 artifacts/ 下；每次创建新子目录。
export SA_OUTPUT_DIR=artifacts/reports/benchmarks

# RUN_NAME：新子目录标签，1–96 字符，仅字母、数字、下划线和连字符。
export SA_RUN_NAME=benchmark_excel

# INCLUDE_DETAILS：true 导出已有的逐题指标/裁判记录；false 仅保留汇总与完整指标。
# 不导出完整 Agent 轨迹；大批量实验可关闭此项以减少工作簿体积。
export SA_INCLUDE_DETAILS=true

# INCLUDE_INCOMPLETE：默认跳过 complete!=true 或所属评分任务未完成的结果。
# true 允许导出 failed/running 的现有摘要用于排错；表中仍保留原始状态。
export SA_INCLUDE_INCOMPLETE=false

# ===== 内部入口身份；无需修改 =====
export SA_ENTRY=tools/82_benchmark_excel
export SA_LAUNCHER="$SCRIPT_DIR/82_benchmark_excel.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/82_benchmark_excel.py"
