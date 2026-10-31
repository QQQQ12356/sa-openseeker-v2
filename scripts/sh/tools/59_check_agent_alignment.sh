#!/usr/bin/env bash
# 59 CPU 协议/小模型一致性回归；不加载真实 checkpoint，不训练，不调用在线 API。
# PYTHON：Python 可执行路径，优先于 CONDA_ENV；留空按 00_common.sh 解析。
# CONDA_ENV：已安装项目依赖的环境名；脚本不安装依赖。
# OMP_NUM_THREADS：CPU 测试线程数；默认 1，避免 tiny 模型线程开销。
# 末尾参数直接传给 pytest，例如 -x、-k pattern；默认运行全部相关回归。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
exec "$PY" -m pytest -q tests/test_unified_agent.py tests/test_openseeker.py \
  tests/test_openseeker_loss_segments.py tests/test_openseeker_evaluation.py \
  tests/test_official_benchmark.py tests/test_offline_benchmarks.py \
  tests/test_beacon_qwen35_linear_memory.py "$@"
