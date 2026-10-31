#!/usr/bin/env bash
# 原始 Qwen3.5 browsecomp-plus 独立入口；不读取 Plain/Beacon 模型。
# 参数和修改提示：docs/benchmark/46_benchmark_scripts.md、docs/baseline/24_qwen35_base_benchmark.md。
# 常用变量（环境变量优先）：
# ACTION=run：download / prepare / index / run / score / all。
# BASE_CHECKPOINT=/path/to/Qwen3.5-4B：原始完整权重；RUN_DIR 仅选择训练提示。
# ROOT_DIR：共享数据和索引；默认 outputs/data/benchmarks/browsecomp-plus。
# OUTPUT_DIR：本次结果的新目录；默认按方案、数据集和 UTC 时间命名。
# LIMIT=-1：全量；正数仅用于小规模验证；DRY_RUN=1：只解析配置。
# 其他检索、生成、裁判变量统一在本目录 73 的顶部配置，或以环境变量覆盖。
set -euo pipefail
export TARGET=browsecomp-plus
export ACTION="${ACTION:-run}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/73_qwen35_offline.sh" "$@"
