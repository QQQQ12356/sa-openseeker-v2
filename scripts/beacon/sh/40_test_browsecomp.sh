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

# ENVIRONMENT.HF_HUB_OFFLINE：1 只用本地文件/缓存；下载脚本必须为 0
export HF_HUB_OFFLINE=1

# ENVIRONMENT.CUDA_VISIBLE_DEVICES：GPU 物理编号；多卡示例 0,1,2,3
export CUDA_VISIBLE_DEVICES=0

# ENVIRONMENT.SUMMARY_API_URL：该字段控制 ENVIRONMENT.SUMMARY_API_URL；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SUMMARY_API_URL="${SUMMARY_API_URL:-}"

# ENVIRONMENT.SUMMARY_API_KEY：该字段控制 ENVIRONMENT.SUMMARY_API_KEY；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SUMMARY_API_KEY="${SUMMARY_API_KEY:-}"

# ENVIRONMENT.SUMMARY_MODEL_NAME：该字段控制 ENVIRONMENT.SUMMARY_MODEL_NAME；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-}"

# ENVIRONMENT.JINA_API_KEYS：该字段控制 ENVIRONMENT.JINA_API_KEYS；约束及推荐值见 docs/60_parameters.md 对应参数表。
export JINA_API_KEYS="${JINA_API_KEYS:-}"

# ENVIRONMENT.SERPER_API_KEY：该字段控制 ENVIRONMENT.SERPER_API_KEY；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SERPER_API_KEY="${SERPER_API_KEY:-}"

# ENVIRONMENT.TAVILY_API_KEY：该字段控制 ENVIRONMENT.TAVILY_API_KEY；约束及推荐值见 docs/60_parameters.md 对应参数表。
export TAVILY_API_KEY="${TAVILY_API_KEY:-}"

# ENVIRONMENT.SEARCH_PROVIDER：该字段控制 ENVIRONMENT.SEARCH_PROVIDER；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SEARCH_PROVIDER=serper

# ENVIRONMENT.SUMMARY_ATTEMPTS：该字段控制 ENVIRONMENT.SUMMARY_ATTEMPTS；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SUMMARY_ATTEMPTS=3

# ENVIRONMENT.SUMMARY_MAX_INPUT_CHARS：该字段控制 ENVIRONMENT.SUMMARY_MAX_INPUT_CHARS；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SUMMARY_MAX_INPUT_CHARS=40000

# ENVIRONMENT.SUMMARY_MAX_TOKENS：该字段控制 ENVIRONMENT.SUMMARY_MAX_TOKENS；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SUMMARY_MAX_TOKENS=2048

# ENVIRONMENT.SUMMARY_TIMEOUT：该字段控制 ENVIRONMENT.SUMMARY_TIMEOUT；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SUMMARY_TIMEOUT=120

# ENVIRONMENT.JINA_TIMEOUT：该字段控制 ENVIRONMENT.JINA_TIMEOUT；约束及推荐值见 docs/60_parameters.md 对应参数表。
export JINA_TIMEOUT=50

# BENCHMARK_DIR：共享的题目/语料/索引版本；必须先完成该 benchmark 准备脚本
export SA_BENCHMARK_DIR=artifacts/benchmarks/browsecomp/v1

# TRAINING_RUN：训练实验根目录；Research 指向 research 实验；baseline 固定 空字符串
export SA_TRAINING_RUN=artifacts/experiments/beacon/openseeker/REPLACE_WITH_RUN

# MODEL_PATH：baseline 必改为本地未训练 Qwen3.5 基座；训练方法固定 空字符串，自动读取 final_model_path
export SA_MODEL_PATH=''

# PROMPT_DATA：baseline 可填写规范化 train.jsonl 以对齐比较提示词；空字符串 使用内置 Polar 提示词；训练方法固定 空字符串
export SA_PROMPT_DATA=''

# RUN_NAME：测试标签，程序追加 UTC 时间与随机后缀；小样本建议 smoke
export SA_RUN_NAME=full

# GENERATION.LIMIT：-1 全量，正数取源顺序前 N 条；小样本不能当全量成绩。
export SA_GENERATION__LIMIT=-1

# GENERATION.MAX_CONTEXT：单问题完整序列上下文预算；应符合模型能力和显存。
export SA_GENERATION__MAX_CONTEXT=131072

# GENERATION.MAX_NEW_TOKENS：每轮 assistant 生成上限，实际还受剩余上下文限制。
export SA_GENERATION__MAX_NEW_TOKENS=8192

# GENERATION.MAX_TURNS：Agent 最大生成轮数。
export SA_GENERATION__MAX_TURNS=200

# GENERATION.MAX_TOOL_TOKENS：单条工具返回编码长度上限；超限终止，不截断观测。
export SA_GENERATION__MAX_TOOL_TOKENS=32768

# GENERATION.TOOL_COUNT：每题工具调用对象数量上限；批量 query 仍属于一个 search 调用。
export SA_GENERATION__TOOL_COUNT=200

# GENERATION.TOOL_CALL_RETRIES：工具调用格式非法时允许的修正次数，不重试真实工具失败。
export SA_GENERATION__TOOL_CALL_RETRIES=1

# GENERATION.DECODING：greedy/sample；Beacon 只支持 greedy。
export SA_GENERATION__DECODING=greedy

# GENERATION.REPETITION_PENALTY：重复惩罚，正有限值；1.0 不施加额外惩罚。
export SA_GENERATION__REPETITION_PENALTY=1.0

# GENERATION.TEMPERATURE：sample 时的正温度；greedy 不采样。
export SA_GENERATION__TEMPERATURE=1.0

# GENERATION.TOP_P：sample 的 nucleus 概率阈值，范围 (0,1]。
export SA_GENERATION__TOP_P=1.0

# GENERATION.GENERATION_SEED：推理随机种子；对 sample 有意义。
export SA_GENERATION__GENERATION_SEED=42

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=beacon/40_test_browsecomp
export SA_LAUNCHER="$SCRIPT_DIR/40_test_browsecomp.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/beacon/py/40_test_browsecomp.py"
