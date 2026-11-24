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

# BENCHMARK_DIR：共享的题目/语料/索引版本；必须先完成该 benchmark 准备脚本
export SA_BENCHMARK_DIR=artifacts/benchmarks/hotpotqa/v1

# TRAINING_RUN：同方法的训练实验根目录，包含 experiment.json 和 models/model。
export SA_TRAINING_RUN=artifacts/experiments/plain/openseeker/REPLACE_WITH_RUN

# CHECKPOINT：空值使用成功训练的最终模型。填 checkpoint-200 或本实验该目录的完整路径测试周期检查点。
export SA_CHECKPOINT=''

# MODEL_PATH：留空，由 TRAINING_RUN 与 CHECKPOINT 选择模型。
export SA_MODEL_PATH=''

# PROMPT_DATA：纯来源实验留空，由 TRAINING_RUN 提供提示词。
# mixed 实验填写单一 system/tools 的规范化 JSONL；纯来源实验留空。
export SA_PROMPT_DATA=''

# RUN_NAME：测试标签，程序追加 UTC 时间与随机后缀；小样本建议 smoke
export SA_RUN_NAME=full

# GENERATION.LIMIT：-1 全量，正数取源顺序前 N 条；小样本不能当全量成绩。
export SA_GENERATION__LIMIT=-1

# 续测来源。留空新跑；填写已完成的原测试根目录（含 job.json 和 evaluation.json）。
# LIMIT 填累计目标题数，如原测 40、目标 60 则填 60。续测结果写入新目录。
export SA_RESUME_FROM=''

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

# GENERATION.DECODING 可选 greedy 或 sample。
export SA_GENERATION__DECODING=greedy

# GENERATION.REPETITION_PENALTY：重复惩罚，正有限值；1.0 不施加额外惩罚。
export SA_GENERATION__REPETITION_PENALTY=1.0

# GENERATION.TEMPERATURE：sample 时的正温度；greedy 不采样。
export SA_GENERATION__TEMPERATURE=1.0

# GENERATION.TOP_P：sample 的 nucleus 概率阈值，范围 (0,1]。
export SA_GENERATION__TOP_P=1.0

# GENERATION.GENERATION_SEED：推理随机种子；对 sample 有意义。
export SA_GENERATION__GENERATION_SEED=42

# GENERATION.NUM_WORKERS：每 GPU 一个完整模型副本；不要与训练 NPROC 混淆。
export SA_GENERATION__NUM_WORKERS=1

# GENERATION.TITLE_WEIGHT：必须与已建索引的标题权重一致。
export SA_GENERATION__TITLE_WEIGHT=1.0

# GENERATION.VERIFY_INDEX：启动时扫描索引摘要，增加大索引读取时间。
export SA_GENERATION__VERIFY_INDEX=true

# GENERATION.TOPK：每条查询返回的最高排名文档数，正整数。
export SA_GENERATION__TOPK=5

# GENERATION.SNIPPET_TOKENS：每篇搜索结果的正文摘录 token 上限。
export SA_GENERATION__SNIPPET_TOKENS=512

# GENERATION.READ_TOKENS：visit 读取本地文档的正文 token 上限。
export SA_GENERATION__READ_TOKENS=4096

# GENERATION.ALLOW_VISIT：是否允许本地 visit；false 采用 search-only 提示词。
export SA_GENERATION__ALLOW_VISIT=true

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=plain/44_test_hotpotqa
export SA_LAUNCHER="$SCRIPT_DIR/44_test_hotpotqa.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/plain/py/44_test_hotpotqa.py"
