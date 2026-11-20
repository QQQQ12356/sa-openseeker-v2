#!/usr/bin/env bash
# 独立评测入口。修改本文件参数后运行。说明见 docs/34_training_data_evaluation.md。
set -euo pipefail

# Linux Python 解释器，默认使用已激活环境。
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux；Windows 仅修改代码。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# 模型镜像地址。评测默认从实验记录中的本地模型加载。
export HF_ENDPOINT=https://huggingface.co
# 1 仅使用本地模型文件或缓存。
export HF_HUB_OFFLINE=1
# 单进程推理使用的 GPU。建议指定一个物理 GPU 编号。
export CUDA_VISIBLE_DEVICES=0

# 训练方法，plain 或 beacon，须与训练实验一致。
export SA_METHOD=plain
# 必填。训练实验根目录，包含 experiment.json、job.json 和 models/model。
export SA_TRAINING_RUN=artifacts/experiments/plain/openseeker/REPLACE_WITH_RUN
# 留空加载最终模型，也可填写 checkpoint-200 或该检查点的完整路径。
export SA_CHECKPOINT=''
# 留空读取该实验归档的划分。也可填写任一规范化 prepared 版本目录。
export SA_DATASET_DIR=''
# train 默认测试训练划分。validation 或 test 读取指定版本的保留划分。
export SA_SPLIT=train
# 检索语料版本目录，包含 index.sqlite 和 index.manifest.json。
export SA_INDEX_DIR=artifacts/benchmarks/browsecomp-plus/v1
# 输出任务标签，自动追加时间和随机后缀。
export SA_RUN_NAME=smoke

# 正数取所选划分的前 N 条；-1 使用全部。建议先跑 10 条。
export SA_GENERATION__LIMIT=10
# 续测来源。留空新跑；填同一训练实验下已完成的 evaluation/training-data 任务根目录。
# LIMIT 填累计目标题数，继续使用原 DATASET_DIR、SPLIT 和 INDEX_DIR。
export SA_RESUME_FROM=''
# 每题上下文 token 上限，以及每轮 assistant 的生成 token 上限。
export SA_GENERATION__MAX_CONTEXT=32768
export SA_GENERATION__MAX_NEW_TOKENS=4096
# 每题最多生成的回合数。
export SA_GENERATION__MAX_TURNS=20
# 单条工具观测的 token 上限；超过限制时停止该题。
export SA_GENERATION__MAX_TOOL_TOKENS=16384
# 每题工具调用对象的总上限；一个批量 query 仍算一次调用。
export SA_GENERATION__TOOL_COUNT=20
# 遇到非法调用格式时允许模型修正的次数；0 表示立即停止。
export SA_GENERATION__TOOL_CALL_RETRIES=1
# greedy 或 sample；Beacon 使用 greedy。
export SA_GENERATION__DECODING=greedy
# 重复惩罚，1.0 不施加额外惩罚。
export SA_GENERATION__REPETITION_PENALTY=1.0
# 采样温度和 nucleus 概率阈值，只在 sample 模式使用。
export SA_GENERATION__TEMPERATURE=1.0
export SA_GENERATION__TOP_P=1.0
# 采样种子，与数据选取无关。
export SA_GENERATION__GENERATION_SEED=42
# true 校验 SQLite 文件 SHA256；false 仍核对索引内部和旁路清单。
export SA_GENERATION__VERIFY_INDEX=true
# 每条 BM25 查询返回的文档数。
export SA_GENERATION__TOPK=5
# search 每篇结果的正文上限，以及 visit 每篇读取的正文上限。
export SA_GENERATION__SNIPPET_TOKENS=512
export SA_GENERATION__READ_TOKENS=4096
# true 时使用样本声明的 visit 能力，false 对所有样本只启用 search。
export SA_GENERATION__ALLOW_VISIT=true

export SA_ENTRY=tools/84_test_training_data
export SA_LAUNCHER="$SCRIPT_DIR/84_test_training_data.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/84_test_training_data.py"
