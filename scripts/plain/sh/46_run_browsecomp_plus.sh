#!/usr/bin/env bash
# 一次运行生成和评分。完整说明见 docs/37_browsecomp_plus_run.md。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此入口仅支持 Linux。' >&2; exit 2; }
(( $# == 0 )) || { echo '请在脚本顶部设置参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# Linux 环境和已完成 31 的数据版本。多卡可用 0,1,2,3。
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export SA_BENCHMARK_DIR=artifacts/benchmarks/browsecomp-plus/v1

# 填写本方法的 Qwen3.5-4B 训练实验根目录。
export SA_TRAINING_RUN=artifacts/experiments/plain/openseeker/REPLACE_WITH_RUN
export SA_CHECKPOINT=''  # 空值使用最终模型；也可填 checkpoint-200。
export SA_PROMPT_DATA=''  # 仅 training 用；可填规范化 JSONL。
export SA_RUN_NAME=plain-bcp4b
export SA_OVERLAP=off  # off 保留完整题集；filter 对训练实验排除重合题。
export SA_OVERLAP_THRESHOLD=0.30
export SA_RESUME_FROM=''  # 从已完成评测扩展题数时填写旧评测根目录。
export SA_EVALUATION_DIR=''  # 填已有评测根目录时，仅重新评分。
export SA_SCORE_AFTER_GENERATION=true  # EVALUATION_DIR 为空时生效。false 仅生成。

# benchmark 使用统一深度检索提示词；training 使用训练提示词并追加相同的答案格式。
export SA_GENERATION__PROMPT_PROFILE=benchmark

# -1 全量；初次运行可改为 5。每个 worker 占用一张可见 GPU 并加载一份模型。
export SA_GENERATION__LIMIT=-1
export SA_GENERATION__NUM_WORKERS=1
# 0 为 No CM；5 保留最近五个成功工具轮次。对照实验使用独立 RUN_NAME。
export SA_GENERATION__OBSERVATION_WINDOW=0
export SA_GENERATION__MAX_TURNS=500
export SA_GENERATION__MAX_CONTEXT=131072
export SA_GENERATION__MAX_NEW_TOKENS=8192
export SA_GENERATION__MAX_TOOL_TOKENS=32768
export SA_GENERATION__TOOL_COUNT=4096
export SA_GENERATION__TOOL_CALL_RETRIES=1
export SA_GENERATION__DECODING=sample
export SA_GENERATION__REPETITION_PENALTY=1.0
export SA_GENERATION__TEMPERATURE=1.0
export SA_GENERATION__TOP_P=1.0
export SA_GENERATION__GENERATION_SEED=42

# bm25=本地 SQLite（31）；qwen3-embedding-8b=本地向量（31、33）。
export SA_GENERATION__RETRIEVER=bm25
export SA_GENERATION__VECTOR_INDEX_DIR=''  # 本地向量模式填 33 的输出目录。
export SA_GENERATION__EMBEDDING_MODEL_PATH=''  # 本地向量模式填匹配的 Qwen3-Embedding-8B。
export SA_GENERATION__EMBEDDING_DEVICE=cuda:0
export SA_GENERATION__VERIFY_INDEX=true
export SA_GENERATION__TITLE_WEIGHT=1.0
export SA_GENERATION__TOPK=15
export SA_GENERATION__SNIPPET_TOKENS=512
export SA_GENERATION__READ_TOKENS=4096
export SA_GENERATION__ALLOW_VISIT=true

# openai 使用 OPENAI_API_KEY（从 Linux 环境传入）；local 使用本地裁判路径。
export SA_JUDGE__JUDGE_BACKEND=openai
export SA_JUDGE__JUDGE_MODEL=gpt-5-mini
export SA_JUDGE__JUDGE_URL=https://api.openai.com/v1
export SA_JUDGE__JUDGE_MAX_TOKENS=4096
export SA_JUDGE__JUDGE_ATTEMPTS=3
export SA_JUDGE__JUDGE_TIMEOUT=120

export SA_ENTRY=plain/46_run_browsecomp_plus
export SA_LAUNCHER="$SCRIPT_DIR/46_run_browsecomp_plus.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/plain/py/46_run_browsecomp_plus.py"
