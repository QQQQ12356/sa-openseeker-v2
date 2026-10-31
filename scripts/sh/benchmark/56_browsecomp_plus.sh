#!/usr/bin/env bash
# 56 BrowseComp-Plus：修改下方参数或设置同名环境变量，直接运行。
# ACTION=download 联网准备；其余阶段严格离线。完整说明见 docs/benchmark/43_offline_benchmarks.md。
set -euo pipefail
BENCHMARK="browsecomp-plus"
ACTION="${ACTION:-all}"                # download / prepare / index / run / score / all
ROOT_DIR="${ROOT_DIR:-outputs/data/benchmarks/browsecomp-plus}"
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"        # 每卡完整模型副本，按题并行；不超过可见 GPU 数
RUN_DIR="${RUN_DIR:-}"                 # 已训练实验目录；自动识别 Plain/Beacon 和最终模型
MODEL_PATH="${MODEL_PATH:-}"           # 可覆盖 RUN_DIR 的模型；必须为本地已合并 checkpoint
PROMPT_DATA="${PROMPT_DATA:-}"          # 原训练 JSONL，仅取 system/schema；优先于 RUN_DIR 自动定位；无两者用内置 Polar 原提示
KIND="${KIND:-}"                       # plain / beacon；留空自动识别
DATA="${DATA:-}"                       # 默认 ROOT_DIR/raw/queries（官方加密 Parquet）
CORPUS="${CORPUS:-}"                   # 默认 ROOT_DIR/raw/corpus；独立完整语料
DECRYPTED="${DECRYPTED:-0}"             # 自备明文官方 query JSONL 时设 1
CORPUS_LIMIT="${CORPUS_LIMIT:-}"        # 留空全量；正数仅供 smoke，不是正式固定语料结果
OFFICIAL_ONLY="${OFFICIAL_ONLY:-0}"    # ACTION=download 时 1：只安装随代码附带的官方参考文件
TITLE_WEIGHT="${TITLE_WEIGHT:-1.0}"    # FTS5 BM25 标题权重；更改需换 ROOT_DIR 重建索引
VERIFY_INDEX="${VERIFY_INDEX:-0}"      # 1：加载时额外核验整个 index.sqlite 的 SHA256
TOPK="${TOPK:-5}"
SNIPPET_TOKENS="${SNIPPET_TOKENS:-512}" # 每篇搜索结果正文上限，使用待测模型 tokenizer
ALLOW_VISIT="${ALLOW_VISIT:-0}"         # 0 search-only；1 可读取本地全文，属于扩展协议
READ_TOKENS="${READ_TOKENS:-4096}"
TOOL_COUNT="${TOOL_COUNT:-200}"
MAX_TURNS="${MAX_TURNS:-200}"
MAX_CONTEXT="${MAX_CONTEXT:-131072}"    # 序列化完整历史的逻辑预算；不是压缩后的物理 KV 长度
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_TOOL_TOKENS="${MAX_TOOL_TOKENS:-32768}"
DECODING="${DECODING:-greedy}"              # Plain: greedy / sample；Beacon: greedy
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"  # 1=不惩罚；>1 降低已出现 token 倾向
TEMPERATURE="${TEMPERATURE:-1.0}"            # 仅 sample
TOP_P="${TOP_P:-1.0}"                        # 仅 sample
GENERATION_SEED="${GENERATION_SEED:-42}"     # 独立于 SEED（裁判）的生成种子
LIMIT="${LIMIT:--1}"                   # -1 全量；正数取前 N 题
OUTPUT_DIR="${OUTPUT_DIR:-}"           # 每次生成新目录；禁止覆盖旧结果
PREDICTIONS="${PREDICTIONS:-}"           # ACTION=score 的本地预测 JSONL
JUDGE_MODEL="${JUDGE_MODEL:-}"           # 本地 Qwen3-32B 权重目录，建议单独评分时加载
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-0}"   # 1：只统计检索，不输出答案 accuracy
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_MAX_CONTEXT="${JUDGE_MAX_CONTEXT:-32768}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.7}"
JUDGE_TOP_P="${JUDGE_TOP_P:-0.8}"
JUDGE_TOP_K="${JUDGE_TOP_K:-20}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/58_offline_common.sh"
