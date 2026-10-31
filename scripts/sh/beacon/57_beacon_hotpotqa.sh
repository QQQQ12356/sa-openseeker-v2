#!/usr/bin/env bash
# 57 HotpotQA dev + 官方 2017 Wikipedia 摘要语料 + 持久化 BM25 + 官方 EM/F1。
# 修改下方参数或设置同名环境变量，直接运行；详见 docs/benchmark/43_offline_benchmarks.md。
set -euo pipefail
BENCHMARK="hotpotqa"
ACTION="${ACTION:-all}"                # download / prepare / index / run / score / all
ROOT_DIR="${ROOT_DIR:-outputs/data/benchmarks/hotpotqa}"
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"        # 每卡完整模型副本，按题并行；不超过可见 GPU 数
RUN_DIR="${RUN_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
PROMPT_DATA="${PROMPT_DATA:-}"          # 原训练 JSONL，仅取 system/schema；优先于 RUN_DIR 自动定位；无两者用内置 Polar 原提示
KIND="${KIND:-beacon}"                       # 固定 beacon 入口
DATA="${DATA:-}"                       # 默认 ROOT_DIR/raw/hotpot_dev_fullwiki_v1.json
CORPUS="${CORPUS:-}"                   # 默认 ROOT_DIR/raw/wiki-abstracts.tar.bz2
CORPUS_LIMIT="${CORPUS_LIMIT:-}"        # 留空全量；smoke 子集不能标为 fullwiki 正式结果
OFFICIAL_ONLY="${OFFICIAL_ONLY:-0}"    # ACTION=download 时只安装官方参考文件，便于自备数据
TITLE_WEIGHT="${TITLE_WEIGHT:-1.0}"
VERIFY_INDEX="${VERIFY_INDEX:-0}"
TOPK="${TOPK:-5}"
SNIPPET_TOKENS="${SNIPPET_TOKENS:-512}"
ALLOW_VISIT="${ALLOW_VISIT:-1}"         # 读取本地摘要文档；不访问 Wikipedia 网站
READ_TOKENS="${READ_TOKENS:-4096}"
TOOL_COUNT="${TOOL_COUNT:-200}"
MAX_TURNS="${MAX_TURNS:-200}"
MAX_CONTEXT="${MAX_CONTEXT:-131072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_TOOL_TOKENS="${MAX_TOOL_TOKENS:-32768}"
DECODING="${DECODING:-greedy}"              # Plain: greedy / sample；Beacon: greedy
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"  # 1=不惩罚；>1 降低已出现 token 倾向
TEMPERATURE="${TEMPERATURE:-1.0}"            # 仅 sample
TOP_P="${TOP_P:-1.0}"                        # 仅 sample
GENERATION_SEED="${GENERATION_SEED:-42}"
LIMIT="${LIMIT:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PREDICTIONS="${PREDICTIONS:-}"
DRY_RUN="${DRY_RUN:-0}"

# 各方案入口独立；参数表和修改提示见 docs/benchmark/46_benchmark_scripts.md。
[[ "$KIND" == beacon ]] || { echo '此入口 KIND 必须为 beacon' >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark/58_offline_common.sh"
