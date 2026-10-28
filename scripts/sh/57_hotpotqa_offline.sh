#!/usr/bin/env bash
# 57 HotpotQA dev + 官方 2017 Wikipedia 摘要语料 + 持久化 BM25 + 官方 EM/F1。
# 修改下方参数或设置同名环境变量，直接运行；详见 docs/43_offline_benchmarks.md。
set -euo pipefail
BENCHMARK="hotpotqa"
ACTION="${ACTION:-all}"                # download / prepare / index / run / score / all
ROOT_DIR="${ROOT_DIR:-outputs/data/benchmarks/hotpotqa}"
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen3.5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_DIR="${RUN_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
KIND="${KIND:-}"                       # plain / beacon；默认自动识别
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
LIMIT="${LIMIT:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PREDICTIONS="${PREDICTIONS:-}"
DRY_RUN="${DRY_RUN:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/58_offline_common.sh"
