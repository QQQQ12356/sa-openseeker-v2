#!/usr/bin/env bash
# 一键分阶段入口；不复制 Plain/Beacon 训练配置。详见 docs/data/09_research_sft.md。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
ACTION="${ACTION:-html}" # html / prepare / plan / train，默认只渲染小样本
METHOD="${METHOD:-plain}" # plain / beacon
export DATASET_DIR="${DATASET_DIR:-outputs/data/research/prepared_16k}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export DATASET_NAME=research
case "$ACTION" in
  html)
    export LIMIT="${LIMIT:-20}" OFFSET="${OFFSET:-0}"
    exec bash scripts/sh/report/40_dataset_html.sh \
      "${INPUT:-outputs/data/openseeker_one.jsonl}" "${OUTPUT_HTML:-outputs/reports/research_sample.html}" ;;
  prepare) exec bash scripts/sh/data/09_prepare_research.sh "$@" ;;
  plan|train)
    export ACTION
    case "$METHOD" in
      plain) exec bash scripts/sh/plain/11_train_research.sh "$@" ;;
      beacon) exec bash scripts/sh/beacon/11_train_research.sh "$@" ;;
      baseline) exec bash scripts/sh/baseline/11_train_research.sh "$@" ;;
      *) echo 'METHOD 必须是 plain、beacon 或 baseline' >&2; exit 2 ;;
    esac ;;
  *) echo 'ACTION 必须是 html/prepare/plan/train' >&2; exit 2 ;;
esac
