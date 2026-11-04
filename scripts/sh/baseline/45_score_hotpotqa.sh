#!/usr/bin/env bash
# baseline hotpotqa 独立评分：EM / F1 / 本地 LLM judge；不加载待测模型。
# 变量、输出、修改提示：docs/project/68_training_benchmark_guide.md。
set -euo pipefail
PYTHON="${PYTHON:-}" # Linux Python 绝对路径；留空自动查找
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" # 裁判使用的 GPU
ROOT_DIR="${ROOT_DIR:-outputs/data/benchmarks/hotpotqa}" # 与生成相同的 prepared 数据
PREDICTIONS="${PREDICTIONS:?请设置测试输出的 predictions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}" # 必须为新目录；留空自动创建
LIMIT="${LIMIT:--1}" # 与生成相同；-1 为完整题集
JUDGE_MODEL="${JUDGE_MODEL:?请设置本地裁判权重目录}" # 不传待测模型
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}" # 单次裁判输出 token 上限
JUDGE_MAX_CONTEXT="${JUDGE_MAX_CONTEXT:-32768}" # 提示和输出总预算，不截断
JUDGE_ATTEMPTS="${JUDGE_ATTEMPTS:-3}" # 每题最大总尝试次数
JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-0}" # 0 关闭，1 开启
SKIP_JUDGE_PREFLIGHT="${SKIP_JUDGE_PREFLIGHT:-0}" # 1 跳过正负例校准
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}" # 0 贪心，正数采样
JUDGE_TOP_P="${JUDGE_TOP_P:-0.8}" # 仅采样时生效
JUDGE_TOP_K="${JUDGE_TOP_K:-20}" # 仅采样时生效
SEED="${SEED:-42}" # 裁判随机种子
DRY_RUN="${DRY_RUN:-0}" # 1 只打印计划
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
for flag in JUDGE_ENABLE_THINKING SKIP_JUDGE_PREFLIGHT DRY_RUN; do
  [[ ${!flag} == 0 || ${!flag} == 1 ]] || { echo "$flag 必须为 0 或 1" >&2; exit 2; }
done
COMMAND=("$PY" scripts/py/52_offline_benchmark.py score --benchmark hotpotqa
  --root "$ROOT_DIR" --predictions "$PREDICTIONS" --limit "$LIMIT" --judge-model "$JUDGE_MODEL"
  --judge-max-tokens "$JUDGE_MAX_TOKENS" --judge-max-context "$JUDGE_MAX_CONTEXT"
  --judge-attempts "$JUDGE_ATTEMPTS" --judge-temperature "$JUDGE_TEMPERATURE"
  --judge-top-p "$JUDGE_TOP_P" --judge-top-k "$JUDGE_TOP_K" --seed "$SEED")
COMMAND+=(--llm-judge)
[[ -z "$OUTPUT_DIR" ]] || COMMAND+=(--output-dir "$OUTPUT_DIR")
[[ "$JUDGE_ENABLE_THINKING" != 1 ]] || COMMAND+=(--judge-enable-thinking)
[[ "$SKIP_JUDGE_PREFLIGHT" != 1 ]] || COMMAND+=(--skip-judge-preflight)
[[ "$DRY_RUN" != 1 ]] || COMMAND+=(--dry-run)
exec "${COMMAND[@]}" "$@"
