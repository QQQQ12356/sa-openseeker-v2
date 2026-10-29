#!/usr/bin/env bash
# 修改下方参数后执行：bash scripts/sh/10_prepare.sh
set -euo pipefail

# ===== 运行环境：仅加载 tokenizer，不加载模型权重，不需要 GPU =====
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen3.5}"

# ===== 输入与输出：输出目录必须不存在；保留原始 system；已有 v3 可复用，v4 替换版需从原始数据重建 =====
# INPUT_FILE：单个原始/旧规范化 JSONL；多文件可编辑 INPUTS 数组。
# MODEL_NAME：待训练 tokenizer；仅加载 tokenizer，不加载模型。CLI 标量参数可在末尾覆盖。
SOURCE="${SOURCE:-polar}"                         # polar / zephyr / combined
INPUTS=(
  "${INPUT_FILE:-outputs/data/raw/polar/openseeker_v1_data.jsonl}"
)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/data/openseeker/polar_correct_qwen35_16k}"
MODELS=("${MODEL_NAME:-Qwen/Qwen3.5-4B}")              # 与后续训练使用的 tokenizer 一致
MIN_LENGTH="${MIN_LENGTH:-1}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
CORRECTNESS="${CORRECTNESS:-Correct}"                  # Correct / Incorrect / any
OBSERVATIONS="${OBSERVATIONS:-any}"                    # any / with / without
WORKERS="${WORKERS:-4}"
OFFLINE="${OFFLINE:-1}"                              # 1：仅使用本地 tokenizer 缓存；0：允许下载

# ===== 数据划分：训练集比例自动为 1 - VAL_FRACTION - TEST_FRACTION =====
# 默认 train / validation / test = 96% / 2% / 2%。
# 例如 80% / 10% / 10%：将下方两个值都改为 0.10。
# 两个值必须非负且和小于 1；哈希划分的实际条数是近似比例。
# 输入文件名为 val/validation/test 时，已有题目优先保留原划分。
VAL_FRACTION="${VAL_FRACTION:-0.02}"
TEST_FRACTION="${TEST_FRACTION:-0.02}"
SEED="${SEED:-42}"                                # 固定 seed 和输入可复现划分

source "$(dirname "${BASH_SOURCE[0]}")/00_common.sh"
PY=$(resolve_python)
ARGS=()
for input in "${INPUTS[@]}"; do ARGS+=(--input "$input"); done
if [[ "$OFFLINE" == 1 ]]; then ARGS+=(--offline); fi

echo "[openseeker-prepare] Python: $PY"
echo "[openseeker-prepare] validation=$VAL_FRACTION test=$TEST_FRACTION；train 为剩余部分"
"$PY" scripts/py/10_prepare.py "${ARGS[@]}" \
  --source "$SOURCE" --output "$OUTPUT_DIR" --models "${MODELS[@]}" \
  --min-length "$MIN_LENGTH" --max-length "$MAX_LENGTH" \
  --correctness "$CORRECTNESS" --observations "$OBSERVATIONS" --workers "$WORKERS" \
  --val-fraction "$VAL_FRACTION" --test-fraction "$TEST_FRACTION" --seed "$SEED" "$@"
