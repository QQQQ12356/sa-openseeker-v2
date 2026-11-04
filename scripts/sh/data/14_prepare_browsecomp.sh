#!/usr/bin/env bash
# 14 BrowseComp SFT：原始 messages -> 当前 OpenSeeker 协议；不加载模型权重。
# 全部变量和修改提示见 docs/data/14_browsecomp_sft.md。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
INPUT="${INPUT:-outputs/data/browsecomp-qwen35-sft/train.jsonl}" # 原始文件，只读
DATASET_DIR="${DATASET_DIR:-outputs/data/browsecomp-qwen35-sft/prepared_16k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}" # 必须与训练路径完全相同，用于分词和 manifest
MAX_LENGTH="${MAX_LENGTH:-16384}" # 完整序列 token 上限；不截断
WORKERS="${WORKERS:-4}"           # CPU 分词进程数
SEED="${SEED:-42}"               # 分组去重/划分种子
OBSERVATIONS="${OBSERVATIONS:-with}" # with / any / without
OFFLINE="${OFFLINE:-1}"          # 默认仅使用本地 tokenizer；0 允许下载
[[ "$OFFLINE" == 0 || "$OFFLINE" == 1 ]] || { echo 'OFFLINE 必须为 0 或 1' >&2; exit 2; }
COMMAND=("$PY" scripts/py/10_prepare.py --source browsecomp --input "$INPUT"
  --output "$DATASET_DIR" --models "$MODEL_PATH" --max-length "$MAX_LENGTH"
  --workers "$WORKERS" --seed "$SEED" --correctness any --observations "$OBSERVATIONS"
  --val-fraction 0 --test-fraction 0)
[[ "$OFFLINE" != 1 ]] || COMMAND+=(--offline)
exec "${COMMAND[@]}" "$@"
