#!/usr/bin/env bash
# 14 BrowseComp-Plus SFT：原始 messages -> 当前 OpenSeeker 协议；仅处理数据。
# 全部变量和修改提示见 docs/data/14_browsecomp_sft.md。
set -euo pipefail
# ===== 运行环境：复用已有环境，不安装依赖 =====
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
# ===== 数据参数：可直接修改默认值，或使用 VAR=value bash ... 覆盖 =====
INPUT="${INPUT:-outputs/data/browsecomp-qwen35-sft/train.jsonl}" # 原始文件，只读
DATASET_DIR="${DATASET_DIR:-outputs/data/browsecomp-qwen35-sft/prepared_16k}" # 必须为尚不存在的新目录
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}" # 必须与训练路径完全相同，用于分词和 manifest
MAX_LENGTH="${MAX_LENGTH:-16384}" # 完整序列 token 上限；不截断
WORKERS="${WORKERS:-4}"           # CPU 分词进程数
SEED="${SEED:-42}"               # 分组去重/划分种子
OBSERVATIONS="${OBSERVATIONS:-with}" # with / any / without
OFFLINE="${OFFLINE:-1}"          # 默认仅使用本地 tokenizer；0 允许下载
[[ "$OFFLINE" == 0 || "$OFFLINE" == 1 ]] || { echo 'OFFLINE 必须为 0 或 1' >&2; exit 2; }
[[ "$MAX_LENGTH" =~ ^[0-9]+$ && "$MAX_LENGTH" =~ [1-9] ]] || { echo 'MAX_LENGTH 必须为正整数' >&2; exit 2; }
[[ "$WORKERS" =~ ^[0-9]+$ && "$WORKERS" =~ [1-9] ]] || { echo 'WORKERS 必须为正整数' >&2; exit 2; }
[[ "$OBSERVATIONS" == with || "$OBSERVATIONS" == any || "$OBSERVATIONS" == without ]] || { echo 'OBSERVATIONS 必须为 with、any 或 without' >&2; exit 2; }
[[ -f "$INPUT" ]] || { echo "找不到原始输入文件：$INPUT" >&2; exit 2; }
[[ ! -e "$DATASET_DIR" ]] || { echo "输出路径已存在：$DATASET_DIR；请复用已有结果进行训练，或设置新的 DATASET_DIR。" >&2; exit 2; }
# 此入口统一使用下方变量；拒绝额外参数，避免 CLI 覆盖变量后数据与训练设置不一致。
(( $# == 0 )) || { echo '预处理入口不接收位置参数；请使用文档中的环境变量。' >&2; exit 2; }
COMMAND=("$PY" scripts/py/10_prepare.py --source browsecomp --input "$INPUT"
  --output "$DATASET_DIR" --models "$MODEL_PATH" --max-length "$MAX_LENGTH"
  --workers "$WORKERS" --seed "$SEED" --correctness any --observations "$OBSERVATIONS"
  --val-fraction 0 --test-fraction 0)
[[ "$OFFLINE" != 1 ]] || COMMAND+=(--offline)
echo "[browsecomp-plus:data] Python: $PY"
echo "[browsecomp-plus:data] input: $INPUT; output: $DATASET_DIR"
echo "[browsecomp-plus:data] tokenizer: $MODEL_PATH; max_length: $MAX_LENGTH; workers: $WORKERS"
exec "${COMMAND[@]}"
