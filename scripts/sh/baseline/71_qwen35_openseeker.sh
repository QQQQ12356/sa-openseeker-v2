#!/usr/bin/env bash
# 原始 Qwen3.5 在 OpenSeeker 的独立入口，不加载 Plain adapter 或 SFT 模型。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ACTION="${ACTION:-run}"              # run / score；prepare/download 调用公共数据处理入口
BASE_CHECKPOINT="${BASE_CHECKPOINT:-Qwen/Qwen3.5-4B}"  # 原始模型 HF ID 或本地目录
RUN_DIR="${RUN_DIR:-}"                # 可选：仅读取训练实验的数据 split 路径，不加载其模型
DATA="${DATA:-}"                      # 自定义预处理 JSONL，优先于 RUN_DIR 的 split
SPLIT="${SPLIT:-test}"                # train / validation / test
BACKEND="${BACKEND:-replay}"          # replay / corpus / http
CORPUS="${CORPUS:-}"                  # corpus 后端 JSONL
ENDPOINT="${ENDPOINT:-}"              # http 工具网关
LIMIT="${LIMIT:-100}"                 # 正整数；21/71 均不支持 -1
NUM_WORKERS="${NUM_WORKERS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-15}"
MAX_CONTEXT="${MAX_CONTEXT:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_TOOL_TOKENS="${MAX_TOOL_TOKENS:-16384}"
MAX_TURNS="${MAX_TURNS:-32}"
DECODING="${DECODING:-greedy}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"     # 仅 sample
TOP_P="${TOP_P:-1.0}"                 # 仅 sample
GENERATION_SEED="${GENERATION_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/results/qwen35_base/openseeker}"
OUTPUT="${OUTPUT:-}"
LOG_PATH="${LOG_PATH:-}"
PREDICTIONS="${PREDICTIONS:-}"         # score 输入
OUTPUT_PATH="${OUTPUT_PATH:-}"         # score 输出 JSON，新文件
DRY_RUN="${DRY_RUN:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo 'DRY_RUN 必须是 0 或 1' >&2; exit 2; }
case "$ACTION" in
  download|prepare)
    [[ "$DRY_RUN" != 1 ]] || { echo '公共数据下载和预处理入口不支持 DRY_RUN；请直接检查数据脚本配置' >&2; exit 2; }
    if [[ "$ACTION" == download ]]; then exec bash scripts/sh/data/16_download.sh "$@"; fi
    exec bash scripts/sh/data/10_prepare.sh "$@" ;;
  run|score) ;;
  *) echo "ACTION 必须是 download、prepare、run 或 score" >&2; exit 2 ;;
esac
if [[ "$ACTION" == score ]]; then
  [[ -f "$PREDICTIONS" && -n "$OUTPUT_PATH" ]] || { echo 'score 需要 PREDICTIONS 和 OUTPUT_PATH' >&2; exit 2; }
  [[ ! -e "$OUTPUT_PATH" ]] || { echo "输出已存在: $OUTPUT_PATH" >&2; exit 2; }
  COMMAND=("$PY" -m search_comp.evaluation.statistics --result_path "$PREDICTIONS" --output_path "$OUTPUT_PATH")
else
  [[ "$SPLIT" == train || "$SPLIT" == validation || "$SPLIT" == test ]] || { echo '无效 SPLIT' >&2; exit 2; }
  if [[ -z "$DATA" && -n "$RUN_DIR" ]]; then
    DATA=$("$PY" - "$RUN_DIR" "$SPLIT" <<'PY'
import json, sys
from pathlib import Path
run, split = sys.argv[1:]
spec = json.loads((Path(run) / 'experiment.json').read_text())
key = {'train': 'train_data_path', 'validation': 'val_data_path', 'test': 'test_data_path'}[split]
value = spec['training_config'].get(key)
if not value:
    sys.exit(f'run has no {split} data; set DATA explicitly')
print(value)
PY
)
  fi
  DATA="${DATA:-outputs/data/openseeker/polar_correct_qwen35_16k/$SPLIT.jsonl}"
  [[ -f "$DATA" ]] || { echo "数据文件不存在: $DATA" >&2; exit 2; }
  case "$BACKEND" in
    replay) ;;
    corpus) [[ -f "$CORPUS" ]] || { echo 'corpus 需要有效 CORPUS' >&2; exit 2; } ;;
    http) [[ -n "$ENDPOINT" ]] || { echo 'http 需要 ENDPOINT' >&2; exit 2; } ;;
    *) echo 'BACKEND 必须是 replay、corpus 或 http' >&2; exit 2 ;;
  esac
  STAMP=$(date -u '+%Y%m%dT%H%M%S%NZ')
  OUTPUT="${OUTPUT:-$OUTPUT_DIR/predictions_${SPLIT}_$STAMP.jsonl}"
  LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/eval_${SPLIT}_$STAMP.log}"
  COMMAND=("$PY" scripts/py/50_evaluate.py --model "$BASE_CHECKPOINT" --model-origin base-qwen35
    --kind plain --data "$DATA" --data-split "$SPLIT" --output "$OUTPUT" --backend "$BACKEND"
    --limit "$LIMIT" --num-workers "$NUM_WORKERS" --progress-interval "$PROGRESS_INTERVAL"
    --max-context "$MAX_CONTEXT" --max-new-tokens "$MAX_NEW_TOKENS"
    --max-tool-tokens "$MAX_TOOL_TOKENS" --max-turns "$MAX_TURNS"
    --decoding "$DECODING" --repetition-penalty "$REPETITION_PENALTY"
    --temperature "$TEMPERATURE" --top-p "$TOP_P" --generation-seed "$GENERATION_SEED")
  [[ "$BACKEND" != corpus ]] || COMMAND+=(--corpus "$CORPUS")
  [[ "$BACKEND" != http ]] || COMMAND+=(--endpoint "$ENDPOINT")
fi
printf '[qwen35-base] command:'
printf ' %q' "${COMMAND[@]}" "$@"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
if [[ "$ACTION" == score ]]; then exec "${COMMAND[@]}" "$@"; fi
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG_PATH")"
run_logged "$LOG_PATH" "${COMMAND[@]}" "$@"
echo "[qwen35-base] predictions: $OUTPUT; metrics: ${OUTPUT%.jsonl}.metrics.json"
