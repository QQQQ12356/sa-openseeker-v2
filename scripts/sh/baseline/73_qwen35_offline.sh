#!/usr/bin/env bash
# 原始 Qwen3.5 固定语料离线基准；只调用通用数据、索引、agent 和评分实现。
set -euo pipefail
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
TARGET="${TARGET:-hotpotqa}"             # hotpotqa / browsecomp-plus
ACTION="${ACTION:-run}"                  # download / prepare / index / run / score / all
BASE_CHECKPOINT="${BASE_CHECKPOINT:-Qwen/Qwen3.5-4B}" # run/all 需要本地原始权重目录
RUN_DIR="${RUN_DIR:-}"                   # 可选：仅读训练 system/schema
PROMPT_DATA="${PROMPT_DATA:-}"            # 可选：原训练 system/schema JSONL
ROOT_DIR="${ROOT_DIR:-outputs/data/benchmarks/$TARGET}"
DATA="${DATA:-}"                         # 基准原始问题文件；留空用 ROOT_DIR 默认
CORPUS="${CORPUS:-}"                     # 独立完整语料；留空用 ROOT_DIR 默认
CORPUS_LIMIT="${CORPUS_LIMIT:-}"         # 正数仅 smoke，不能称全语料
OFFICIAL_ONLY="${OFFICIAL_ONLY:-0}"
TITLE_WEIGHT="${TITLE_WEIGHT:-1.0}"
VERIFY_INDEX="${VERIFY_INDEX:-0}"
TOPK="${TOPK:-5}"
SNIPPET_TOKENS="${SNIPPET_TOKENS:-512}"
if [[ "$TARGET" == hotpotqa ]]; then DEFAULT_VISIT=1; else DEFAULT_VISIT=0; fi
ALLOW_VISIT="${ALLOW_VISIT:-$DEFAULT_VISIT}"
READ_TOKENS="${READ_TOKENS:-4096}"
TOOL_COUNT="${TOOL_COUNT:-200}"
MAX_TURNS="${MAX_TURNS:-200}"
MAX_CONTEXT="${MAX_CONTEXT:-131072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_TOOL_TOKENS="${MAX_TOOL_TOKENS:-32768}"
LIMIT="${LIMIT:--1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"             # 新目录，不覆盖
PREDICTIONS="${PREDICTIONS:-}"           # score 输入
DECRYPTED="${DECRYPTED:-0}"              # BCP 自备明文问题时设 1
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-0}"   # BCP 仅检索，无答案 accuracy
JUDGE_MODEL="${JUDGE_MODEL:-}"           # BCP 本地裁判目录
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_MAX_CONTEXT="${JUDGE_MAX_CONTEXT:-32768}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.7}"
JUDGE_TOP_P="${JUDGE_TOP_P:-0.8}"
JUDGE_TOP_K="${JUDGE_TOP_K:-20}"
SEED="${SEED:-42}"                       # BCP 裁判种子，独立于生成种子
DECODING="${DECODING:-greedy}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
GENERATION_SEED="${GENERATION_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
[[ "$TARGET" == hotpotqa || "$TARGET" == browsecomp-plus ]] || { echo 'TARGET 必须是 hotpotqa 或 browsecomp-plus' >&2; exit 2; }
case "$ACTION" in download|prepare|index|run|score|all) ;; *) echo '无效 ACTION' >&2; exit 2 ;; esac
if [[ "$ACTION" == run || "$ACTION" == score || "$ACTION" == all ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-outputs/results/qwen35_base/$TARGET/${ACTION}_$(date -u '+%Y%m%dT%H%M%S%NZ')}"
fi
for flag in OFFICIAL_ONLY VERIFY_INDEX ALLOW_VISIT DECRYPTED RETRIEVAL_ONLY DRY_RUN; do
  [[ ${!flag} == 0 || ${!flag} == 1 ]] || { echo "$flag 必须是 0 或 1" >&2; exit 2; }
done
if [[ ( "$ACTION" == run || "$ACTION" == all ) && "$DRY_RUN" != 1 && ! -d "$BASE_CHECKPOINT" ]]; then
  echo '离线 benchmark 需要本地原始模型目录：BASE_CHECKPOINT=/path/to/Qwen3.5-4B' >&2; exit 2
fi
COMMAND=("$PY" scripts/py/52_offline_benchmark.py "$ACTION" --benchmark "$TARGET" --root "$ROOT_DIR"
  --model "$BASE_CHECKPOINT" --model-origin base-qwen35 --kind plain --num-workers "$NUM_WORKERS"
  --title-weight "$TITLE_WEIGHT" --topk "$TOPK" --snippet-tokens "$SNIPPET_TOKENS"
  --read-tokens "$READ_TOKENS" --tool-count "$TOOL_COUNT" --max-turns "$MAX_TURNS"
  --max-context "$MAX_CONTEXT" --max-new-tokens "$MAX_NEW_TOKENS" --max-tool-tokens "$MAX_TOOL_TOKENS"
  --limit "$LIMIT" --decoding "$DECODING" --repetition-penalty "$REPETITION_PENALTY"
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --generation-seed "$GENERATION_SEED")
for pair in "RUN_DIR:--run-dir" "PROMPT_DATA:--prompt-data" "DATA:--data" "CORPUS:--corpus" \
            "CORPUS_LIMIT:--corpus-limit" "OUTPUT_DIR:--output-dir" "PREDICTIONS:--predictions"; do
  key=${pair%%:*}; flag=${pair#*:}
  [[ -z ${!key} ]] || COMMAND+=("$flag" "${!key}")
done
[[ "$OFFICIAL_ONLY" != 1 ]] || COMMAND+=(--official-only)
[[ "$VERIFY_INDEX" != 1 ]] || COMMAND+=(--verify-index)
[[ "$ALLOW_VISIT" != 1 ]] || COMMAND+=(--allow-visit)
[[ "$DRY_RUN" != 1 ]] || COMMAND+=(--dry-run)
if [[ "$TARGET" == browsecomp-plus ]]; then
  [[ "$DECRYPTED" != 1 ]] || COMMAND+=(--decrypted)
  [[ "$RETRIEVAL_ONLY" != 1 ]] || COMMAND+=(--retrieval-only)
  [[ -z "$JUDGE_MODEL" ]] || COMMAND+=(--judge-model "$JUDGE_MODEL")
  COMMAND+=(--judge-max-tokens "$JUDGE_MAX_TOKENS" --judge-max-context "$JUDGE_MAX_CONTEXT"
    --judge-temperature "$JUDGE_TEMPERATURE" --judge-top-p "$JUDGE_TOP_P" --judge-top-k "$JUDGE_TOP_K" --seed "$SEED")
fi
exec "${COMMAND[@]}" "$@"
