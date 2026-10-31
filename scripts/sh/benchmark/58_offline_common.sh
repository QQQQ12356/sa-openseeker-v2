#!/usr/bin/env bash
# Plain/Beacon 56/57 共用参数映射；配置在各方案目录的独立脚本顶部。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
for flag in OFFICIAL_ONLY VERIFY_INDEX ALLOW_VISIT DRY_RUN; do
  if [[ ${!flag} != 0 && ${!flag} != 1 ]]; then echo "$flag must be 0 or 1" >&2; exit 2; fi
done
COMMAND=("$PY" scripts/py/52_offline_benchmark.py "$ACTION" --benchmark "$BENCHMARK"
  --root "$ROOT_DIR" --num-workers "${NUM_WORKERS:-1}" --title-weight "$TITLE_WEIGHT" --topk "$TOPK"
  --snippet-tokens "$SNIPPET_TOKENS" --read-tokens "$READ_TOKENS"
  --tool-count "$TOOL_COUNT" --max-turns "$MAX_TURNS" --max-context "$MAX_CONTEXT"
  --max-new-tokens "$MAX_NEW_TOKENS" --max-tool-tokens "$MAX_TOOL_TOKENS" --limit "$LIMIT"
  --decoding "$DECODING" --repetition-penalty "$REPETITION_PENALTY"
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --generation-seed "$GENERATION_SEED")
if [[ -n "$RUN_DIR" ]]; then COMMAND+=(--run-dir "$RUN_DIR"); fi
if [[ -n "$MODEL_PATH" ]]; then COMMAND+=(--model "$MODEL_PATH"); fi
if [[ -n "${PROMPT_DATA:-}" ]]; then COMMAND+=(--prompt-data "$PROMPT_DATA"); fi
if [[ -n "$KIND" ]]; then COMMAND+=(--kind "$KIND"); fi
if [[ -n "$DATA" ]]; then COMMAND+=(--data "$DATA"); fi
if [[ -n "$CORPUS" ]]; then COMMAND+=(--corpus "$CORPUS"); fi
if [[ -n "$CORPUS_LIMIT" ]]; then COMMAND+=(--corpus-limit "$CORPUS_LIMIT"); fi
if [[ -n "$OUTPUT_DIR" ]]; then COMMAND+=(--output-dir "$OUTPUT_DIR"); fi
if [[ -n "$PREDICTIONS" ]]; then COMMAND+=(--predictions "$PREDICTIONS"); fi
if [[ "$OFFICIAL_ONLY" == 1 ]]; then COMMAND+=(--official-only); fi
if [[ "$VERIFY_INDEX" == 1 ]]; then COMMAND+=(--verify-index); fi
if [[ "$ALLOW_VISIT" == 1 ]]; then COMMAND+=(--allow-visit); fi
if [[ "$DRY_RUN" == 1 ]]; then COMMAND+=(--dry-run); fi
if [[ "$BENCHMARK" == browsecomp-plus ]]; then
  if [[ "$DECRYPTED" != 0 && "$DECRYPTED" != 1 || "$RETRIEVAL_ONLY" != 0 && "$RETRIEVAL_ONLY" != 1 ]]; then
    echo 'DECRYPTED and RETRIEVAL_ONLY must be 0 or 1' >&2; exit 2
  fi
  if [[ "$DECRYPTED" == 1 ]]; then COMMAND+=(--decrypted); fi
  if [[ "$RETRIEVAL_ONLY" == 1 ]]; then COMMAND+=(--retrieval-only); fi
  if [[ -n "$JUDGE_MODEL" ]]; then COMMAND+=(--judge-model "$JUDGE_MODEL"); fi
  COMMAND+=(--judge-max-tokens "$JUDGE_MAX_TOKENS" --judge-max-context "$JUDGE_MAX_CONTEXT"
    --judge-temperature "$JUDGE_TEMPERATURE" --judge-top-p "$JUDGE_TOP_P" --judge-top-k "$JUDGE_TOP_K" --seed "$SEED")
fi
exec "${COMMAND[@]}" "$@"
