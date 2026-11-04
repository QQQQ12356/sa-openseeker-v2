#!/usr/bin/env bash
# RESEARCH 本地导出预处理；变量与产物详见 docs/data/09_research_sft.md。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
INPUT="${INPUT:-outputs/data/openseeker_one.jsonl}" # 只读本地 JSONL，不下载
SNAPSHOT_DIR="${SNAPSHOT_DIR:-}" # 非空时优先使用下载 manifest 中全部 jsonl_inputs，避免只处理一个 shard
DATASET_DIR="${DATASET_DIR:-outputs/data/research/prepared_16k}" # 新目录
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}" # 本地 tokenizer 身份须与训练相同
MAX_LENGTH="${MAX_LENGTH:-16384}" # 完整序列上限，整条过滤，不截断
WORKERS="${WORKERS:-4}" # CPU 分词进程数
SEED="${SEED:-42}" # 按问题分组划分种子
VAL_FRACTION="${VAL_FRACTION:-0.02}" # 可设 0
TEST_FRACTION="${TEST_FRACTION:-0.02}" # 可设 0；不是独立 benchmark
OBSERVATIONS="${OBSERVATIONS:-with}" # with / without / any
(( $# == 0 )) || { echo '请使用环境变量配置，不接受位置参数' >&2; exit 2; }
INPUTS=("$INPUT")
if [[ -n "$SNAPSHOT_DIR" ]]; then
  INPUT_LIST=$("$PY" - "$SNAPSHOT_DIR" <<'PY'
import json
from pathlib import Path
import sys
paths = json.loads((Path(sys.argv[1]) / 'download_manifest.json').read_text(encoding='utf-8')).get('jsonl_inputs')
if not paths or any(not Path(p).is_file() for p in paths):
    raise SystemExit('下载清单缺少有效 jsonl_inputs；重跑下载入口或使用 INPUT 指定本地 JSONL')
print('\n'.join(paths))
PY
)
  mapfile -t INPUTS <<< "$INPUT_LIST"
fi
ARGS=()
for path in "${INPUTS[@]}"; do ARGS+=(--input "$path"); done
exec "$PY" scripts/py/10_prepare.py --source research "${ARGS[@]}" \
  --output "$DATASET_DIR" --models "$MODEL_PATH" --offline \
  --max-length "$MAX_LENGTH" --workers "$WORKERS" --seed "$SEED" \
  --correctness any --observations "$OBSERVATIONS" \
  --val-fraction "$VAL_FRACTION" --test-fraction "$TEST_FRACTION"
