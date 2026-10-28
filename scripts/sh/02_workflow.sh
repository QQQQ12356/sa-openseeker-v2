#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"
PY=$(resolve_python)
ACTION="plan"                       # plan / train / eval / test；复杂数据参数优先使用专用脚本
RUN_DIR=""                         # train/eval 可填写已有实验
action=${1:-$ACTION}
if [[ $# -gt 0 ]]; then shift; fi
if [[ -n "$RUN_DIR" ]]; then set -- --run-dir "$RUN_DIR" "$@"; fi
case "$action" in
  download) "$PY" scripts/py/16_download.py "$@" ;;
  prepare) "$PY" scripts/py/10_prepare.py "$@" ;;
  corpus) "$PY" scripts/py/15_build_corpus.py "$@" ;;
  plan|train|eval) "$PY" scripts/py/51_experiment.py "$action" "$@" ;;
  offline-benchmark) "$PY" scripts/py/52_offline_benchmark.py "$@" ;;
  benchmark) "$PY" scripts/py/47_benchmark.py "$@" ;;
  test) "$PY" -m pytest tests/test_openseeker.py "$@" ;;
  *) echo 'Usage: bash scripts/sh/02_workflow.sh {download|prepare|corpus|plan|train|eval|benchmark|offline-benchmark|test} [args]'; exit 2 ;;
esac
