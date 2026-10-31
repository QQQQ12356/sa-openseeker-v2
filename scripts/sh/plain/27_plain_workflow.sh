#!/usr/bin/env bash
# Plain 分阶段入口；完整参数和执行顺序见 docs/plain/21_plain_experiment_guide.md。
# TARGET=openseeker ACTION=plan bash scripts/sh/plain/27_plain_workflow.sh
# TARGET=hotpotqa ACTION=run RUN_DIR=/path/to/run LIMIT=8 bash scripts/sh/plain/27_plain_workflow.sh
# browsecomp 是在线 BrowseComp；browsecomp-plus 是固定语料离线基准。
set -euo pipefail
TARGET="${TARGET:-openseeker}"
ACTION="${ACTION:-plan}"
ENTRY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${KIND:-plain} != plain ]]; then
  echo '27 是 Plain 入口，KIND 必须为 plain' >&2
  exit 2
fi
export KIND=plain ACTION
case "$TARGET:$ACTION" in
  openseeker:download) ENTRY=../data/16_download.sh ;;
  openseeker:prepare) ENTRY=../data/10_prepare.sh ;;
  openseeker:plan|openseeker:train) ENTRY=../plain/20_plain_train.sh ;;
  openseeker:run) ENTRY=../plain/21_plain_eval.sh ;;
  openseeker:score)
    if [[ -z ${PREDICTIONS:-} && $# -lt 2 ]]; then
      echo 'OpenSeeker 评分请设置 PREDICTIONS 和 OUTPUT_PATH，或传入输出和预测文件参数' >&2; exit 2
    fi
    ENTRY=../report/42_summarize_results.sh ;;
  browsecomp:prepare) ENTRY=../benchmark/46_prepare_benchmark.sh ;;
  browsecomp:run|browsecomp:score)
    export MODE=native
    if [[ "$ACTION" == score && -z ${PREDICTIONS:-} ]]; then
      echo '在线独立评分必须设置 PREDICTIONS' >&2; exit 2
    fi
    if [[ "$ACTION" == run && -n ${PREDICTIONS:-} ]]; then
      echo 'run 阶段请清除 PREDICTIONS；已有预测评分使用 ACTION=score' >&2; exit 2
    fi
    ENTRY=../plain/47_plain_browsecomp.sh ;;
  browsecomp-plus:download|browsecomp-plus:prepare|browsecomp-plus:index|browsecomp-plus:run|browsecomp-plus:score|browsecomp-plus:all)
    ENTRY=../plain/56_plain_browsecomp_plus.sh ;;
  hotpotqa:download|hotpotqa:prepare|hotpotqa:index|hotpotqa:run|hotpotqa:score|hotpotqa:all)
    ENTRY=../plain/57_plain_hotpotqa.sh ;;
  *)
    echo "不支持 TARGET=$TARGET ACTION=$ACTION。OpenSeeker 支持 download/prepare/plan/train/run/score；browsecomp 支持 prepare/run/score；离线基准支持 download/prepare/index/run/score/all。评测集不用于 SFT。" >&2
    exit 2 ;;
esac
exec bash "$ENTRY_DIR/$ENTRY" "$@"
