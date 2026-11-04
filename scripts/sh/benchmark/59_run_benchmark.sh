#!/usr/bin/env bash
# 统一入口：METHOD=plain / beacon / base，BENCHMARK=browsecomp / browsecomp-plus / hotpotqa。
# 其余变量透传原脚本；完整示例和裁判参数见 docs/benchmark/59_llm_judge_matrix.md。
set -euo pipefail
METHOD="${METHOD:-plain}"
BENCHMARK="${BENCHMARK:-hotpotqa}"
SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$METHOD:$BENCHMARK" in
  plain:browsecomp) ENTRY=plain/47_plain_browsecomp.sh ;;
  plain:browsecomp-plus) ENTRY=plain/56_plain_browsecomp_plus.sh ;;
  plain:hotpotqa) ENTRY=plain/57_plain_hotpotqa.sh ;;
  beacon:browsecomp) ENTRY=beacon/47_beacon_browsecomp.sh ;;
  beacon:browsecomp-plus) ENTRY=beacon/56_beacon_browsecomp_plus.sh ;;
  beacon:hotpotqa) ENTRY=beacon/57_beacon_hotpotqa.sh ;;
  base:browsecomp) ENTRY=baseline/72_qwen35_browsecomp.sh ;;
  base:browsecomp-plus) ENTRY=baseline/74_qwen35_browsecomp_plus.sh ;;
  base:hotpotqa) ENTRY=baseline/75_qwen35_hotpotqa.sh ;;
  *) echo 'METHOD 必须为 plain/beacon/base；BENCHMARK 必须为 browsecomp/browsecomp-plus/hotpotqa' >&2; exit 2 ;;
esac
# 上游在线入口使用正式名称识别官方 BrowseComp 数据来源。
if [[ "$BENCHMARK" == browsecomp ]]; then export BENCHMARK=BrowseComp; fi
exec bash "$SCRIPT_ROOT/$ENTRY" "$@"
