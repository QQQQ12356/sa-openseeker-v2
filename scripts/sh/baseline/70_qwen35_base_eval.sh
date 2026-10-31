#!/usr/bin/env bash
# 原始 Qwen3.5 基线总入口；只调用 baseline 目录内的独立脚本。
# TARGET=openseeker / browsecomp / browsecomp-plus / hotpotqa
# 使用和参数：docs/baseline/22_qwen35_baseline_and_adapter.md。
set -euo pipefail
TARGET="${TARGET:-openseeker}"
export TARGET
BASELINE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$TARGET" in
  openseeker) exec bash "$BASELINE_DIR/71_qwen35_openseeker.sh" "$@" ;;
  browsecomp) exec bash "$BASELINE_DIR/72_qwen35_browsecomp.sh" "$@" ;;
  browsecomp-plus) exec bash "$BASELINE_DIR/74_qwen35_browsecomp_plus.sh" "$@" ;;
  hotpotqa) exec bash "$BASELINE_DIR/75_qwen35_hotpotqa.sh" "$@" ;;
  *) echo "未知 TARGET=$TARGET；可选 openseeker、browsecomp、browsecomp-plus、hotpotqa" >&2; exit 2 ;;
esac
