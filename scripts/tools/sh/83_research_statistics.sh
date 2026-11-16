#!/usr/bin/env bash
# 独立入口：只修改本文件参数；对应 py 文件负责执行。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux；Windows 仅修改代码。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# 必填，指向 11_download_research.sh 打印的快照目录。
export SA_SNAPSHOT_DIR=''

# 与 Research 预处理使用同一个本地 tokenizer 路径，无需模型权重。
export HF_HUB_OFFLINE=1
export SA_TOKENIZER=/models/Qwen3.5-4B

# 与待诊断的预处理配置保持一致。
export SA_MAX_LENGTH=16384
export SA_OBSERVATIONS=with
export SA_WORKERS=4

export SA_ENTRY=tools/83_research_statistics
export SA_LAUNCHER="$SCRIPT_DIR/83_research_statistics.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/83_research_statistics.py"
