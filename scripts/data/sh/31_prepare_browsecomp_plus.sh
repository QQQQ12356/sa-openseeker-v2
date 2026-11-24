#!/usr/bin/env bash
# 独立入口：只修改本文件参数；对应 py 文件负责执行。完整说明：docs/60_parameters.md。
set -euo pipefail

# Python 解释器；先激活 Linux 环境，或填写其 bin/python 绝对路径。
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux；Windows 仅修改代码。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# ===== 可编辑参数：该字段控制 ===== 可编辑参数；约束及推荐值见 docs/60_parameters.md 对应参数表。
# ENVIRONMENT.HF_ENDPOINT：Hugging Face 来源；可改为 https://hf-mirror.com。
export HF_ENDPOINT=https://huggingface.co

# ENVIRONMENT.HF_HUB_OFFLINE：1 只用本地文件/缓存；下载脚本必须为 0
export HF_HUB_OFFLINE=0

# ENVIRONMENT.CUDA_VISIBLE_DEVICES：GPU 物理编号；多卡示例 0,1,2,3
export CUDA_VISIBLE_DEVICES=0

# VERSION：共享 benchmark 版本名；改变语料/来源/索引权重时新建版本
export SA_VERSION=v2

# PHASE：依次运行 download（题目、语料、qrels、预建 Lucene BM25）、prepare（解密与规范化）、index（文档库）。
export SA_PHASE=download

# DATA：可选本地官方题目路径；空字符串 使用 download 阶段的默认文件
export SA_DATA=''

# CORPUS：可选独立本地语料；空字符串 使用下载的全量语料，不能用题目候选段落替代
export SA_CORPUS=''

# 三项 revision 留空时首次下载取各自 main，后续按 assets.json 复用。
export SA_QUERY_REVISION=''
export SA_CORPUS_REVISION=''
export SA_INDEX_REVISION=''

# DECRYPTED：BrowseComp-Plus 题目已解密时改 true；HotpotQA 保持 false
export SA_DECRYPTED=false

# TITLE_WEIGHT：旧版或自定义语料的 SQLite BM25 标题权重。
export SA_TITLE_WEIGHT=1.0

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=data/31_prepare_browsecomp_plus
export SA_LAUNCHER="$SCRIPT_DIR/31_prepare_browsecomp_plus.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/data/py/31_prepare_browsecomp_plus.py"
