#!/usr/bin/env bash
# BrowseComp-Plus 向量索引准备。运行说明见 docs/35_browsecomp_plus_dense.md。
set -euo pipefail

# Linux Python 解释器，默认使用已激活环境。
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux；Windows 仅修改代码。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# Hub 地址，可改用可访问的镜像。
export HF_ENDPOINT=https://huggingface.co
# download 使用 0。prepare 使用本地文件，可设为 1。
export HF_HUB_OFFLINE=0
# 仅自行编码需要 GPU。选择物理 GPU 编号。
export CUDA_VISIBLE_DEVICES=0

# BrowseComp-Plus 数据版本，与 31_prepare_browsecomp_plus.sh 一致。
export SA_VERSION=v1
# 向量索引版本名。切换来源、模型或编码配置时改名，禁止覆盖。
export SA_INDEX_PROFILE=qwen3-embedding-8b-official
# 先 download 下载资产，再 prepare 导入向量或编码本地语料。
export SA_PHASE=download
# official 导入官方 4 个分片；encode 对 prepared/corpus.jsonl 逐批编码。
export SA_SOURCE=official
# download 阶段的模型 revision，main 会被解析为实际 commit。
export SA_MODEL_REVISION=main
# download 阶段的官方向量 revision，仅 official 使用。
export SA_INDEX_REVISION=980cfd0c8250482e8be900326fddca806511d05c
# prepare 必填。填下载输出的本地 embedding 模型 commit 目录。
export SA_EMBEDDING_MODEL_PATH=''
# official 的 prepare 必填。填下载输出的官方向量快照目录。
export SA_OFFICIAL_INDEX_DIR=''
# encode 的模型设备。cuda:0 指第一张可见 GPU；也可使用 cpu。
export SA_DEVICE=cuda:0
# encode 批大小。显存不足先减小此值。
export SA_BATCH_SIZE=4
# encode 单篇正文 token 上限，最大 32768。只影响用于生成向量的文本。
export SA_DOC_MAX_LENGTH=4096

export SA_ENTRY=data/33_prepare_browsecomp_plus_dense
export SA_LAUNCHER="$SCRIPT_DIR/33_prepare_browsecomp_plus_dense.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/data/py/33_prepare_browsecomp_plus_dense.py"
