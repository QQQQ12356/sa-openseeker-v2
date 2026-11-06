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

# REPO_ID：可改为训练基座或本地裁判对应的 Hugging Face 仓库
export SA_REPO_ID=Qwen/Qwen3.5-4B

# REVISION：程序解析并记录不可变 commit；正式复现建议填 commit
export SA_REVISION=main

# MODEL_NAME：本地归档名称；修改 REPO_ID 时同步修改
export SA_MODEL_NAME=qwen35-4b

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=data/01_download_model
export SA_LAUNCHER="$SCRIPT_DIR/01_download_model.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/data/py/01_download_model.py"
