#!/usr/bin/env bash
# 独立入口：只修改本文件参数；对应 py 文件负责执行。完整说明：docs/60_parameters.md。
set -euo pipefail

# Python 解释器；先激活 Linux 环境，或填写其 bin/python 绝对路径。
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux；Windows 仅修改代码。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# ===== 可编辑参数：该字段控制 ===== 可编辑参数；约束及推荐值见 docs/60_parameters.md 对应参数表。
# ENVIRONMENT_DIR：独立环境，不覆盖系统 Python
export SA_ENVIRONMENT_DIR=artifacts/environments/train

# TORCH_SPEC：沿用 tmp 的 CUDA 12.6 基线
export SA_TORCH_SPEC=torch==2.7.0+cu126

# TORCH_INDEX：PyTorch CUDA wheel 源
export SA_TORCH_INDEX=https://download.pytorch.org/whl/cu126

# PIP_INDEX：可修改为服务器可访问的 PyPI 镜像
export SA_PIP_INDEX=https://pypi.org/simple

# HF_ENDPOINT：可改为 https://hf-mirror.com
export HF_ENDPOINT=https://huggingface.co

# INSTALL_DEEPSPEED：多卡 ZeRO 训练需 true；单卡/DDP 可保持 false
export SA_INSTALL_DEEPSPEED=false

# DEEPSPEED_SPEC：生产复现建议在 Linux 验证后固定具体版本
export SA_DEEPSPEED_SPEC='deepspeed>=0.15,<1'

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=tools/00_setup_environment
export SA_LAUNCHER="$SCRIPT_DIR/00_setup_environment.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/00_setup_environment.py"
