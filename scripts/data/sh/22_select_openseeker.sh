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
export HF_HUB_OFFLINE=1

# ENVIRONMENT.CUDA_VISIBLE_DEVICES：GPU 物理编号；多卡示例 0,1,2,3
export CUDA_VISIBLE_DEVICES=0

# PREPARED：父数据版本；必须保留 normalized.jsonl 和原始输入，用于恢复公开划分
export SA_PREPARED=artifacts/datasets/openseeker/prepared/qwen35_L16384_s42

# PROFILE：新数据版本名，不能覆盖父版本
export SA_PROFILE=qwen35_L8192_s42

# TOKENIZER：必须出现在父数据 tokenizers 清单内
export SA_TOKENIZER=/models/Qwen3.5-4B

# MIN_LENGTH：完整轨迹 token 下限，至少 1
export SA_MIN_LENGTH=1

# MAX_LENGTH：完整轨迹 token 上限；超长过滤或报错，不静默截断
export SA_MAX_LENGTH=8192

# EXACT_LENGTH：可选精确 token 长度；空字符串 不限定
export SA_EXACT_LENGTH=''

# CORRECTNESS：Correct/Incorrect/any；Research 只能 any
export SA_CORRECTNESS=Correct

# OBSERVATIONS：any/with/without：工具观测筛选
export SA_OBSERVATIONS=any

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=data/22_select_openseeker
export SA_LAUNCHER="$SCRIPT_DIR/22_select_openseeker.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/data/py/22_select_openseeker.py"
