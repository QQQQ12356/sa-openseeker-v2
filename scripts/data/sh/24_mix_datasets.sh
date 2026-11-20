#!/usr/bin/env bash
# 编辑本文件后运行。完整说明见 docs/20_data.md。
set -euo pipefail

# Python 解释器。可填写 Linux 环境的 bin/python 绝对路径。
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# 父版本目录。填写任意两个或三个来源的预处理或二次筛选目录，其余留空。
export SA_OPENSEEKER_PREPARED=artifacts/datasets/openseeker/prepared/qwen35_L16384_s42
export SA_RESEARCH_PREPARED=artifacts/datasets/research/prepared/qwen35_L16384_s42
# BrowseComp SFT 父版本目录。填入 25 或 26 脚本输出的目录。
export SA_BROWSECOMP_SFT_PREPARED=''
# 输出版本名。已有同名版本时请改名。
export SA_PROFILE=qwen35_L16384_mix_1to1_s42
# 各父版本共同使用的 tokenizer 清单键，与训练 SA_MODEL_PATH 一致。
export SA_TOKENIZER=/models/Qwen3.5-4B
# 完整轨迹 token 上限。超长整条排除。训练 max_length 至少取此值。
export SA_MAX_LENGTH=16384
# 比例依次对应已填写的 OpenSeeker、Research、BrowseComp SFT，跳过空项。例如两来源 1:1，三来源 2:1:1。
export SA_RATIO=1:1
# 控制无放回抽样与训练集输出顺序。
export SA_SEED=42

export SA_ENTRY=data/24_mix_datasets
export SA_LAUNCHER="$SCRIPT_DIR/24_mix_datasets.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/data/py/24_mix_datasets.py"
