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

# SOURCE：固定 browsecomp-sft，产物使用 Research 训练入口。
export SA_SOURCE=browsecomp-sft

# SNAPSHOT_DIR：必改：下载输出的 commit 目录；自动读取清单内全部 shard；与 INPUTS 互斥
export SA_SNAPSHOT_DIR=''

# INPUTS：本地 JSONL/Parquet 路径列表；使用它时将 SNAPSHOT_DIR 留空
export SA_INPUTS='[]'

# SHARDS：使用快照时选择 Parquet 文件；默认仅预处理辅助轨迹。
export SA_SHARDS='["syn_planner.parquet"]'

# CANONICAL_QUESTIONS：处理 data.parquet 时必填，用于把改写题对齐到同一个原题。
# 填 BrowseComp 版本目录的 questions.jsonl；只处理辅助分片时可留空。
export SA_CANONICAL_QUESTIONS=''

# PROTECTED_QUESTIONS：列出实验保留的评测题 JSONL，重合轨迹在计长前剔除。
# 用 30/31/32 脚本准备题目。仍需独立评测时请保留默认的三套题目。
export SA_PROTECTED_QUESTIONS='["artifacts/benchmarks/browsecomp/v1/questions.jsonl","artifacts/benchmarks/browsecomp-plus/v1/prepared/questions.jsonl","artifacts/benchmarks/hotpotqa/v1/prepared/questions.jsonl"]'

# OVERLAP_THRESHOLD：(0,1]，四词片段覆盖率阈值。较低值更容易检出改写，也会增加误报。
export SA_OVERLAP_THRESHOLD=0.30

# PROFILE：数据版本名；已有版本禁止覆盖；更换模型、筛选条件或划分时使用新名称
export SA_PROFILE=browsecomp_sft_planner_qwen35_L16384_s42

# MODELS：必改：与训练 MODEL_PATH 使用完全相同的本地 tokenizer 路径；只加载 tokenizer
export SA_MODELS='["/models/Qwen3.5-4B"]'

# MIN_LENGTH：完整轨迹 token 下限，至少 1
export SA_MIN_LENGTH=1

# MAX_LENGTH：完整轨迹 token 上限；超长过滤或报错，不静默截断
export SA_MAX_LENGTH=16384

# CORRECTNESS：上游无独立正确性列；用 any 保留可用轨迹。
export SA_CORRECTNESS=any

# OBSERVATIONS：any/with/without：工具观测筛选
export SA_OBSERVATIONS=with

# WORKERS：CPU 分词进程数，不需要 GPU
export SA_WORKERS=4

# SEED：按题目哈希分组划分的随机种子。
export SA_SEED=42

# VAL_FRACTION：验证集目标比例，允许 0
export SA_VAL_FRACTION=0.02

# TEST_FRACTION：保留轨迹测试集目标比例；与验证比例之和小于 1；不属于独立 benchmark
export SA_TEST_FRACTION=0.02

# OFFLINE：true 只使用本地 tokenizer；false 允许下载 tokenizer
export SA_OFFLINE=true

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=data/25_prepare_browsecomp_sft
export SA_LAUNCHER="$SCRIPT_DIR/25_prepare_browsecomp_sft.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/data/py/25_prepare_browsecomp_sft.py"
