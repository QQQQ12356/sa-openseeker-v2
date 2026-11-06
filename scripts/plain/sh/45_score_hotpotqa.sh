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

# EVALUATION_DIR：必改：对应 test 脚本输出的测试目录（含 job.json、evaluation.json、results/）
export SA_EVALUATION_DIR=artifacts/experiments/plain/openseeker/REPLACE_WITH_RUN/evaluation/hotpotqa/REPLACE_WITH_EVALUATION

# RUN_NAME：评分版本标签；每次评分单独归档，不覆盖已有结果
export SA_RUN_NAME=judge

# JUDGE.JUDGE_MODEL：本地裁判目录，默认占位 /models/judge，执行前必改；不能填在线端点。
export SA_JUDGE__JUDGE_MODEL=/models/judge

# JUDGE.LLM_JUDGE：默认 true，加语义裁判；需要本地 judge_model。仅算官方 EM/F1 时同时设 false 和 judge_model=空字符串。
export SA_JUDGE__LLM_JUDGE=true

# JUDGE.JUDGE_ATTEMPTS：最多尝试次数，默认 3；解析失败仍保留错误状态和分母。
export SA_JUDGE__JUDGE_ATTEMPTS=3

# JUDGE.JUDGE_MAX_TOKENS：每次裁判输出预算；在线默认 1024，离线默认 4096。
export SA_JUDGE__JUDGE_MAX_TOKENS=4096

# JUDGE.JUDGE_MAX_CONTEXT：本地裁判上下文上限，默认 32768。
export SA_JUDGE__JUDGE_MAX_CONTEXT=32768

# JUDGE.JUDGE_ENABLE_THINKING：是否允许 thinking，默认 false；与裁判兼容性一致。
export SA_JUDGE__JUDGE_ENABLE_THINKING=false

# JUDGE.JUDGE_TEMPERATURE：裁判采样温度，默认 0。
export SA_JUDGE__JUDGE_TEMPERATURE=0.0

# JUDGE.JUDGE_TOP_P：本地裁判 nucleus 采样阈值。
export SA_JUDGE__JUDGE_TOP_P=0.8

# JUDGE.JUDGE_TOP_K：本地裁判 top-k 采样大小。
export SA_JUDGE__JUDGE_TOP_K=20

# JUDGE.SEED：训练随机种子
export SA_JUDGE__SEED=42

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=plain/45_score_hotpotqa
export SA_LAUNCHER="$SCRIPT_DIR/45_score_hotpotqa.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/plain/py/45_score_hotpotqa.py"
