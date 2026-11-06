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

# ENVIRONMENT.SCORER_URLS：该字段控制 ENVIRONMENT.SCORER_URLS；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SCORER_URLS="${SCORER_URLS:-}"

# ENVIRONMENT.SCORER_API_KEY：该字段控制 ENVIRONMENT.SCORER_API_KEY；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SCORER_API_KEY="${SCORER_API_KEY:-}"

# ENVIRONMENT.SCORER_MODEL_NAME：该字段控制 ENVIRONMENT.SCORER_MODEL_NAME；约束及推荐值见 docs/60_parameters.md 对应参数表。
export SCORER_MODEL_NAME="${SCORER_MODEL_NAME:-}"

# EVALUATION_DIR：必改：对应 test 脚本输出的测试目录（含 job.json、evaluation.json、results/）
export SA_EVALUATION_DIR=artifacts/experiments/beacon/openseeker/REPLACE_WITH_RUN/evaluation/browsecomp/REPLACE_WITH_EVALUATION

# RUN_NAME：评分版本标签；每次评分单独归档，不覆盖已有结果
export SA_RUN_NAME=judge

# JUDGE.JUDGE_WORKERS：远程裁判并发数，默认 1。
export SA_JUDGE__JUDGE_WORKERS=1

# JUDGE.JUDGE_ATTEMPTS：最多尝试次数，默认 3；解析失败仍保留错误状态和分母。
export SA_JUDGE__JUDGE_ATTEMPTS=3

# JUDGE.JUDGE_MAX_TOKENS：每次裁判输出预算；在线默认 1024，离线默认 4096。
export SA_JUDGE__JUDGE_MAX_TOKENS=1024

# JUDGE.JUDGE_TIMEOUT：单次在线请求超时秒数，默认 120。
export SA_JUDGE__JUDGE_TIMEOUT=120

# JUDGE.JUDGE_ENABLE_THINKING：是否允许 thinking，默认 false；与裁判兼容性一致。
export SA_JUDGE__JUDGE_ENABLE_THINKING=false

# ===== 内部身份与单次 Python 调用；无需修改 =====
export SA_ENTRY=beacon/41_score_browsecomp
export SA_LAUNCHER="$SCRIPT_DIR/41_score_browsecomp.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/beacon/py/41_score_browsecomp.py"
