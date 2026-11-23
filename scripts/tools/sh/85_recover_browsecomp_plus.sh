#!/usr/bin/env bash
# 汇总中断的 BrowseComp-Plus 测试并评分。使用说明见 docs/36_browsecomp_plus_resume.md。
set -euo pipefail

# Python 解释器，默认使用当前激活的 Linux 环境。
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此运行入口仅支持 Linux；Windows 仅修改代码。' >&2; exit 2; }
(( $# == 0 )) || { echo '请直接编辑 SH 参数，不接受命令行附加参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

# 评分离线执行。检索指标无需 GPU；答案评分时只暴露获分配的 GPU。
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# 必填。填 42 的测试记录根目录，包含 job.json 和 results/。先确保所有 worker 已退出。
export SA_EVALUATION_DIR=''
# 恢复评分标签。每次在原测试的 recovery/ 下创建独立记录。
export SA_RUN_NAME=partial_score
# 仅对已恢复的预测计分。true 只算检索指标；false 同时用本地裁判计算答案准确率。
export SA_RETRIEVAL_ONLY=true

# 仅答案评分必填。使用与正式评分相同的本地裁判模型目录。
export SA_JUDGE__JUDGE_MODEL=''
# 每条答案的最大裁判尝试次数。
export SA_JUDGE__JUDGE_ATTEMPTS=3
# 裁判输出 token 上限和总上下文 token 上限。
export SA_JUDGE__JUDGE_MAX_TOKENS=4096
export SA_JUDGE__JUDGE_MAX_CONTEXT=32768
# 是否启用裁判模型的 thinking 模式。
export SA_JUDGE__JUDGE_ENABLE_THINKING=false
# 裁判采样参数和随机种子。与正式评分对比时保持一致。
export SA_JUDGE__JUDGE_TEMPERATURE=0.0
export SA_JUDGE__JUDGE_TOP_P=0.8
export SA_JUDGE__JUDGE_TOP_K=20
export SA_JUDGE__SEED=42

export SA_ENTRY=tools/85_recover_browsecomp_plus
export SA_LAUNCHER="$SCRIPT_DIR/85_recover_browsecomp_plus.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/py/85_recover_browsecomp_plus.py"
