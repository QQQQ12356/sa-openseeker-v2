# 24 原始 Qwen3.5-4B 独立基线：数据、测试、评分和对照

本指南为 **未经本项目 SFT 的原始 Qwen3.5-4B** 提供独立的 OpenSeeker、在线 BrowseComp、离线 BrowseComp-Plus 和 HotpotQA 评测。先检查环境和权重，再准备相应基准，先测少量题、检查轨迹，然后运行全量。本文不会启动训练。

## 24.1 入口和独立边界

| 入口 | 功能 | 常用 ACTION |
|---|---|---|
| `scripts/sh/baseline/70_qwen35_base_eval.sh` | 总入口，用 TARGET 选择后面的脚本 | 由所选 TARGET 决定 |
| `scripts/sh/baseline/71_qwen35_openseeker.sh` | OpenSeeker 留出/训练 split 生成与 EM/F1 | run、score；download/prepare 使用公共数据脚本 |
| `scripts/sh/baseline/72_qwen35_browsecomp.sh` | 在线 BrowseComp 生成及官方裁判 | prepare、run、score |
| `scripts/sh/baseline/74_qwen35_browsecomp_plus.sh` | 固定语料 BrowseComp-Plus 独立入口 | download、prepare、index、run、score、all |
| `scripts/sh/baseline/75_qwen35_hotpotqa.sh` | 固定语料 HotpotQA 独立入口 | download、prepare、index、run、score、all |
| `scripts/sh/baseline/73_qwen35_offline.sh` | 74/75 的共享配置与兼容入口 | 同上，TARGET 选择数据集 |

70 仅分发到同目录的 71/72/74/75（74/75 共用 73）；它们不调用 `scripts/sh/plain/` 下的训练或评测脚本。原始模型通过 `search_comp/models/base_qwen35.py` 检查原始多模态 Qwen3.5 配置，再使用文本主干加载器；不挂载 LoRA，不调用 Plain SFT 模型加载器。模型运行时使用的 `kind=plain` 仅表示无 Beacon 压缩的 agent 模式，并非选择 Plain 训练模型。

为确保实验可比，数据适配、序列化、search/visit 执行、预算、预测和评分仍与训练模型评测共用通用实现：`scripts/py/50_evaluate.py`、`47_benchmark.py`、`52_offline_benchmark.py`。运行时传 `--model-origin base-qwen35`，结果的生成配置记录 `model_origin` 和 `base_model_path`。共享评测器不等于加载训练模型。

原始 Qwen3.5-4B HF checkpoint 是多模态结构；本项目提取其文本主干做文本 agent 测试。基线加载器拒绝 adapter、Beacon 和 `qwen3_5_text` 训练输出布局。**检查架构不能证明权重从未被微调**：务必自行记录原始模型版本、下载来源和文件指纹，使用与训练基座相同版本；不要将已训练模型目录作为原始基线。

## 24.2 环境、模型及公共变量

```bash
cd /home/huangzj/proj/searchagent/sa-openseeker-v2
conda activate search-comp-qwen35-cu126
export PYTHON="$(command -v python)"
# 原始 Qwen/Qwen3.5-4B 的完整本地 snapshot；需包含 config、tokenizer、权重及索引。
export BASE_CHECKPOINT=/models/Qwen3.5-4B
# 可选：训练实验目录，仅用于取得同一 OpenSeeker 数据路径或提示 profile。
export RUN_DIR=/absolute/path/to/your/plain_run
```

71 允许 `BASE_CHECKPOINT=Qwen/Qwen3.5-4B`，前提是环境能通过本地缓存或联网加载；72/73 实际生成时要求本地完整目录。加载原始多模态权重并提取文本模型需要足够主机内存、磁盘与 GPU。脚本不会下载模型，也不会验证硬件资源。

| 变量 | 默认 | 修改与注意事项 |
|---|---|---|
| TARGET | 70 为 openseeker；73 为 hotpotqa | 70：openseeker、browsecomp、browsecomp-plus、hotpotqa；直调 73 只接受后两者 |
| ACTION | run | 各入口支持的阶段见上表；`all` 是离线准备/建索引/生成/评分，不下载 |
| BASE_CHECKPOINT | `Qwen/Qwen3.5-4B` | 唯一模型输入；71 可用 HF ID，72/73 的实际 run/all 用本地目录 |
| RUN_DIR | 空 | 71 用其 `experiment.json` 找 split；72/73 用其训练文件找提示词，不读取其 final 模型 |
| PYTHON / CONDA_ENV | 空 / `search-comp-qwen35-cu126` | 可执行 PYTHON 优先；否则找同名 conda 环境，再回退当前 Python |
| CUDA_VISIBLE_DEVICES | 默认 0 | 在 71/73 设置可见 GPU；72 也可由外部环境设置 |
| DRY_RUN | 0 | run/score 等审查参数；不加载模型、确认服务或证明 checkpoint 可用。71 的 download/prepare 不支持 dry-run |
| DECODING / REPETITION_PENALTY | greedy / 1.0 | greedy 或 sample；1 不惩罚，>1 惩罚出现过的 token |
| TEMPERATURE / TOP_P / GENERATION_SEED | 1.0 / 1.0 / 42 | 仅采样有效；生成 seed 不等于 BCP 裁判 SEED |

各脚本支持环境变量覆盖，可在命令前临时赋值，或编辑脚本顶部 `${变量:-默认值}`；已有非空环境变量优先。参数附在 shell 命令末尾会传给通用 Python 入口，避免传入与脚本重复的 `--model`、`--model-origin` 或 `--kind`，以免改变对照身份。不要用 `MODEL_PATH` 选择基线；基线脚本使用 `BASE_CHECKPOINT`。

RUN_DIR 只提供数据或提示来源，不是运行基座所必需的。迁移后原训练路径不可用时：71 显式设置 `DATA`；72/73 使用 `PROMPT_DATA=/path/to/original_train.jsonl`，或不设 RUN_DIR 采用内置 Polar 提示。离线测试仍针对本地工具与答案格式适配 system；详情见 [45](../common/45_unified_agent_flow.md)。

## 24.3 OpenSeeker 轨迹诊断：71

### 准备数据

71 不构造专用训练数据，读取已有 OpenSeeker prepared JSONL。确认同一 protocol、问题分组和 tokenizer，见 [10](../data/10_data.md)。若需重建，使用公共数据处理入口：

```bash
# 先编辑 16_download.sh 固定 revision；已有 raw 时跳过。
ACTION=download bash scripts/sh/baseline/71_qwen35_openseeker.sh
# 新目录，准备 tokenizer 必须和模型版本一致。
INPUT_FILE=/data/polar/openseeker_v1_data.jsonl \
OUTPUT_DIR=/data/prepared/polar_qwen35_25k MODEL_NAME="$BASE_CHECKPOINT" \
MAX_LENGTH=25000 ACTION=prepare bash scripts/sh/baseline/71_qwen35_openseeker.sh
```

公共数据入口的 OUTPUT_DIR 必须不存在；prepare 只加载 tokenizer。`DRY_RUN=1` 对这两个阶段会报错，不会悄悄运行。已经有合适 prepared 数据就直接复用，不必重新处理。

### 生成及评分

```bash
# 默认使用 RUN_DIR 对应 train split；先只检查命令。
ACTION=run RUN_DIR="$RUN_DIR" SPLIT=train LIMIT=8 DRY_RUN=1 \
  bash scripts/sh/baseline/71_qwen35_openseeker.sh

# 保持同一数据、预算、工具环境与 adapter 对比。
ACTION=run RUN_DIR="$RUN_DIR" SPLIT=train LIMIT=8 BACKEND=replay \
  bash scripts/sh/baseline/71_qwen35_openseeker.sh

# 或总入口：
# TARGET=openseeker ACTION=run RUN_DIR="$RUN_DIR" SPLIT=test LIMIT=8 \
#   bash scripts/sh/baseline/70_qwen35_base_eval.sh
```

71 默认将预测、`.metrics.json` 和日志写在 `outputs/results/qwen35_base/openseeker/` 的时间戳文件中，不写训练实验目录。它生成时已经计算 EM/F1；需要重新汇总时指定新输出：

```bash
ACTION=score PREDICTIONS=/path/to/predictions_test.jsonl \
OUTPUT_PATH=/path/to/new_summary.json bash scripts/sh/baseline/71_qwen35_openseeker.sh
```

| 71 变量 | 默认 | 用法 |
|---|---|---|
| DATA / SPLIT | 自动 / test | DATA 覆盖 RUN_DIR 的 train/validation/test 路径；无二者使用默认 16K prepared split |
| BACKEND | replay | replay 仅在生成动作与历史完全一致时返回观察，不代表自由搜索能力 |
| CORPUS / ENDPOINT | 空 | corpus 后端须提供 JSONL；http 后端须提供可访问的工具网关 |
| LIMIT | 100 | 正整数；选择前 N 条，不能设 -1 |
| NUM_WORKERS / PROGRESS_INTERVAL | 1 / 15 秒 | 每卡完整模型副本，分片按题；多卡须设置互异 CUDA_VISIBLE_DEVICES |
| MAX_CONTEXT / MAX_NEW_TOKENS | 16384 / 8192 | 完整逻辑历史与每轮生成预算 |
| MAX_TOOL_TOKENS / MAX_TURNS | 16384 / 32 | 单条工具返回与最大轮次 |
| OUTPUT_DIR / OUTPUT / LOG_PATH | `outputs/results/qwen35_base/openseeker` / 自动 / 自动 | 各次独立输出，已有预测文件由评测器拒绝覆盖 |
| PREDICTIONS / OUTPUT_PATH | 空 / 空 | score 的预测输入和新汇总 JSON |

只看第一轮可帮助分析复读；如果输出在第一次工具调用前已反复推理，工具结果不是该轮的直接诱因。`BACKEND=corpus` 允许自由查询，但历史证据语料不是独立搜索 benchmark；详见 [40](../benchmark/40_evaluation.md)。

## 24.4 在线 BrowseComp：72

### 准备与配置

```bash
# 准备固定版本 OpenSeeker 代码与 BrowseComp 问题，需网络。
ACTION=prepare bash scripts/sh/baseline/72_qwen35_browsecomp.sh

mkdir -p "$HOME/.config/searchagent"
cp scripts/sh/benchmark/49_benchmark_env.example.sh "$HOME/.config/searchagent/benchmark.env.sh"
# 在私有副本设置搜索、Jina、摘要及裁判服务配置。
```

native 直接加载原始基座并调用固定官方 v1 Search/Visit，不需要 OPENSEEKER 模型服务或 E2B。搜索、网页读取、摘要与裁判需要相应在线服务；接口、key、模型和网络连通性见 [42](../benchmark/42_openseeker_benchmark.md)。准备阶段可用 `DRY_RUN=1 ACTION=prepare` 审查，不会下载。

```bash
# 显示配置；dry-run 不验证模型或服务。
ACTION=run LIMIT=2 DRY_RUN=1 RUN_DIR="$RUN_DIR" \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/baseline/72_qwen35_browsecomp.sh

# 实际运行访问在线服务并自动裁判，可能产生服务费用。
ACTION=run LIMIT=2 RUN_DIR="$RUN_DIR" \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/baseline/72_qwen35_browsecomp.sh
```

用 `LIMIT=-1` 全量；确保 BASE_CHECKPOINT 是本地原始模型目录。72 固定 native，不提供 official-v1/v2 模式；如要原样官方复现，使用原有 47/48 流程并作为独立实验。独立重评分：

```bash
ACTION=score LIMIT=2 PREDICTIONS=/path/to/result_tool200.jsonl \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/baseline/72_qwen35_browsecomp.sh
```

| 72 变量 | 默认 | 用法 |
|---|---|---|
| DATA / BENCHMARK | 默认 BrowseComp JSONL / BrowseComp | 准备或评测问题；DATA 与 LIMIT 必须与重评分一致 |
| ENV_FILE | 空 | 私有环境配置文件；不要将密钥提交到仓库 |
| PROMPT_DATA | 空 | 原训练 system/schema JSONL，优先于 RUN_DIR；两者无时使用内置 Polar profile |
| LIMIT | -1 | 全量；先设正数 smoke |
| MAX_TOKENS / MAX_CONTEXT | 32768 / 131072 | 每轮生成及完整逻辑历史上限 |
| MAX_TURNS / MAX_TOOL_TOKENS | 200 / 32768 | native 最大轮次和单条工具返回预算 |
| TOOL_COUNT | 200 | 官方结果文件名参数；native 实际循环上限为 MAX_TURNS |
| WORKERS / JUDGE_WORKERS | 60 / 20 | 前者不使 native 自动多卡；后者为裁判并发 |
| PREDICTIONS / OUTPUT_DIR | 空 / 自动 | score 输入 / 本次新结果目录；禁止覆盖旧结果 |

默认结果写入 `outputs/results/qwen35_base/browsecomp/` 下按阶段和时间命名的新目录，即使设置 RUN_DIR 也不会写入训练目录。检查 `result_tool*.jsonl`、`coverage.json`、`summary.metrics.json` 和 `judge.log`；必须报告全部题数和裁判覆盖。

## 24.5 固定语料：73

`TARGET=hotpotqa` 使用有标注 dev 和独立 2017 Wikipedia 摘要库；`TARGET=browsecomp-plus` 使用官方问题及固定网页语料。两个 ROOT_DIR 必须独立。下载阶段联网，prepare/index/run/score/all 离线；语料和索引可被原始模型和已训练模型共同复用。默认结果写入 `outputs/results/qwen35_base/$TARGET/`，即使设置 RUN_DIR 也不会写入训练目录。

### HotpotQA

```bash
TARGET=hotpotqa ACTION=download bash scripts/sh/baseline/73_qwen35_offline.sh
TARGET=hotpotqa ACTION=prepare bash scripts/sh/baseline/73_qwen35_offline.sh
TARGET=hotpotqa ACTION=index bash scripts/sh/baseline/73_qwen35_offline.sh
TARGET=hotpotqa ACTION=run RUN_DIR="$RUN_DIR" LIMIT=8 DRY_RUN=1 \
  bash scripts/sh/baseline/73_qwen35_offline.sh
TARGET=hotpotqa ACTION=run RUN_DIR="$RUN_DIR" LIMIT=8 \
  bash scripts/sh/baseline/73_qwen35_offline.sh

# 把终端打印的生成目录填入，使用相同 ROOT_DIR 和 LIMIT；score 不加载模型。
GEN_DIR=/absolute/path/to/hotpot_base_generation
TARGET=hotpotqa ACTION=score LIMIT=8 PREDICTIONS="$GEN_DIR/predictions.jsonl" \
  bash scripts/sh/baseline/73_qwen35_offline.sh
```

Hotpot 默认开启本地 search/visit；评分为官方 EM/F1、支持事实和 Joint，不需要裁判模型。模型需输出自己识别的支持句才能得到有意义的 SP/Joint；标注不输入模型。

### BrowseComp-Plus

```bash
TARGET=browsecomp-plus ACTION=download bash scripts/sh/baseline/73_qwen35_offline.sh
TARGET=browsecomp-plus ACTION=prepare bash scripts/sh/baseline/73_qwen35_offline.sh
TARGET=browsecomp-plus ACTION=index bash scripts/sh/baseline/73_qwen35_offline.sh
TARGET=browsecomp-plus ACTION=run RUN_DIR="$RUN_DIR" LIMIT=8 \
  bash scripts/sh/baseline/73_qwen35_offline.sh

GEN_DIR=/absolute/path/to/bcp_base_generation
TARGET=browsecomp-plus ACTION=score LIMIT=8 PREDICTIONS="$GEN_DIR/predictions.jsonl" \
JUDGE_MODEL=/models/Qwen3-32B bash scripts/sh/baseline/73_qwen35_offline.sh
```

默认 search-only；`ALLOW_VISIT=1` 属于扩展协议。完整答案评分需要本地裁判，`RETRIEVAL_ONLY=1` 只计算检索诊断而非答案 accuracy。保持 score 与 run 的 ROOT_DIR、LIMIT 和题目一致，OUTPUT_DIR 必须新建。

### 73 参数及修改提示

| 变量 | 默认 | 含义 |
|---|---|---|
| ROOT_DIR | `outputs/data/benchmarks/$TARGET` | 包含 raw、prepared、official 和 index.sqlite；两个 benchmark 分目录 |
| DATA / CORPUS | 空 | 分别覆盖官方问题与独立语料默认路径；不使用题目正样本构成全库 |
| CORPUS_LIMIT / OFFICIAL_ONLY | 空 / 0 | 正数只作语料 smoke；1 只准备官方附带文件 |
| TITLE_WEIGHT / VERIFY_INDEX | 1.0 / 0 | 建索引的标题权重 / 运行前完整哈希校验；权重变更需新索引 |
| TOPK / SNIPPET_TOKENS / READ_TOKENS | 5 / 512 / 4096 | 每查询返回条数、每篇片段和 visit 正文 token 上限 |
| ALLOW_VISIT | Hotpot 1，BCP 0 | 仅访问 `local://`，不联网 |
| TOOL_COUNT / MAX_TURNS | 200 / 200 | 每题工具次数与生成轮数上限 |
| MAX_CONTEXT / MAX_NEW_TOKENS / MAX_TOOL_TOKENS | 131072 / 8192 / 32768 | 逻辑历史、单轮生成与单条工具结果预算 |
| LIMIT / NUM_WORKERS | -1 / 1 | -1 全量；每个 worker 在一张可见 GPU 加载完整模型，不是 tensor parallel |
| OUTPUT_DIR / PREDICTIONS | 自动 / 空 | 新结果目录 / score 输入；禁止覆盖生成或评分目录 |
| DECRYPTED | 0 | BCP 自备明文问题时设 1 |
| JUDGE_MODEL / RETRIEVAL_ONLY | 空 / 0 | BCP 本地裁判路径 / 仅检索 |
| JUDGE_MAX_TOKENS / JUDGE_MAX_CONTEXT | 4096 / 32768 | BCP 裁判生成和上下文预算 |
| JUDGE_TEMPERATURE / JUDGE_TOP_P / JUDGE_TOP_K / SEED | 0.7 / 0.8 / 20 / 42 | BCP 裁判采样，与待测模型参数独立 |

详细来源、空间需求、数据版本、checksum、索引、断网限制和故障排查见 [43](../benchmark/43_offline_benchmarks.md)、[44](../benchmark/44_offline_sources.md)。实际 download 会联网；完整数据与索引可能占用较多磁盘。

## 24.6 结果检查与可比性

1. 对同一测试使用相同题目顺序、提示 profile、工具配置、语料与索引、tokenizer、解码和预算；每组写独立结果。
2. 检查 `status`、`messages`、`events`、`generated_tokens` 和工具调用；第一轮复读、工具后复读和多轮重复分别分析。
3. OpenSeeker 使用严格 EM/F1；Hotpot 使用官方答案/支持事实；BCP 使用官方裁判提示、本地裁判及累计检索召回；在线 BrowseComp 使用在线裁判。不同 benchmark 的指标不可直接互换。
4. 保存 `BASE_CHECKPOINT` 的来源/版本/文件 hash、题目与语料版本、system hash、生成参数、工具服务、裁判、题数和失败分母。模型架构检查不替代模型权重来源核验。
5. adapter 和合并模型的训练后对照使用 [22](22_qwen35_baseline_and_adapter.md) 的 21/47/56/57；解码和重复惩罚控制见 [23](../common/23_decoding.md)。

本指南给出可执行流程，但真实模型生成质量、显存、耗时、外网服务及全量基准成绩需要在目标服务器实际验证。

各方案独立 Benchmark 入口、完整变量表、配置优先级与路径迁移见 [46 脚本手册](../benchmark/46_benchmark_scripts.md)。

离线基准现有数据集专属入口：`scripts/sh/baseline/74_qwen35_browsecomp_plus.sh`、`scripts/sh/baseline/75_qwen35_hotpotqa.sh`。它们固定 TARGET 并复用本目录 73 的配置；下文/原有 `TARGET=... 73` 命令仍兼容，70 总入口已使用专属入口。完整变量与修改提示见 [46 脚本手册](../benchmark/46_benchmark_scripts.md)。
