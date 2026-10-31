# 22 Plain adapter 与 Qwen3.5-4B 原始基座对照评测

本指南用于分析训练前后复读、工具调用和答案质量。OpenSeeker 使用 21 的 replay/corpus/http 流程；在线 BrowseComp、离线 BrowseComp-Plus 和 HotpotQA 使用各自原有 benchmark。所有命令从项目根目录执行；本文不会自动启动训练。

## 22.1 支持范围和对照关系

| 对象 | OpenSeeker 21 | 在线 BrowseComp 47 | 离线 BCP 56 / Hotpot 57 |
|---|---|---|---|
| 原始 Qwen3.5-4B 基座 | 支持 HF ID 或本地目录 | 本地完整目录，native | 本地完整目录 |
| Plain LoRA adapter / checkpoint | **直接加载原基座并挂载 adapter** | 先合并 | 先合并 |
| Plain 合并后的完整模型 | 支持 | 支持 | 支持 |
| Beacon adapter | 先用 35 合并 | 先合并 | 先合并 |

21 直接挂载 adapter 不改变磁盘权重，也不执行 merge_and_unload。47/56/57 仍要求完整本地模型；不要以为 21 的能力自动扩展到其他入口。原始模型基线是未经本项目 SFT 的 Qwen3.5-4B；实际语义取决于 `BASE_CHECKPOINT` 指向的权重，用户需要确认它不是之前训练/合并的模型。

比较时固定问题及顺序、训练提示来源、工具服务或语料和索引、模型 tokenizer、预算、解码及评测版本。21 默认贪心、无重复惩罚；可显式调整采样和惩罚，见 [23](../common/23_decoding.md)，不会因观察到复读自动调整。先对相同少量题目检查第一轮与工具调用，再评测全量；子集指标不能标成全量成绩。

## 22.2 环境与模型

```bash
cd /home/huangzj/proj/searchagent/sa-openseeker-v2
conda activate search-comp-qwen35-cu126
export PYTHON="$(command -v python)"
export RUN_DIR=/absolute/path/to/your/plain_run
# 绝对路径示例：需是与训练时完全相同的原始基座文件/版本。
export BASE_CHECKPOINT=/models/Qwen3.5-4B
```

脚本不会下载完整模型或验证它与训练基座逐字节相同。21 也接受可在本机加载的 HF ID `Qwen/Qwen3.5-4B`；47/56/57 的 native 模型解析要求本地完整目录，不能只填 HF ID。加载多模态原始 Qwen3.5 时，本项目提取文本主干，需足够 CPU 内存，另需 GPU 资源。仅 dry-run 不加载模型，也不会验证模型或接口可用。

`RUN_DIR` 可指训练实验根目录，21 从实验配置选择 split；47/56/57 从其中定位训练提示。迁移后训练数据路径可能不可用：21 显式指定 DATA；47/56/57 用 PROMPT_DATA 指向原训练 system/schema 的 JSONL，或不传 RUN_DIR 使用内置 Polar profile。`PROMPT_DATA` 只用于后者，不控制 21 的测试记录 system。

## 22.3 21 直接测试 LoRA adapter

训练产物通常如下；中间 checkpoint 是否存在取决于保存间隔和训练实际步数：

```text
RUN_DIR/models/model/
├── resolved_config.json      # 原训练的 model_name_or_path
├── final/                    # 默认 LoRA 最终 adapter 和 tokenizer
├── checkpoint-200/           # 可能存在的周期 adapter
├── final_merged/             # 自动合并的完整模型
└── train_summary.json
```

```bash
# final adapter：先检查命令，再实际评测。
RUN_DIR="$RUN_DIR" MODEL_PATH="$RUN_DIR/models/model/final" \
SPLIT=train LIMIT=8 BACKEND=replay DRY_RUN=1 bash scripts/sh/plain/21_plain_eval.sh

RUN_DIR="$RUN_DIR" MODEL_PATH="$RUN_DIR/models/model/final" \
SPLIT=train LIMIT=8 BACKEND=replay bash scripts/sh/plain/21_plain_eval.sh

# 中间 checkpoint：不需要先合并；若该目录没有 tokenizer 会回退到基座。
RUN_DIR="$RUN_DIR" MODEL_PATH="$RUN_DIR/models/model/checkpoint-200" \
BASE_MODEL_PATH="$BASE_CHECKPOINT" TOKENIZER_PATH="$RUN_DIR/models/model/final" \
SPLIT=train LIMIT=8 BACKEND=replay bash scripts/sh/plain/21_plain_eval.sh
```

选择 replay 是为了诊断历史动作，只有调用和历史记录完全匹配才返回观察；不是独立的真实搜索评测。自由查询时设置 `BACKEND=corpus CORPUS=/path/to/corpus.jsonl` 或 `BACKEND=http ENDPOINT=...`，并使用同一工具环境比较。OpenSeeker 只读取每条记录的 system/问题开始生成，不把该题历史助手回答或参考答案作为前缀。SPLIT=train 的得分只说明训练样本表现，还需用 validation/test 评估泛化。

### 22.3.1 adapter、基座与 tokenizer 的解析规则

1. `MODEL_PATH` 指向包含 `adapter_config.json` 和 `adapter_model.safetensors` 或 `adapter_model.bin` 的 Plain LoRA 目录时，识别为 adapter。
2. 基座优先级：显式 `BASE_MODEL_PATH` → adapter 同级的 `resolved_config.json` 的 `model_name_or_path` → `adapter_config.json` 的 `base_model_name_or_path`。三者都为空时报错；外部移动 adapter 后优先显式设置训练时**实际**使用的基座。
3. tokenizer 优先级：显式 `TOKENIZER_PATH` → adapter 目录中保存的 tokenizer → 基座 tokenizer。训练期间修改过 tokenizer 时必须指向训练保存的版本；仅基座路径相同不能证明 tokenizer 文件一致。
4. 按基座类型加载 Plain Qwen3 / Qwen3.5 文本模型，再加载 PEFT LoRA，保持 eval；不产生合并目录。缺少权重、缺少基座、Beacon adapter 或非 LoRA adapter 报错。
5. 预测 JSONL 的 `generation_config` 记录 `adapter_path`、`base_model_path`、`tokenizer_path`；指标 JSON 的 arguments 记录命令参数。`NUM_WORKERS>1` 会把这些参数传递给每个按题分片的 worker。

如果训练收尾中断导致 summary 的 `final_model_path` 仍指向 adapter，可显式用 `MODEL_PATH` 指定它；其他 benchmark 不接受该路径。若要测试合并版本，使用 29 输出新目录，显式用 MODEL_PATH 指向该完整目录。分布式训练产生的完整 adapter 权重和 tokenizer 必须已经成功保存；本入口不是 optimizer 恢复工具，也不会合并损坏的 checkpoint。

| 21 变量 | 默认 | 修改提示 |
|---|---|---|
| RUN_DIR | 空 | 实验根目录，自动选最终模型和 split；并用 MODEL_PATH 覆盖模型 |
| MODEL_PATH | 空 | 完整模型目录/HF ID，或 Plain adapter 目录；覆盖 RUN_DIR 模型 |
| BASE_MODEL_PATH | 空 | **仅适用于 adapter**；显式覆盖保存的原基座，模型必须实际匹配 |
| TOKENIZER_PATH | 空 | adapter/完整模型均可指定；优先于默认 tokenizer |
| DATA / SPLIT | 自动 / test | DATA 显式覆盖对应 split；train/validation/test；必须使用预处理 JSONL |
| BACKEND / CORPUS / ENDPOINT | replay / 空 / 空 | corpus 需有效 JSONL；http 需可访问网关 |
| LIMIT | 100 | 正整数，选前 N 条；21 不支持 -1 |
| MAX_CONTEXT / MAX_NEW_TOKENS | 16384 / 8192 | 总逻辑上下文、单轮生成上限；过长可能显存不足或耗尽预算 |
| MAX_TOOL_TOKENS / MAX_TURNS | 16384 / 32 | 单条完整工具返回和循环轮数限制 |
| CUDA_VISIBLE_DEVICES / NUM_WORKERS | 0 / 1 | 每 GPU 一个完整模型副本；显式卡列表至少与 worker 数一致 |
| OUTPUT_DIR / OUTPUT / LOG_PATH | 自动 | 输出与日志；避免覆盖以前结果，注意跨 run 的输出路径 |
| PYTHON / CONDA_ENV | 自动 / search-comp-qwen35-cu126 | PYTHON 优先；库缺失需在该环境安装依赖 |
| DRY_RUN | 0 | 1 只打印计划，不验证权重或实际加载 adapter |

## 22.4 原始 Qwen3.5-4B 一键基线入口

统一入口 `scripts/sh/baseline/70_qwen35_base_eval.sh` 调用独立的 71/72/73 脚本和原始 Qwen3.5 基座加载器，不调用 Plain 训练模型加载器或 Plain shell 流程；通用工具循环与评分器用于保持对照口径。完整的独立脚本、变量和修改提示见 [24 独立基线实验指南](24_qwen35_base_benchmark.md)。以 `BASE_CHECKPOINT` 为模型输入；即使同时传 RUN_DIR，也只用该 run 选择数据/训练提示，不会自动选择其训练后模型。`TARGET` 选择数据集，`ACTION` 选择阶段：

| TARGET | ACTION | 说明 |
|---|---|---|
| openseeker | run、score；另支持 download/prepare | 21 生成并评分；score 为已有预测汇总 |
| browsecomp | prepare、run、score | 在线生成时自动裁判；独立 score 需 PREDICTIONS |
| browsecomp-plus | download、prepare、index、run、score、all | 固定网页库；完整评分需本地 JUDGE_MODEL |
| hotpotqa | download、prepare、index、run、score、all | 固定 Wiki 摘要库；官方评分无需裁判模型 |

### OpenSeeker 基线

```bash
# 同一训练 run 的训练题、提示和 replay 诊断；改成 SPLIT=test 检查留出样本。
TARGET=openseeker ACTION=run BASE_CHECKPOINT="$BASE_CHECKPOINT" \
RUN_DIR="$RUN_DIR" SPLIT=train LIMIT=8 BACKEND=replay DRY_RUN=1 \
  bash scripts/sh/baseline/70_qwen35_base_eval.sh

TARGET=openseeker ACTION=run BASE_CHECKPOINT="$BASE_CHECKPOINT" \
RUN_DIR="$RUN_DIR" SPLIT=train LIMIT=8 BACKEND=replay \
  bash scripts/sh/baseline/70_qwen35_base_eval.sh
```

在与 adapter 或合并模型对比时，保留相同的 RUN_DIR、DATA/SPLIT、LIMIT、后端与预算；仅改变模型。需要自由生成查询时按 22.3 配置 corpus 或 HTTP。也可不提供 RUN_DIR，显式用 DATA 指定原始评测 JSONL；没有 DATA 时退回默认 prepared split。

### 在线 BrowseComp 基线

```bash
# 首次：下载固定版本上游与题目，并按 42 配置私有 API 文件。
TARGET=browsecomp ACTION=prepare bash scripts/sh/baseline/70_qwen35_base_eval.sh

TARGET=browsecomp ACTION=run BASE_CHECKPOINT="$BASE_CHECKPOINT" \
RUN_DIR="$RUN_DIR" LIMIT=2 DRY_RUN=1 ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/baseline/70_qwen35_base_eval.sh

# 真实运行会调用搜索、网页读取、摘要和裁判服务。
TARGET=browsecomp ACTION=run BASE_CHECKPOINT="$BASE_CHECKPOINT" \
RUN_DIR="$RUN_DIR" LIMIT=2 ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/baseline/70_qwen35_base_eval.sh
```

此入口固定 native，不要求待测模型服务；search/visit 使用官方 v1 工具。`ENV_FILE` 为私有配置文件；使用训练提示时 RUN_DIR 的原训练文件必须可访问，否则显式 `PROMPT_DATA=/path/to/train.jsonl`。仅评分已有生成用 `ACTION=score PREDICTIONS=/path/to/result_tool200.jsonl`，也需要裁判服务。在线预算和网络依赖详见 [42](../benchmark/42_openseeker_benchmark.md)。

### 离线 BrowseComp-Plus 基线

```bash
TARGET=browsecomp-plus ACTION=download bash scripts/sh/baseline/70_qwen35_base_eval.sh
TARGET=browsecomp-plus ACTION=prepare bash scripts/sh/baseline/70_qwen35_base_eval.sh
TARGET=browsecomp-plus ACTION=index bash scripts/sh/baseline/70_qwen35_base_eval.sh
TARGET=browsecomp-plus ACTION=run BASE_CHECKPOINT="$BASE_CHECKPOINT" \
RUN_DIR="$RUN_DIR" LIMIT=8 bash scripts/sh/baseline/70_qwen35_base_eval.sh

# 使用终端打印的生成目录，独立加载本地裁判。
GEN_DIR=/absolute/path/to/bcp_generation
TARGET=browsecomp-plus ACTION=score LIMIT=8 PREDICTIONS="$GEN_DIR/predictions.jsonl" \
JUDGE_MODEL=/models/Qwen3-32B bash scripts/sh/baseline/70_qwen35_base_eval.sh
```

默认 search-only；`RETRIEVAL_ONLY=1` 不需要裁判但不输出答案 accuracy。评分 ROOT_DIR、LIMIT 必须与生成一致，评分 OUTPUT_DIR 必须是新目录。已准备的 ROOT 可复用，prepare/index 不要重复覆盖。

### 离线 HotpotQA 基线

```bash
TARGET=hotpotqa ACTION=download bash scripts/sh/baseline/70_qwen35_base_eval.sh
TARGET=hotpotqa ACTION=prepare bash scripts/sh/baseline/70_qwen35_base_eval.sh
TARGET=hotpotqa ACTION=index bash scripts/sh/baseline/70_qwen35_base_eval.sh
TARGET=hotpotqa ACTION=run BASE_CHECKPOINT="$BASE_CHECKPOINT" \
RUN_DIR="$RUN_DIR" LIMIT=8 bash scripts/sh/baseline/70_qwen35_base_eval.sh

GEN_DIR=/absolute/path/to/hotpot_generation
TARGET=hotpotqa ACTION=score LIMIT=8 PREDICTIONS="$GEN_DIR/predictions.jsonl" \
  bash scripts/sh/baseline/70_qwen35_base_eval.sh
```

Hotpot 默认可 search+visit，使用官方答案 EM/F1、支持事实与 Joint 指标，无本地裁判模型。离线其他参数（ROOT_DIR、TOPK、ALLOW_VISIT、SNIPPET_TOKENS、READ_TOKENS、NUM_WORKERS 等）和准备来源见 [43](../benchmark/43_offline_benchmarks.md)。两个离线 benchmark 的 `LIMIT=-1` 表示全量；21 不接受 -1。

### 22.4.1 基线入口变量

| 变量 | 默认 | 含义 |
|---|---|---|
| TARGET / ACTION | openseeker / run | 选择上表流程；实际 download、run 等会执行对应操作 |
| BASE_CHECKPOINT | Qwen/Qwen3.5-4B | 原始基座；在线/离线 benchmark run 使用本地完整目录 |
| RUN_DIR | 空 | 从该训练 run 取得数据 split 或训练提示；不会把模型覆盖为训练后模型 |
| PROMPT_DATA | 空 | 47/56/57 原训练 JSONL；优先 RUN_DIR 提示，仅提取 system/schema |
| DATA / SPLIT | 各入口默认 | 21 DATA 为 prepared JSONL；benchmark DATA 含义由各入口定义 |
| LIMIT | 21 为 100，其他为 -1 | 小样本先指定正数；后续全量时明确分母 |
| BACKEND / CORPUS / ENDPOINT | replay / 空 / 空 | 仅 OpenSeeker 21；真实后端要事先配置 |
| ROOT_DIR / OUTPUT_DIR | 各 benchmark 默认 | 固定基准数据目录 / 新结果目录；不要覆盖生成结果 |
| PREDICTIONS / JUDGE_MODEL | 空 | score 时预测文件；BCP 完整评分需本地裁判 |
| ENV_FILE | 空 | 在线 BrowseComp 私有 API 文件，不写入仓库 |
| DRY_RUN | 0 | 下游支持时审查命令；不会加载模型、确认 API 或检验权重 |
| 其他变量 | 下游脚本默认 | 直接透传；21、47、56、57 参数和预算不完全相同 |

基线脚本使用 `BASE_CHECKPOINT` 和 `--model-origin base-qwen35`，不读取 MODEL_PATH 或仅用于 adapter 的 BASE_MODEL_PATH。需要比较已训练模型，请直接使用 21/47/56/57，而不是将训练后模型伪装为基线。

## 22.5 复读诊断和判读

先固定同一数据、提示、工具与预算，比较原基座、周期 adapter、最终 adapter 与合并模型。查看 predictions 的 `status`、`messages`、`events` 和 `generated_tokens`：第一轮就复读与工具执行后的复读应分开；replay_miss 只说明动作与历史不一致。短上限可减少诊断耗时，但需要所有对照相同，而且长推理会被截断。

adapter 与合并模型在相同 tokenizer、基座、环境、贪心解码下应高度接近；GPU 算子和浮点舍入可能带来细微误差。若文本不同，先核验 tokenizer、基座及 adapter 身份，必要时在相同硬件对短前缀比较 logits；不以字符串相等作为数值等价的唯一证明。

实际训练 loss、验证、优化步数与生成表现应一起检查；不要只增加 MAX_NEW_TOKENS 掩盖循环，也不要把抽样、惩罚视为训练质量的证明。一次只变更一个条件，记录模型路径/版本、提示来源、语料索引、预算、分母和失败状态。真实效果、GPU 显存及吞吐需要在目标服务器实测。
