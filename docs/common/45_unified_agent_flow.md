# 45 保留原始 system 的 Plain / Beacon 统一训练与测试

需要按顺序完成 Beacon 实验时，从 [32 完整实验指南](../beacon/32_beacon_experiment_guide.md) 开始；本文解释提示词与协议实现。

## 45.1 当前规则与适用范围

训练数据已有 system，不需要重新设计或替换。当前 `normalize()` 保留每条 OpenSeeker 轨迹原始 system；native 在线测试读取对应训练数据的 system 与工具 schema；离线测试在同一主体上仅调整实际能力、语料限制和答案格式。Plain、Beacon 使用相同构造器、ChatML 编码及 native Agent。

在线 Search/Visit、固定语料离线测试、replay、本地 corpus、HTTP 网关、官方 v1/v2 原样复现和 Search-R1 入口全部保留。本次不改变 Beacon 压缩公式、线性状态 writer、LoRA、损失定义、索引或官方评分函数，不启动真实训练。

当前训练协议为 `openseeker-chatml-v3-original-answer`。它是本项目的数据版本名，不是 OpenSeeker 官方发布的协议编号；“统一”指测试复用训练约定，不是把训练文本改成另一套提示。

## 45.2 参考来源与统一边界

参考 OpenSeeker 固定提交 `e3dec1f727d87c5eaa93506e4882c67237647098`：

- [官方源码](https://github.com/PolarSeeker/OpenSeeker/tree/e3dec1f727d87c5eaa93506e4882c67237647098) 的 `src/llm_tool_openseeker.py` 提供 v1 developer_prompt、Search/Visit 和完整历史工具循环。
- `src/config/chat_template.jinja` 定义消息、schema、工具响应和 assistant 生成前缀；公开 Polar 轨迹携带相应 system 和工具调用文本。
- 本地 Polar 原始首条记录与现有 v3 预处理首条记录的 system 完全相等，含 `<tools>` 的 schema、换行和尾部空行。默认参考副本保存为 `search_comp/data/openseeker_v1_system.txt`，不是重新编写的通用提示。
- 上游 v1/v2 的提示、工具和 boxed 要求并不完全相同。`official-v1` / `official-v2` 保持原样复现，不强行改成 native 的模板。
- 本项目 native 保留原 assistant 文本并共用自己的 `encode()`，生成前缀为 `<|im_start|>assistant\n`；官方 Jinja 会处理历史 reasoning，并在生成前缀加 `<think>`。共享原 system 不代表逐 token 复现整个官方推理引擎。

本地 upstream checkout 不是其完整训练器的审计证据；训练侧以公开轨迹和本项目训练输入为准。vendor 文件和官方评分源未修改。

## 45.3 训练数据：什么保留，什么仅在渲染时处理

| 内容 | 当前行为 |
|---|---|
| 原始 system.content | 原样保存，不替换、不清理空白、不添加新的角色指令 |
| 已内嵌的 `<tools>` schema | 原样保留，不重写为另一份描述 |
| 单独提供的 row.tools | 作为独立元数据保留；没有内嵌 schema 时，在编码输入中补充一次 |
| 问题、assistant 推理/调用/回答、工具观察 | 保留；结构化 OpenAI calls 和 tool role 按既有 wire protocol 转换 |
| 独立参考 answer | 仅训练数据监督/评分使用，不加入测试问题 |
| 原始文件与现有 v3 文件 | 不写回，不覆盖 |

`render_system(system, tools)` 只负责把独立 schema 渲染成模型可见的工具说明，不修改保存的 system。若 system 已有实际 JSON `<tools>` 块，不再次拼接，避免重复 schema。说明文字中的字面量 `<tools></tools>` 不会被误判为真正的 schema 块。

因此，对于当前 Polar 数据（schema 已内嵌、row.tools 为空），预处理、SFT 编码和 native 在线看到的是同一个 system 字符串。对于工具 schema 单独存储的结构化数据，SFT、预处理计长和 native 测试都用同一个 renderer；比较前缀时也必须传入同一份 tools。

缺失 system 的原始输入仍沿用既有默认补全规则；“保留原始 system”适用于原本携带 system 的记录，不凭空推断丢失文本。

## 45.4 native 如何选择对应训练提示词

唯一实现是 `search_comp/data/agent_prompt.py` 的 `resolve_system_prompt()`。优先级：

1. 显式 `PROMPT_DATA` / `--prompt-data` 指定的训练 JSONL。
2. `RUN_DIR/experiment.json` 中 `training_config.train_data_path` 指向的训练 JSONL。
3. 两者均未提供时，使用内置原始 Polar v1 提示副本。

不是从测试集、参考答案、qrels 或工具返回中提取 system。JSONL 可以是原始 trajectory、规范化 messages，或只含 system/messages 与独立 tools 的提示配置记录；messages/trajectory 和 tools 的 JSON 字符串形式同样支持。

读取过程逐行检查所有有效记录的**实际渲染 system**是否相等：

- 有不同 system：报 mixed training systems，不擅自选择第一条；用一个只包含目标提示配置的 PROMPT_DATA 指定实验所需 profile。
- 文件不存在、文件为空、缺少/空 system：报错；RUN_DIR 的训练文件不可访问时不会静默换成默认提示。
- 指定 PROMPT_DATA 时，它优先于 RUN_DIR 的训练路径，适用于训练目录迁移到另一台机器。
- 默认内置副本只代表已核对的 Polar v1 profile。单独测试其他来源的 MODEL_PATH 时，应显式指定其训练 PROMPT_DATA，不能据默认副本断言与任意 checkpoint 的训练提示一致。

这里只使用 system/schema；JSONL 中即使存在训练问题、参考答案或推理，也不会被复制到新测试问题。使用 API 的 native 在线工具仍是 upstream v1 Search/Visit；包含其他工具的训练 profile 不会使在线工具注册表自动增加该工具。

## 45.5 离线只做必要适配

离线也从上述规则读取原提示，并调用 `build_system_prompt(system=原提示, local=True, ...)`：

| 部分 | 处理方式 | 原因 |
|---|---|---|
| schema 前的角色说明、工具格式要求与尾部主体 | 保留原文 | 保持训练约定 |
| search.description | 改为固定本地语料搜索 | 不再承诺网络搜索或固定 top 10 |
| visit.description | 改为读取 local://docid 的存储文本 | 不再承诺网页联网或按 goal 进行 LLM 摘要 |
| 工具 name、参数结构 | 保留 | 不改变模型已经学习的参数接口 |
| `ALLOW_VISIT=0` | 从 schema 移除 visit，明确禁止调用 | 不宣称不存在的能力 |
| 本地环境限制 | 追加仅固定语料、无外网、local URL 的说明 | 防止把 source_url 当成可联网地址 |
| BCP 答案要求 | 追加 `[docid]` 引用 | 评分契约保留 |
| Hotpot 答案要求 | 追加 answer/sp JSON | 支持事实评分契约保留 |

离线后端只实现 search/visit。profile 没有实际 JSON schema、缺少 search，或包含本地不支持的工具时，明确报错；不凭空补另一套 schema。搜索返回仍为真实离线 JSON 文档，在线返回仍为 upstream 文本，不能为了看起来一致伪造来源。TOPK、snippet/read 预算由真实后端控制。

在线默认 `build_system_prompt(system=原提示)` 不做这些本地适配，返回原提示的相同字节内容。离线完整 system 的 hash 与在线不同是正常现象；应比较主体、schema 能力和具体后端，不能靠删除真实环境限制强求相同 hash。

## 45.6 数据格式与准备方式

| 输入 | 当前支持 | 操作 |
|---|---|---|
| 原始 Polar / 结构化轨迹 | 支持 | prepare 保留 system，正常规范化和计长 |
| 原有 v3 prepared 数据/manifest | 支持 | 原 system 和 tokenizer 未变时可直接训练、筛选和测试，无需仅因本次修改重建 |
| 未知 protocol | 拒绝 | 确认数据来源和版本，再显式准备 |

当前 `VERSION` 是 `openseeker-chatml-v3-original-answer`。保留原始 system；只有模型/tokenizer、消息渲染或筛选条件变化时才需要重新准备相应数据。

默认 10/20/30、21/31 回退路径、筛选/报告/HTML 入口指向 `outputs/data/openseeker/polar_correct_qwen35_16k`，显存探针指向 `polar_memory_probe`。目录名不决定数据版本；读取 manifest 中的 protocol 和 tokenizer 身份。

```bash
# 已有原始 v3 数据时，无需再运行 prepare。
# 需要重新准备时，选择尚不存在的输出目录；只加载 tokenizer，不训练。
INPUT_FILE=/data/original_polar.jsonl OUTPUT_DIR=/data/polar_original_system \
  MODEL_NAME=/models/Qwen3.5-4B bash scripts/sh/data/10_prepare.sh
```

从原始源文件重新准备时，输出应使用新目录。只保留旧 train 无法恢复原 validation/test；多文件时编辑 INPUTS，保留原问题划分规则。

## 45.7 一键入口与保留范围

| 入口 | 用途 | 当前提示行为 |
|---|---|---|
| 20 / 30，28 / 38 | Plain/Beacon 单卡/多卡训练 | 使用每条训练轨迹原始 system |
| 21 / 31 | replay / corpus / HTTP，EM/F1 | 直接复用所选数据记录的 system/schema |
| 47，默认 native | Plain/Beacon 在线工具测试 | 从 PROMPT_DATA/RUN_DIR/默认 Polar profile 读取原提示 |
| 47，official-v1/v2；48 | 原样官方评测与 Plain 服务 | 保留官方提示、模板与工具，不读取 native PROMPT_DATA |
| 56 / 57 | BCP/Hotpot 固定语料评测 | 原提示主体 + 必要本地能力/格式适配 |
| 59 | CPU 协议/小模型一致性检查 | 无真实训练、模型下载或在线 API 调用 |

21 是设置 KIND=plain 后调用 31 的薄包装；参数在 31 顶部或用同名环境变量覆盖。Plain/Beacon 不按模型类型选择另一套 prompt。Search-R1 保留独立 `<search>/<information>` 协议。

```bash
# 只核对代码，不运行真实模型。
bash scripts/sh/tools/59_check_agent_alignment.sh

# 在线：由 RUN_DIR 自动选择实际训练提示。
RUN_DIR=/path/to/run ENV_FILE=/private/api.env LIMIT=8 \
  bash scripts/sh/benchmark/47_openseeker_benchmark.sh

# 模型和训练文件搬迁后：显式给对应训练数据。
MODEL_PATH=/models/merged KIND=beacon PROMPT_DATA=/data/original_train.jsonl \
  ENV_FILE=/private/api.env bash scripts/sh/benchmark/47_openseeker_benchmark.sh

# 离线：先按 43 完成数据与索引准备；PROMPT_DATA 对两种模型使用同一来源。
ACTION=run RUN_DIR=/path/to/run PROMPT_DATA=/data/original_train.jsonl LIMIT=8 \
  bash scripts/sh/benchmark/56_browsecomp_plus.sh
ACTION=run RUN_DIR=/path/to/run PROMPT_DATA=/data/original_train.jsonl LIMIT=8 \
  bash scripts/sh/benchmark/57_hotpotqa_offline.sh

# 原样官方模式继续保留，不设置 PROMPT_DATA。
MODE=official-v1 ENV_FILE=/private/api.env bash scripts/sh/benchmark/47_openseeker_benchmark.sh
MODE=official-v2 ENV_FILE=/private/api.env bash scripts/sh/benchmark/47_openseeker_benchmark.sh

# 只检查参数计划，不加载模型、不调用 API。
DRY_RUN=1 PROMPT_DATA=/data/original_train.jsonl bash scripts/sh/benchmark/47_openseeker_benchmark.sh
```

Dry-run 只展示参数/命令，不宣称训练提示文件、模型或 API 已完成真实联调。native 在线和离线生成前会读取并校验 prompt profile。离线多卡 worker 同样接收 PROMPT_DATA 和 RUN_DIR，不会在子进程意外退回默认 profile。

## 45.8 参数与修改提示

| 参数/变量 | 默认或位置 | 说明与修改提示 |
|---|---|---|
| `PROMPT_DATA` / `--prompt-data` | 47/56/57，空 | 原训练 JSONL，优先级最高；单 profile；不会读取其问题作为评测输入 |
| `RUN_DIR` / `--run-dir` | 空 | 从 experiment.json 解析模型与训练提示来源；搬迁后路径失效请指定 PROMPT_DATA |
| `MODEL_PATH` / `--model` | 空 | 47/56/57 使用本地完整 checkpoint，优先于模型自动定位；21 另支持 Plain adapter，见 22 |
| `KIND` | 自动 | plain/beacon，仅选择模型，不改变 system |
| `MODE` | 47=native | official-v1/v2 使用原样 upstream；显式给它们 --prompt-data 会报错 |
| `ENV_FILE` | 47，空 | 私有在线 API 配置；不影响离线工具 |
| `ALLOW_VISIT` | 56=0、57=1 | 同时改变实际能力及 schema，比较两个模型时保持一致 |
| `TOPK` | 56/57=5 | 每个查询返回数，不再由 system 承诺固定 10 |
| `SNIPPET_TOKENS` / `READ_TOKENS` | 512/4096 | 每篇搜索/读取正文上限，按待测 tokenizer 计数 |
| `MAX_TOOL_TOKENS` | 47/56/57=32768 | 单条完整观察上限，含 JSON/包装；超出报预算失败 |
| `MAX_CONTEXT` | 47/56/57=131072 | 完整未压缩历史逻辑上限，不是物理 KV 长度 |
| `MAX_TOKENS` | 47=32768 | native 每轮生成上限；官方模式交给原脚本 |
| `MAX_NEW_TOKENS` | 56/57=8192 | 离线每轮生成上限，不是整题总数 |
| `MAX_TURNS` | 47/56/57=200 | native 循环上限；不能解决单轮重复生成 |
| `TOOL_COUNT` | 200 | 离线真实调用上限；47 沿用官方模式工具次数参数，native 仍由轮数控制 |
| `LIMIT` | -1 | 全量；正数为前 N 题，不能称全量成绩 |
| `NUM_WORKERS` | 56/57=1 | 每 GPU 完整模型副本，按题并行；训练提示来源随子进程传播 |
| `WORKERS` / `JUDGE_WORKERS` | 47=60/20 | 官方生成/裁判并发；native 生成仍逐题 |
| `DATA` | 测试脚本默认问题集 | 评测问题来源，与 PROMPT_DATA 的训练提示来源区分 |
| `CORPUS` / `ROOT_DIR` | 56/57 | 独立文档/索引；保留原来源和指纹检查 |
| `OUTPUT_DIR` | 新结果目录 | 禁止覆盖旧结果，防止缓存混用 |
| `PREDICTIONS` | 空 | 导入已有预测评分；提示来源以原预测记录为准，不重新生成 |
| `DRY_RUN` | 0 | 1 展示计划；不验证实际模型效果 |
| `PYTHON` / `CONDA_ENV` | 项目公共解析规则 | 不自动安装依赖；使用已装 torch/transformers/peft 的环境 |
| `OMP_NUM_THREADS` | 59=1 | CPU tiny 测试线程数；末尾额外参数直接传 pytest |

10_prepare.sh 仍支持同名环境变量：SOURCE、INPUT_FILE、OUTPUT_DIR、MODEL_NAME、MIN_LENGTH、MAX_LENGTH、CORRECTNESS、OBSERVATIONS、WORKERS、OFFLINE、VAL_FRACTION、TEST_FRACTION、SEED。INPUTS/MODELS 多值编辑脚本数组；--input 在末尾是追加而非替换。默认 MAX_LENGTH=16384、Correct、观察不限、4 进程、OFFLINE=1、划分约 96%/2%/2%、seed=42。OUTPUT_DIR 必须不存在，已有 v3 数据无需覆盖重建。详见 10/11。

改训练 system，应修改自己的源数据并建立新实验，而不是修改 normalizer 统一覆盖；新增数据必须重新计长。改工具能力只改后端及适配函数，保留训练 profile；自定义工具需同时提供真实实现。改默认参考副本必须说明数据来源、核对字节/hash，并运行 59；自定义 checkpoint 优先显式 PROMPT_DATA。

## 45.9 模型算法和结果追踪

训练仍是整条真实轨迹的 teacher forcing。Plain/Beacon 使用相同 input_ids 和 labels，Beacon 额外按 observation 正文 regions 压缩。SFT 的工具 schema 渲染与 native 初始 system 渲染共用 render_system；每个生成前缀与训练 token 前缀对齐。

native 默认使用贪心生成；Plain 可配置采样和重复惩罚，Beacon 可在贪心时对压缩后可见普通 token 设置重复惩罚，具体参数及对照边界见 [23](23_decoding.md)。统一停止文本与多 EOS 处理。每轮完整历史重新 prefill（reuse_cache=false），轮内缓存解码。Qwen3.5 tiny 的 forward/prefill/decode 数值一致性测试保留；这些不等于所有真实 GPU kernel/checkpoint 均已数值验证。

Beacon whole SFT 布局仍在归档和加载前统一解析；随机压缩训练时采样、推理固定的既有选项保持不变。参数与算法细节见 30/31/60。本次不因保留 system 改变压缩算法或训练损失。

native 预测保存完整 messages、prompt_source、prompt_version（当前构造策略版本）、system_prompt_sha256 和 generation_config。prompt_source 区分显式训练文件、RUN_DIR 自动定位与内置参考文件；完整 system hash 才代表具体文本身份，不能仅凭策略版本号判断不同 profile 相同。generation_config 记录模型、配置指纹及实际 Beacon 设置。

已有 v3 模型可直接使用其原训练提示评测；不要求为本次修复重训。已有实验仍保留代码/数据指纹校验；代码变化后要新建训练计划，不改写旧实验指纹绕过校验。重复推理/非法调用是否改善，仍需真实模型评测；本次仅修改代码、文档和回归测试。
