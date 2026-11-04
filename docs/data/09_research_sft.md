# 09 RESEARCH OpenSeeker SFT：本地导出、审计、训练与测试

> 从下载到评分的完整实验顺序见 [71 Research 完整实验指南](71_research_experiment_guide.md)。本文专讲角色适配、数据边界和接口参数。

> 当前操作入口以 [68 两数据集、三种方法与独立测试/评分](../project/68_training_benchmark_guide.md) 为准。训练仅使用 OpenSeeker / Research；新入口每种方法均为 10/11 训练、40/42/44 测试、41/43/45 评分。下文旧编号用于兼容或历史实验。

适用数据为 `AmanPriyanshu/tool-reasoning-sft-RESEARCH-OpenSeeker-v1-Data` 的本地 JSONL 导出，例如 `outputs/data/openseeker_one.jsonl`。本文件的 prepare/html 阶段不调用 Hub、不下载数据或模型；独立下载使用 data/08_download_research.sh，见 71。所有预处理、训练、推理命令在 Linux 项目根目录运行。Windows 本次只编辑与检查代码；没有运行真实预处理、训练或模型评分。

## 09.1 最短使用路径

先在 Linux 激活已有项目环境，建议显式指定 Python 和本地完整模型目录。所有 shell 文件均以 `bash` 调用，不依赖 executable 位。

```bash
# 只浏览原始数据的前 20 条；不分词、不加载模型。
ACTION=html INPUT=outputs/data/openseeker_one.jsonl LIMIT=20 \
  PYTHON=python3 bash scripts/sh/tools/09_research_workflow.sh

# 仅预处理。MODEL_PATH 需包含本地 tokenizer，路径字符串要与训练一致。
ACTION=prepare MODEL_PATH=/data/models/Qwen3.5-4B \
  INPUT=/data/exports/research.jsonl DATASET_DIR=outputs/data/research/prepared_16k \
  bash scripts/sh/tools/09_research_workflow.sh

# 审阅配置，不训练。数据目录必须已有完整的 prepared 产物。
ACTION=plan METHOD=plain MODEL_PATH=/data/models/Qwen3.5-4B \
  bash scripts/sh/tools/09_research_workflow.sh
ACTION=plan METHOD=beacon MODEL_PATH=/data/models/Qwen3.5-4B \
  bash scripts/sh/tools/09_research_workflow.sh

# 之后需要真实训练时自行执行；本次没有执行。
ACTION=train METHOD=beacon MODEL_PATH=/data/models/Qwen3.5-4B \
  CUDA_VISIBLE_DEVICES=0 bash scripts/sh/tools/09_research_workflow.sh
```

四个 ACTION 分阶段执行。训练不会隐式重新预处理，预处理不会隐式启动训练。默认 ACTION=html，避免误启动训练。预处理输出目录必须不存在；已有结果可直接用于 plan/train。仅一条示例可检查格式，无法代表全量分布或训练质量；分组划分后也可能没有验证集，极小输入甚至可能没有训练样本，训练入口会拒绝空训练集。

## 09.2 格式映射与训练/推理一致性

新增 `search_comp/data/research.py::normalize_research`，通过 `10_prepare.py --source research` 显式选择，其他来源的适配规则保持原入口。支持 messages 为数组或 JSON 字符串。

| 原始角色 | prepared 角色/内容 | 目的 |
|---|---|---|
| system | system | 去除导出的旧 Tool-Use Protocol 前缀，保留实际研究说明和内嵌工具 schema，追加当前调用协议及 answer 约束 |
| user | user | 原始问题；问题归一化哈希用于去重和划分 |
| reasoning | assistant 中的 think 内容 | 与紧随的 action 合并，避免学习在工具调用前结束回合 |
| tool_call | 同一 assistant 回合，增加 tool_calls_begin/end 包络 | 使用当前工具解析器所需格式，保留调用 JSON |
| tool_output | user 中的 tool_response 块 | 通过共享 normalize 包装；正文为 observation |
| answer | assistant 中保留 answer 标签 | 与其前置 reasoning 合并；兼容现有结束标记、答案提取和分段 loss |

search 要求非空字符串数组 `query`；visit 要求 `url` 字符串或数组、非空 `goal`。其他工具会记录为 invalid，不能未经适配就进入训练。调用必须有 observation，最后必须是 answer；不完整 reasoning、损坏 JSON、未知角色会拒绝。observation 内部的协议同名标签转义，避免页面文字改变压缩边界。原始文件不修改，prepared 记录带 source、source_index、adapter_version，可回到原文件对应行检查。

prepared 继续使用 `data_mode: openseeker` 和现有协议版本，新增的是来源适配器，不增加训练器。Plain/Beacon 都复用 OpenSeekerDataset、encode 和 collator。系统/问题/observation labels 为 -100，只有 assistant 文本参与监督；工具结果仅正文成为 Beacon 可压缩区域，外层标签与 ChatML 边界不压缩。分段 loss 的 think/tool_call/answer 权重继续可配置。

没有独立 answer 时，用演示最终回答的标签正文满足现有训练结构，标记 `answer_source=demonstration_final_response`、`training_only=true`。已有 answer 则保留并标记 reference。无论哪一种，都不把本来源的 held-out 演示当成独立 benchmark 金标准；不根据演示自己的答案宣称其正确。此来源默认 correctness=any，显式要求 Correct/Incorrect 会报错。

评测时 `RUN_DIR` 自动定位 prepared 训练提示，或 `PROMPT_DATA=.../train.jsonl` 显式指定。当前 search/visit 工具参数与该导出一致，无需另写推理执行器。HotpotQA/BrowseComp-Plus 仍按已有能力适配把远端搜索改为本地固定语料，属于工具环境差异，结果应说明 offline。Hotpot 的答案/支持事实输出要求继续由 benchmark 入口追加。多个不同 system 的数据不能随意选第一条进行评测；现有提示解析器会报 mixed training systems，需明确指定单一匹配提示配置。

## 09.3 长度、筛选与审计

使用当前共享 encode 对转换后的完整训练序列分词，包含系统、schema、ChatML、reasoning、调用包络、observations 和最终回答。不是字符数，也不是原始 messages 的序列长度。先测量、记录，再按完整长度筛选；不截断 observation 或最终答案，不用 Beacon 压缩后的长度冒充训练输入长度。

默认 16384；改为 32768 时，先设置新的 DATASET_DIR 和 MAX_LENGTH 重新 prepare，再使用相同 MODEL_PATH/MAX_LENGTH plan/train。不同 tokenizer 必须分别测量。直接 Python CLI 支持 `--models` 多值，默认要求所有测量模型均满足长度；`--tokenizer` 可指定筛选所用的测量键。训练仍要求 manifest 包含训练模型 tokenizer 的同名键。

| 产物 | 内容 |
|---|---|
| normalized.jsonl | 所有规范化成功样本，包含之后因超长、重复或无 observation 而被筛掉的记录及 lengths |
| train/validation/test.jsonl | 通过筛选、按 question_id 去重与分组划分的数据 |
| rejected.jsonl | 无效样本 source/index/reason；有效但拒绝的样本 id/reason/lengths |
| measurements.jsonl | 每个有效样本的长度、工具调用数与拒绝原因；invalid 无可用 token 测量 |
| manifest.json | 参数、adapter 版本、dataset ID、输入路径/字节数/SHA256、tokenizer 指纹、计数、角色和工具统计 |

manifest 的 distributions 保持原口径：每个 tokenizer 的 total/observations/assistant，以及 calls/messages 的 n/min/max/mean/p50/p90/p95/p99 和超 8k/16k/32k/64k 计数。分布覆盖规范化成功的筛选前总体；无效样本独立计数。normalized 和 measurements 可继续使用既有 refilter、报告与绘图工具；原始 Research 结构审计使用 18 的 --source research。45_report 显式选 tokenizer，并区分筛选前与实际各 split；44_plot 的 POPULATION 可选 all/retained/rejected，详见 71.6。`OBSERVATIONS=with` 只保留有非空 observation 正文的轨迹，`without` 选择无正文轨迹，`any` 不按 observation 筛选。原始 tool_output 的 HTML 展示保留原文包装，字符统计不是 token 统计。

## 09.4 脚本变量和修改位置

| 变量 | 默认值 | 适用阶段与修改提示 |
|---|---|---|
| ACTION | html | html/prepare/plan/train；分别执行，不自动串联 |
| METHOD | plain | 兼容 workflow 的 plan/train 可选 plain/beacon/baseline；日常直接使用方法目录下独立 11 |
| INPUT | outputs/data/openseeker_one.jsonl | html/prepare 的本地输入，真实数据请显式覆盖 |
| DATASET_DIR | outputs/data/research/prepared_16k | prepare 输出；plan/train 输入 |
| SNAPSHOT_DIR | 空 | prepare 非空时优先读取 download_manifest.json 中全部 jsonl_inputs；覆盖 INPUT |
| MODEL_PATH | Qwen/Qwen3.5-4B | prepare 的 tokenizer 与训练模型；默认只查本地缓存，推荐服务器绝对路径 |
| MAX_LENGTH | 16384 | prepare 和训练的完整 token 上限，正整数 |
| WORKERS | 4 | prepare CPU 进程数；多进程会各自加载 tokenizer |
| SEED | 42 | prepare 问题分组划分种子；训练种子在原训练脚本 YAML 中修改 |
| VAL_FRACTION / TEST_FRACTION | 各 0.02 | prepare 留出比例；非负且和小于 1；均 0 表示全部训练 |
| OBSERVATIONS | with | prepare：with/without/any |
| LIMIT / OFFSET | 20 / 0 | html 的记录数和跳过条数；LIMIT=0 表示全部，注意 HTML 大小 |
| OUTPUT_HTML | outputs/reports/research_sample.html | html 输出文件；可离线打开 |
| PYTHON / CONDA_ENV | 现有环境选择规则 | prepare/训练调用 common.resolve_python；html 使用 PYTHON 或 python3 |
| CUDA_VISIBLE_DEVICES | 训练脚本默认 0 | plan/train 复用原训练入口的 GPU 设置 |
| DISTRIBUTED_BACKEND | zero2 | 多卡复用原入口；详见既有 Plain/Beacon 文档 |
| DATASET_NAME | workflow 固定 research | 经共享 05 写入实验配置，区分数据来源和 data_mode |

直接运行 `data/09_prepare_research.sh` 只做 prepare；workflow 不增加另一份训练超参数。Plain 调整 `plain/11_train_research.sh` 内联 YAML，Beacon 调整 `beacon/11_train_research.sh` 内联 YAML；训练变量、LoRA、梯度累积、显存、损失和分布式说明见 [20](../plain/20_plain.md)、[30](../beacon/30_beacon.md)、[53](../tools/53_self_contained_training.md)。common/05 仅应用明确设置的 DATASET_DIR、MODEL_PATH、MAX_LENGTH、DATASET_NAME；原入口默认行为继续沿用。

HTML 可查看原始角色或 prepared 后的 assistant/user。原始 JSON 数据仍保留，新增 reasoning/tool_call/tool_output/answer 角色筛选和正文着色。分批查看另一段用 OFFSET=100 LIMIT=20；prepared 文件直接使用 `report/40_dataset_html.sh`，详见 [41](../report/41_html.md)。

## 09.5 三种方法、三个 Benchmark 与独立评分

Research 训练入口分别为 `plain/11_train_research.sh`、`beacon/11_train_research.sh`、`baseline/11_train_research.sh`。Baseline 使用标准全参数 SFT，Plain 默认 LoRA，Beacon 使用压缩训练。旧 baseline/72/74/75 是未训练原始基座，不用于测试新的 baseline SFT。

每个方法目录的 `40_test_browsecomp.sh`、`42_test_browsecomp_plus.sh`、`44_test_hotpotqa.sh` 只生成；对应 `41_score_browsecomp.sh`、`43_score_browsecomp_plus.sh`、`45_score_hotpotqa.sh` 只评分。脚本、全部变量、离线资产准备和修改提示见 [68](../project/68_training_benchmark_guide.md)，训练字段逐项见 [69](../project/69_training_parameters.md)。

```bash
RUN_DIR=/experiments/research_plain LIMIT=8 DRY_RUN=1 \
  bash scripts/sh/plain/44_test_hotpotqa.sh

# 数据和索引准备完毕后正式生成；评分在另一个进程进行。
RUN_DIR=/experiments/research_plain LIMIT=8 OUTPUT_DIR=outputs/results/research_hotpot_run \
  bash scripts/sh/plain/44_test_hotpotqa.sh
PREDICTIONS=outputs/results/research_hotpot_run/predictions.jsonl LIMIT=8 \
  JUDGE_MODEL=/models/judge OUTPUT_DIR=outputs/results/research_hotpot_score \
  bash scripts/sh/plain/45_score_hotpotqa.sh
```

三项 Benchmark 全部输出 EM、F1 和 LLM judge；Hotpot 的 EM/F1 为原官方实现，BrowseComp 两项为补充答案指标。Research 测试的 RUN_DIR 必须属于本次 Research 训练；脱离实验指定模型时，必须同时指定规范化的 Research train.jsonl 作为 PROMPT_DATA。测试只读取 system/schema，不读取训练问题或答案；遇到混合提示配置会明确拒绝。在线与离线的必要工具/格式调整见 68.3。

## 09.6 验证与后续扩展

`tests/test_research_sft.py` 覆盖输入不变、角色合并、调用解析、observation 的监督 mask/region、HTML 角色、非法轨迹拒绝和 reference 追踪。本次已有 Python 3.12 环境执行这 7 项检查通过，没有真实分词、预处理、GPU 前向或训练。

```bash
python -m unittest discover -s tests -p test_research_sft.py -v
python -m unittest discover -s tests -p test_hotpot_judge.py -v
python -m unittest discover -s tests -p test_training_benchmark_matrix.py -v
```

未来扩展原始角色或工具时修改 research 适配器并升级 ADAPTER_VERSION，不放宽通用 normalize 的轨迹校验。改变计长或 observation 边界时补充 encode/region 检查，之后在 Linux 用真实 tokenizer 重新预处理。验证边界见 [99](../project/99_validation.md)。
