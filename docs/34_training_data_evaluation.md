# 34 用训练数据做生成式评测

`scripts/tools/sh/84_test_training_data.sh` 用于检查一个已经训练过的 Plain 或 Beacon 模型，能否从指定数据划分的问题出发重新检索并作答。默认测试该实验实际使用的 `train.jsonl`，结果用于训练集诊断。也可以读取同一 prepared 版本的 validation/test，或显式指定另一份 prepared 数据。BrowseComp 系列和 HotpotQA 的官方题集与指标使用各自的测试入口。

## Linux 上运行

1. 确认目标训练实验有 `experiment.json`、`job.json` 和可加载的最终模型或周期检查点。
2. 准备一份本地语料索引。可用 [数据准备](20_data.md) 中的 31 或 32 脚本完成 `download → prepare → index`。本脚本从所选版本读取 `index.sqlite` 和 `index.manifest.json`。该版本的 benchmark 题目不参与这次评测。
3. 编辑 `84_test_training_data.sh`。首次填写 `SA_METHOD`、`SA_TRAINING_RUN`、`SA_INDEX_DIR`，保留默认的 `SA_SPLIT=train` 和 `SA_GENERATION__LIMIT=10`。
4. 在项目根目录运行：

```bash
bash scripts/tools/sh/84_test_training_data.sh
```

例如，检查 Beacon 的周期检查点，可在同一个 SH 中填写：

```bash
export SA_METHOD=beacon
export SA_TRAINING_RUN=artifacts/experiments/beacon/browsecomp-sft/REPLACE_WITH_RUN
export SA_CHECKPOINT=checkpoint-200
export SA_DATASET_DIR=''
export SA_SPLIT=train
export SA_INDEX_DIR=artifacts/benchmarks/browsecomp-plus/v1
export SA_RUN_NAME=ckpt200_train_smoke
export SA_GENERATION__LIMIT=10
```

如果想看该实验的保留轨迹，改 `SA_SPLIT=test`。如果想让同一模型测试另一个 prepared 版本，填写 `SA_DATASET_DIR` 并选择相应划分。比较不同检查点时，沿用相同数据版本、语料版本和生成参数。

## 参数表

### 模型、题目和语料

| SH 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `PYTHON_BIN` | 当前环境的 `python` | 激活 Linux 环境，或在启动前设置解释器绝对路径。 |
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face 地址，可改成允许访问的镜像。 |
| `HF_HUB_OFFLINE` | `1` | 优先按默认值使用本地模型文件和缓存。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 单进程模型推理使用的可见设备，建议指定一个 GPU。 |
| `SA_METHOD` | `plain` | 选 `plain` 或 `beacon`，与训练实验记录一致。 |
| `SA_TRAINING_RUN` | 实验目录占位路径 | 必填，填写训练实验根目录。Plain、Beacon 的 OpenSeeker、Research、BrowseComp SFT 和 mixed 实验都可以使用。 |
| `SA_CHECKPOINT` | 空 | 留空加载成功完成训练的最终模型。也可填 `checkpoint-200`，或当前实验下该周期检查点的完整路径。 |
| `SA_DATASET_DIR` | 空 | 留空从训练实验记录定位划分，并验证数据 SHA256。填写时读取给定 prepared 目录下的 `<SPLIT>.jsonl`。可用项目相对路径或 Linux 绝对路径。 |
| `SA_SPLIT` | `train` | 可选 `train`、`validation`、`test`。所选文件至少有一条当前协议的轨迹。 |
| `SA_INDEX_DIR` | `artifacts/benchmarks/browsecomp-plus/v1` | SQLite 语料索引所在目录，须同时包含 `index.sqlite` 和 `index.manifest.json`。索引可与其他测试共享。证据覆盖取决于实际语料。 |
| `SA_RUN_NAME` | `smoke` | 任务标签。用字母、数字、下划线或连字符，系统追加时间和随机后缀。 |

### 推理与工具预算

| SH 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `SA_GENERATION__LIMIT` | `10` | 正数取当前文件顺序的前 N 条；`-1` 使用全部。全量推理前建议确认少量样本能加载和检索。 |
| `SA_RESUME_FROM` | 空 | 填同一实验下已完成的训练数据评测根目录，复用原预测，只生成新增题。LIMIT 填累计题数。见 [测试续测](36_browsecomp_plus_resume.md)。 |
| `SA_GENERATION__MAX_CONTEXT` | `32768` | 每题完整上下文预算，包括提示词、历史生成和实时工具返回。按模型支持长度和设备内存调整。 |
| `SA_GENERATION__MAX_NEW_TOKENS` | `4096` | 每轮模型生成的最大 token 数，还受剩余上下文约束。 |
| `SA_GENERATION__MAX_TURNS` | `20` | 每题最大生成轮数。 |
| `SA_GENERATION__MAX_TOOL_TOKENS` | `16384` | 完整单条工具返回的 token 预算。超出时结束该题并记录 `tool_budget`。 |
| `SA_GENERATION__TOOL_COUNT` | `20` | 每题最多执行的工具调用对象数。批量 search 仍按一个调用对象计。 |
| `SA_GENERATION__TOOL_CALL_RETRIES` | `1` | 非法调用格式的修正次数，可设 0。 |
| `SA_GENERATION__DECODING` | `greedy` | Plain 支持 `greedy` 或 `sample`，Beacon 使用 `greedy`。 |
| `SA_GENERATION__REPETITION_PENALTY` | `1.0` | 正有限值，1.0 表示无额外重复惩罚。 |
| `SA_GENERATION__TEMPERATURE` | `1.0` | `sample` 模式的温度，使用正有限值。 |
| `SA_GENERATION__TOP_P` | `1.0` | `sample` 模式的 nucleus 阈值，范围 `(0,1]`。 |
| `SA_GENERATION__GENERATION_SEED` | `42` | 解码种子，范围 0 到 2147483647。题目选择始终遵循文件顺序。 |
| `SA_GENERATION__VERIFY_INDEX` | `true` | 校验 SQLite 文件 SHA256。设为 `false` 会跳过全文件哈希，仍核对索引内部与旁路清单。 |
| `SA_GENERATION__TOPK` | `5` | 每条 BM25 查询最多返回的文档数。 |
| `SA_GENERATION__SNIPPET_TOKENS` | `512` | 每篇 search 结果的正文截取上限。 |
| `SA_GENERATION__READ_TOKENS` | `4096` | 每篇 visit 文档的正文截取上限。 |
| `SA_GENERATION__ALLOW_VISIT` | `true` | 样本工具定义有 visit 时启用本地文档读取。设为 `false` 时统一采用 search-only。 |

## 输入边界和结果解读

每题的模型输入由本行的系统提示词、工具定义和首条 user 问题组成。mixed 数据按行使用各自的提示词。工具定义按本地 BM25 环境改写，历史推理、历史工具返回、顶层答案和正确性标签留在输入之外。模型发出的 search 和 visit 在指定索引上重新执行，visit 接受 `local://<docid>`。原始轨迹使用开放网页时，实际命中文档可能与历史观测不同。

结果写在 `<训练实验>/evaluation/training-data/<任务标签_时间_后缀>/`：

| 文件 | 内容 |
|---|---|
| `job.json` / `console.log` | 配置、运行状态和进度。失败时可查看日志及已写出的逐题预测。 |
| `predictions.jsonl` | 每题的 ID、预测、比较答案、生成轨迹、实时检索记录、EM/F1、结束状态和耗时。 |
| `summary.metrics.json` | 结束状态分布、完成率，以及按 `answer_source` 分组的平均 EM/F1。未生成有效答案的记录按 0 分计入其组。 |
| `evaluation.json` | 实际模型、划分、数据文件 SHA256、索引元数据、生成参数和预测文件 SHA256。`matches_training_train_sha256=true` 表示所用输入文件与实验训练文件的指纹一致。 |

EM/F1 适合短答案的字面匹配。使用 `reference` 的行对比顶层参考答案。使用 `demonstration_final_response` 的行对比历史示范回复，长解释可能拉低字面分数。两类评分分组展示。默认训练划分用于观察样本内表现，`validation` 和 `test` 的结果也以所选 prepared 版本的划分为准。与官方评测结果比较时，应分别报告。
