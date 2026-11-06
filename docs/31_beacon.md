# 31 Beacon 完整实验指南

本文字段名称表示逻辑参数：SH 中普通变量加 `SA_` 前缀，例如 `MODEL_PATH` 对应 `SA_MODEL_PATH`；嵌套键用双下划线，例如 `TRAINING.max_length` 对应 `SA_TRAINING__MAX_LENGTH`。环境变量如 `HF_ENDPOINT` 不加前缀。实际可编辑名称、类型与默认值见 [61 脚本参数清单](61_script_inventory.md)。

## 1. 实验顺序

安装环境 → 准备本地模型 → 准备一种训练数据 → 准备三个 benchmark → 校验训练计划 → 训练 → 三项测试 → 三项独立评分 → 检查产物。

公共前置步骤按 [10 环境准备](10_environment.md) 和 [20 数据准备](20_data.md) 执行。每个 SH 文件顶部包含完整待改参数；执行时只传脚本文件名。

## 2. 选择数据并修改训练脚本

| 训练数据 | 入口 | `SA_DATASET_DIR` 指向 |
|---|---|---|
| OpenSeeker | `scripts/beacon/sh/10_train_openseeker.sh` | `artifacts/datasets/openseeker/prepared/<PROFILE>` |
| Research | `scripts/beacon/sh/11_train_research.sh` | `artifacts/datasets/research/prepared/<PROFILE>` |

在选定文件中修改：

1. `SA_MODEL_PATH`：本地基座路径，与该数据 manifest 的 tokenizer 身份完全相同。
2. `SA_DATASET_DIR`：对应 prepared 版本，不能交叉使用数据身份。
3. `SA_MODEL_FAMILY`：默认 qwen35；使用 Qwen3 时改 qwen3 并重新按其 tokenizer 预处理。
4. `SA_RUN_NAME`：如 `qwen35_L16384_s42`；创建目录时自动加 UTC 时间和随机后缀。
5. `TRAINING`：学习率、长度、梯度累积、训练轮数/步数、随机种子、保存与评估间隔。
6. `ENVIRONMENT`：GPU 编号；多卡同步 `NPROC_PER_NODE`、`DISTRIBUTED_BACKEND`，ZeRO 配置位于本文件 `DISTRIBUTED`。

## 3. Beacon 专用参数

默认 `beacon_layout_policy='configured'`，`beacon_information_mode='whole'`，`beacon_window_enabled=false`，按完整工具观测区段压缩；默认固定比例 `beacon_ratio=16`。只压缩工具观测体，保留原始 system、问题和 assistant 监督协议。

| 目标 | 修改项 | 联动要求 |
|---|---|---|
| 固定压缩比例 | `SA_BEACON__BEACON_RATIO` | 训练与测试读取同一 checkpoint 配置 |
| 随机压缩训练 | `beacon_compression_mode='random'`、`beacon_compression_ratios` | 推理仍使用固定 `beacon_ratio` |
| 窗口布局 | `beacon_information_mode='window'`、`beacon_window_enabled=true`、`beacon_window`、`beacon_stride` | 保持 `beacon_layout_policy='configured'`；stride 不大于 window |
| 兼容整段布局 | `SA_TRAINING__BEACON_LAYOUT_POLICY='whole'` | 会规范化部分布局字段；先检查归档 train.yaml |
| 输出蒸馏 | `beacon_output_distill_weight`、teacher 参数 | 大于 0 会额外加载 teacher；预留内存与显存 |
| 显存优化 | `beacon_checkpoint_blocks`、`beacon_checkpoint_loss`、CPU offload | 训练默认开启；不用普通 gradient_checkpointing 代替 |

逐项默认值和字段说明见 [60 参数手册](60_parameters.md)。多卡使用 DDP 或 ZeRO-2；当前不支持 Beacon ZeRO-3。推理仅支持 greedy。

## 4. 校验计划与训练

先将选定文件 `SA_PLAN_ONLY=true` 后运行；要求 prepared 数据已存在，程序校验来源、协议、tokenizer、长度与划分，归档配置但不加载模型。计划目录的 `operation=plan`，不能直接用于测试。

将同一文件 `SA_PLAN_ONLY=false` 后运行正式训练。例如选择 OpenSeeker：

```bash
bash scripts/beacon/sh/10_train_openseeker.sh
```

选择 Research 则运行：

```bash
bash scripts/beacon/sh/11_train_research.sh
```

每次正式运行建立新实验，保存 `train.yaml`、`experiment.json`、脚本副本、日志和模型。默认 LoRA 结束后自动合并，成功的 `models/model/train_summary.json` 中 `final_model_path` 是测试使用的模型。合并失败将整个训练记录为失败，不能当作可测试实验。

## 5. 配置测试

三个测试文件相互独立。分别修改：

- `SA_TRAINING_RUN`：刚完成训练的实验根目录；Research 实验应位于 `beacon/research/`。
- `SA_BENCHMARK_DIR`：公共准备脚本产生的准确版本目录。
- `SA_RUN_NAME`：正式全量可用 `full`；小样本可用 `smoke`。
- `GENERATION`：先用正数 `limit` 验证流程，正式结果改为 `-1`。调整上下文、每轮输出、工具预算等参数时各方法保持可比较设置。

训练方法固定 `SA_MODEL_PATH=''`、`SA_PROMPT_DATA=''`：模型和提示词从 `SA_TRAINING_RUN` 解析，拒绝跨方法实验、计划目录和失败训练。

| Benchmark | 测试入口 | 执行条件 |
|---|---|---|
| BrowseComp | `scripts/beacon/sh/40_test_browsecomp.sh` | 题目与固定版工具已准备，在线搜索/访问环境可用 |
| BrowseComp-Plus | `scripts/beacon/sh/42_test_browsecomp_plus.sh` | 全量语料、prepared 和索引已准备，完全离线 |
| HotpotQA | `scripts/beacon/sh/44_test_hotpotqa.sh` | fullwiki 题目、独立维基语料和索引已准备，完全离线 |

```bash
bash scripts/beacon/sh/40_test_browsecomp.sh
bash scripts/beacon/sh/42_test_browsecomp_plus.sh
bash scripts/beacon/sh/44_test_hotpotqa.sh
```

每项生成一个独立测试目录。在线测试默认逐题运行；离线 `num_workers>1` 时每张可见卡加载一个完整模型副本。测试不会启动裁判。

## 6. 独立评分

将各项测试目录分别填入对应评分脚本的 `SA_EVALUATION_DIR`，目录应包含 `evaluation.json` 与 `results/`。不要填预测文件或训练目录。

| Benchmark | 评分入口 | 裁判配置 |
|---|---|---|
| BrowseComp | `scripts/beacon/sh/41_score_browsecomp.sh` | 服务器环境 `SCORER_*`；脚本 `JUDGE` 配置并发、重试、预算 |
| BrowseComp-Plus | `scripts/beacon/sh/43_score_browsecomp_plus.sh` | `SA_JUDGE__JUDGE_MODEL` 填本地裁判目录 |
| HotpotQA | `scripts/beacon/sh/45_score_hotpotqa.sh` | 默认本地 LLM judge + 官方 EM/F1；本地裁判目录必填 |

```bash
bash scripts/beacon/sh/41_score_browsecomp.sh
bash scripts/beacon/sh/43_score_browsecomp_plus.sh
bash scripts/beacon/sh/45_score_hotpotqa.sh
```

评分沿用测试的 benchmark 版本和 `limit`，核对方法、预测和 manifest 摘要。新的裁判设置创建新的评分目录，原结果保留。指标位于 `scores/<评分记录>/results/summary.metrics.json`。失败/缺失生成保持在分母内；裁判不完整会报错，不能将部分分数当作完成结果。

## 7. 归档与排查

实验归档在 `artifacts/experiments/beacon/<dataset>/<run>/`。依次检查根 `job.json` 的状态、`console.log`、模型训练摘要、各测试 `evaluation.json` 和评分指标。

使用 `scripts/tools/sh/81_prediction_report.sh` 检查预测轨迹；将 `METHOD='beacon'` 并填写对应 benchmark 和测试目录。目录规范见 [70 产物规范](70_artifacts.md)。遇到失败保留原记录，修正参数后重新运行，程序不自动覆盖或继续旧训练。
