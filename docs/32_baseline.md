# 32 Baseline 完整实验指南

本文字段名称表示逻辑参数：SH 中普通变量加 `SA_` 前缀，例如 `MODEL_PATH` 对应 `SA_MODEL_PATH`；嵌套键用双下划线，例如 `TRAINING.max_length` 对应 `SA_TRAINING__MAX_LENGTH`。环境变量如 `HF_ENDPOINT` 不加前缀。实际可编辑名称、类型与默认值见 [61 脚本参数清单](61_script_inventory.md)。

## 1. 实验范围与顺序

Baseline 是原始、未训练的 Qwen3.5 基座，使用专用 `base_qwen35` 加载逻辑。只有三项 benchmark 测试与评分，不提供训练或 adapter 合并入口。

安装环境 → 准备本地基座 → 准备三项 benchmark → 选择比较提示词 → 三项测试 → 三项独立评分 → 检查产物。

按 [10 环境准备](10_environment.md)、[20 数据准备](20_data.md) 完成模型及 benchmark 准备。如果使用内置 Polar 提示词，不需要准备训练数据。

## 2. 修改各项测试参数

| 参数 | 设置 | 说明 |
|---|---|---|
| `SA_MODEL_PATH` | Linux 原始 Qwen3.5 本地目录 | 不能是 LoRA adapter 或 Beacon checkpoint |
| `SA_TRAINING_RUN` | 固定 `空字符串` | baseline 不依赖任何训练实验 |
| `SA_PROMPT_DATA` | 默认 `空字符串` | 使用内置原始 Polar system/schema |
| `SA_BENCHMARK_DIR` | 对应共享 benchmark 版本 | 与 Plain/Beacon 使用同一题目与语料版本 |
| `SA_RUN_NAME` | 如 `full` / `smoke` | 程序追加 UTC 时间与随机后缀 |
| `GENERATION` | 与比较方法一致的推理预算 | 正数 `limit` 验证流程，正式全量 `-1` |

如要与 Research 训练方法对比，将每个 baseline 测试脚本的 `SA_PROMPT_DATA` 设为该训练数据版本的规范化 `train.jsonl`，只读取 system/schema。不能使用 Research 原始角色导出，也不能在一份输入中混合不同 system。程序不会向模型输入训练问题或答案。

## 3. 三项测试

| Benchmark | 入口 | 前置条件 |
|---|---|---|
| BrowseComp | `scripts/baseline/sh/40_test_browsecomp.sh` | 在线搜索、读取、总结服务；准备好的官方工具与题目 |
| BrowseComp-Plus | `scripts/baseline/sh/42_test_browsecomp_plus.sh` | 独立全量语料与本地 BM25 索引 |
| HotpotQA | `scripts/baseline/sh/44_test_hotpotqa.sh` | fullwiki dev、独立维基摘要语料与本地索引 |

逐个编辑并运行：

```bash
bash scripts/baseline/sh/40_test_browsecomp.sh
bash scripts/baseline/sh/42_test_browsecomp_plus.sh
bash scripts/baseline/sh/44_test_hotpotqa.sh
```

测试使用本地模型，不需要部署模型服务。BrowseComp 工具连接外网；另外两项禁止联网。离线多卡可改 `SA_GENERATION__NUM_WORKERS`，并在 `ENVIRONMENT` 中列出足够的不同 GPU；每张卡加载一个完整模型副本。

记录位于 `artifacts/experiments/baseline/untrained/evaluation/<benchmark>/<测试记录>/`，每次独立创建。保留 `evaluation.json` 和 `results/`，下一步填入评分脚本。

## 4. 三项独立评分

每个评分文件修改 `SA_EVALUATION_DIR` 为同方法、同 benchmark 的成功测试目录：

| Benchmark | 入口 | 裁判设置 |
|---|---|---|
| BrowseComp | `scripts/baseline/sh/41_score_browsecomp.sh` | 服务器配置 `SCORER_URLS`、`SCORER_API_KEY`、`SCORER_MODEL_NAME` |
| BrowseComp-Plus | `scripts/baseline/sh/43_score_browsecomp_plus.sh` | `SA_JUDGE__JUDGE_MODEL` 本地模型目录 |
| HotpotQA | `scripts/baseline/sh/45_score_hotpotqa.sh` | 默认官方 EM/F1 加本地 LLM judge，本地裁判目录必填 |

```bash
bash scripts/baseline/sh/41_score_browsecomp.sh
bash scripts/baseline/sh/43_score_browsecomp_plus.sh
bash scripts/baseline/sh/45_score_hotpotqa.sh
```

评分自动复用原测试的题目范围和版本，检查预测摘要，拒绝将 Plain/Beacon 测试归到 baseline。每次评分位于测试下 `scores/` 的新目录，汇总文件为 `results/summary.metrics.json`。正式比较需同时对齐模型、提示词来源、推理预算、检索设置、题目范围和裁判配置。

## 5. 结果检查

确认测试和评分根 `job.json` 都为 `complete`，查看日志及指标中的覆盖率/完整性。使用 `scripts/tools/sh/81_prediction_report.sh` 时将 `METHOD='baseline'` 并填写 benchmark 和测试目录。参数查 [60 参数手册](60_parameters.md)，归档查 [70 产物规范](70_artifacts.md)。
