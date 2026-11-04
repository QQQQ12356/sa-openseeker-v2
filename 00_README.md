# 00 sa-openseeker-v2

当前训练数据集为 **OpenSeeker** 和 **Research**。BrowseComp Qwen3.5 SFT 的预处理、适配器、训练脚本已移除；BrowseComp、BrowseComp-Plus、HotpotQA 三项独立 benchmark 继续保留。原始数据与已有输出未改动。

完整实验先按数据集选择 [70 OpenSeeker 全流程](docs/data/70_openseeker_experiment_guide.md) 或 [71 Research 全流程](docs/data/71_research_experiment_guide.md)，两者均包含下载、预处理、原始/规范化 HTML、全量分布分析、三方法训练、三项测试与独立评分。脚本变量查 [68](docs/project/68_training_benchmark_guide.md)；训练参数逐项说明见 [69 训练配置参数](docs/project/69_training_parameters.md)；全部文档见 [00 索引](docs/00_index.md)。所有预处理、训练、真实推理和评分均在 Linux 执行。Windows 仅修改代码并做无模型静态/单元检查。

| 方法目录 | 训练入口 | 测试入口 | 评分入口 |
|---|---|---|---|
| `scripts/sh/plain/` | `10_train_openseeker.sh`、`11_train_research.sh` | `40_test_browsecomp.sh`、`42_test_browsecomp_plus.sh`、`44_test_hotpotqa.sh` | `41_score_browsecomp.sh`、`43_score_browsecomp_plus.sh`、`45_score_hotpotqa.sh` |
| `scripts/sh/beacon/` | 同上，配置独立 | 同上，Beacon 推理 | 同上，独立进程 |
| `scripts/sh/baseline/` | 同上，全参数 SFT | 同上，使用训练后的模型 | 同上，独立进程 |

每个方法有自己的脚本及参数，直接修改对应文件或设置其环境变量。底层复用已有 Python 实现。Plain 默认 LoRA；Beacon 使用已有压缩训练；baseline 新训练入口采用标准全参数 SFT。Baseline 的旧 `70`–`75` 入口仍表示**未训练原始 Qwen3.5 基座**，供历史结果复现，不能用于替代新训练模型的测试。

Linux 最小示例（先完成预处理、模型缓存与 benchmark 准备）：

```bash
# 修改对应训练脚本 YAML 或设置路径；plan 不加载模型，但要求预处理数据存在。
ACTION=plan bash scripts/sh/plain/11_train_research.sh
bash scripts/sh/plain/11_train_research.sh

# RUN_DIR 使用训练输出的具体实验目录。
RUN_DIR=/path/to/research_run bash scripts/sh/plain/44_test_hotpotqa.sh
PREDICTIONS=/path/to/test/predictions.jsonl JUDGE_MODEL=/models/judge \
  bash scripts/sh/plain/45_score_hotpotqa.sh
```

测试和评分分开执行，三个 benchmark 均提供 EM、F1、LLM judge。训练与测试共用原始 system/schema 和轨迹序列化；离线测试仅调整工具能力和答案格式。Research 评测必须读取其对应训练实验或规范化训练 JSONL。完整参数、离线准备步骤、失败样本口径、脚本兼容说明与修改提示见 68/69。
