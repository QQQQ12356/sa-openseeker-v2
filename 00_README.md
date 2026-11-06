# 00 SearchAgent 实验工程

本项目从 `sa-openseeker-v2` 的 `tmp` 分支重构，运行目标为 **Linux / Python 3.11 / NVIDIA CUDA**。Windows 用于编辑代码和无模型验证。

## 支持矩阵

| 方法 | OpenSeeker 训练 | Research 训练 | BrowseComp | BrowseComp-Plus | HotpotQA |
|---|---|---|---|---|---|
| Plain | 支持 | 支持 | 测试 + 评分 | 测试 + 评分 | 测试 + 评分 |
| Beacon | 支持 | 支持 | 测试 + 评分 | 测试 + 评分 | 测试 + 评分 |
| Baseline | — | — | 测试 + 评分 | 测试 + 评分 | 测试 + 评分 |

Baseline 指未训练的原始 Qwen3.5 基座。Plain 默认使用 LoRA；Beacon 使用工具观测压缩及 LoRA。两种训练方法均沿用原分支的训练与损失默认配置。

## 阅读与执行顺序

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [10 环境准备](docs/10_environment.md) | Linux 环境、依赖安装、镜像、GPU 和在线服务 |
| 2 | [20 数据准备](docs/20_data.md) | 模型、两种训练数据、筛选、三项 benchmark 的准备 |
| 3 | [30 Plain 指南](docs/30_plain.md) | 数据选择、训练计划、训练、三项测试与评分 |
| 3 | [31 Beacon 指南](docs/31_beacon.md) | 压缩参数、训练计划、训练、三项测试与评分 |
| 3 | [32 Baseline 指南](docs/32_baseline.md) | 基座与提示词配置、三项测试与评分 |
| 4 | [60 参数手册](docs/60_parameters.md) | SH 脚本变量、训练参数、压缩参数及修改提示 |
| 5 | [70 产物规范](docs/70_artifacts.md) | 目录、命名、运行身份、来源校验、失败归档 |
| 6 | [82 Excel 导出](docs/82_benchmark_excel.md) | benchmark 评分汇总、逐题明细、完整参数与修改提示 |
| 参考 | [90 迁移说明](docs/90_migration.md) | 来源提交、架构决策、旧目录迁移边界 |
| 验证 | [99 验证记录](docs/99_validation.md) | 已执行检查和 Linux 后续验证范围 |

## 脚本使用约定

1. 先编辑待执行 SH 文件中的 `SA_*` 参数与环境变量；必改项有中文注释。
2. 激活环境后直接运行该文件，例如 `bash scripts/plain/sh/10_train_openseeker.sh`。不需要追加参数或设置命令行覆盖。
3. 方法 SH 各自包含完整配置，仅调用同目录体系内的同名 PY，不调用其他方法的 SH。公共数据和 benchmark 版本可以复用。
4. 训练完成后，将终端输出的实验目录复制到对应测试脚本的 `SA_TRAINING_RUN`。
5. 测试完成后，将测试目录复制到对应评分脚本的 `SA_EVALUATION_DIR`。评分自动沿用数据版本与题目范围。
6. 所有运行产物写入 `artifacts/`。每次运行创建独立目录；失败保留清单和日志，修正后重新运行。
7. 评分完成后，编辑并运行 `bash scripts/tools/sh/82_benchmark_excel.sh`，将一份或多份 benchmark 评分结果汇总到 Excel；详细步骤见 [82 Excel 导出](docs/82_benchmark_excel.md)。

## 代码结构

```text
scripts/                  按 sh/、py/ 分离的独立运行文件
  data/sh/、data/py/       模型下载、数据处理、筛选和 benchmark 准备
  plain/sh/               两个训练入口 + 三项测试/评分（编辑与运行）
  plain/py/               同名 Python 执行适配器（不维护参数默认值）
  beacon/sh/、beacon/py/   Beacon 独立入口与执行适配器
  baseline/sh/、baseline/py/  Baseline 独立入口与执行适配器
  tools/sh/、tools/py/     环境安装、HTML 检查、评分 Excel 导出与验证
  templates/              离线 HTML 模板
search_comp/
  experiments/            统一流程、参数校验和归档接口
  workflows/              数据预处理及 benchmark 实现
  data/ models/ trainer/  协议、模型和训练算法
  evaluation/ offline/    Agent、打分及本地检索
  utils/ milestones/      运行工具及必要的模型适配
artifacts/                运行时创建，不纳入代码版本控制
```

当前交付没有执行真实数据下载、预处理、训练、模型推理或远程裁判请求。检查结果见 [99 验证记录](docs/99_validation.md)。

脚本职责、类型规则和调用方式见 [05 脚本结构](docs/05_script_layout.md)，每份 SH 的完整变量表见 [61 脚本参数清单](docs/61_script_inventory.md)。
