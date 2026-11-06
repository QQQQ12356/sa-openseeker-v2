# 70 产物目录、命名与归档规范

## 1. 总体规则

所有新产物位于项目根目录 `artifacts/`，被版本控制忽略。相对路径以项目根解析；Linux 本地模型和已有数据可用绝对路径作为输入。原始外部输入只读取。

实验标签采用英文字母、数字、下划线和连字符。程序为每次操作追加 UTC 时间和随机后缀，如 `qwen35_L16384_s42_20261106T083000123456Z_a1b2c3`。固定数据版本名由操作者明确填写；禁止覆盖已生成 prepared 数据或索引。

## 2. 标准目录

```text
artifacts/
├── environments/train/                   独立 Linux Python 环境和实际依赖清单
├── cache/                                Hub、Torch、XDG 缓存
├── reports/benchmarks/<导出记录>/          benchmark_scores.xlsx、export_manifest.json 及操作归档
├── models/<model_name>/<commit>/          模型快照、下载清单和文件摘要
│   └── download_manifest.json
├── models/jobs/<下载记录>/                 模型下载操作的日志和来源
├── datasets/
│   ├── openseeker/
│   │   ├── raw/<polar或zephyr>/<commit>/  原文件、exports/、下载清单
│   │   ├── prepared/<profile>/            规范化数据或派生筛选版本
│   │   │   └── reports/<报告记录>/         HTML 和报告操作清单
│   │   └── jobs/<数据操作记录>/            下载/处理/筛选的配置与日志
│   └── research/
│       ├── raw/research/<commit>/
│       ├── prepared/<profile>/
│       └── jobs/<数据操作记录>/
├── benchmarks/<benchmark>/<version>/
│   ├── jobs/<准备阶段记录>/                download / prepare / index 操作日志
│   ├── vendor/OpenSeeker/                 仅 BrowseComp：固定版本在线工具
│   ├── questions.jsonl                    仅 BrowseComp：解密题目
│   ├── questions.manifest.json            仅 BrowseComp：来源与摘要
│   ├── raw/                               离线题目与独立全量语料
│   ├── official/                          离线固定版本官方评测资产
│   ├── assets.json                        离线版本和来源清单
│   ├── prepared/                          离线规范化题目/语料与清单
│   └── index.sqlite                       离线持久化 BM25 索引
└── experiments/
    ├── plain/<openseeker或research>/<run>/
    ├── beacon/<openseeker或research>/<run>/
    └── baseline/untrained/
```

Plain/Beacon 的训练实验内部结构：

```text
<run>/
├── job.json                              操作身份、完成/失败状态、代码摘要、环境白名单
├── source_launcher.sh                    操作者实际编辑的 SH 参数与环境副本
├── source_script.py                      同名 PY 执行适配器副本
├── console.log                           标准输出和错误日志
├── experiment.json                       方法、数据身份、模型、manifest、数据指纹
├── train.yaml                            最终训练配置
├── launch.json                           内部分布式进程参数，用户无需修改
├── models/model/
│   ├── resolved_config.json              trainer 实际解析配置
│   ├── train_summary.json                最终模型和适配器路径
│   ├── final*/                           完整模型、适配器及合并产物
│   └── checkpoint-*/ ...                 trainer 约定的周期保存与指标
└── evaluation/<benchmark>/<evaluation>/
    ├── job.json
    ├── source_script.py
    ├── console.log
    ├── evaluation.json                   测试身份、预测路径、预测/manifest 摘要
    ├── results/
    │   ├── manifest.json                 题目、选择范围、模型、工具及代码来源
    │   ├── predictions.jsonl             离线两项预测
    │   └── result_tool<N>.jsonl           在线 BrowseComp 预测
    ├── reports/<报告记录>/                预测 HTML 和报告清单
    └── scores/<score>/
        ├── job.json
        ├── source_script.py
        ├── console.log
        └── results/                      官方指标、EM/F1、裁判逐题记录与 summary.metrics.json
```

Baseline 使用同样的 `evaluation/<benchmark>/...` 结构，直接位于 `baseline/untrained/` 下，没有训练层或模型复制。

## 3. 身份和来源校验

| 阶段 | 校验内容 | 防止的问题 |
|---|---|---|
| 预处理 | 原文件摘要、tokenizer 指纹、编码协议、来源和长度 | 同名数据内容不一致 |
| 训练 | 数据身份、协议、tokenizer 名称、长度上限、划分存在与训练非空 | Research/OpenSeeker 混用、静默截断 |
| 方法测试 | 训练已完成，方法与数据身份匹配，最终模型存在；训练提示词数据摘要 | 误用计划目录、失败训练、跨方法模型 |
| 基座测试 | baseline 无 TRAINING_RUN，原始 Qwen3.5 配置且无 Beacon/adapter | 基座与训练后模型混淆 |
| 离线测试 | 题目/语料摘要，索引来源、标题权重和完整性 | 检索资产与题目版本不符 |
| 评分 | 测试完成、方法/benchmark 匹配、预测和测试 manifest 摘要不变、题目版本与 LIMIT 相同 | 给错方法评分、修改预测后复用旧身份 |

测试记录保存模型路径，不复制完整权重。保留模型和训练数据原位置；如跨服务器迁移导致绝对路径失效，应从准备和训练阶段创建新的可追溯记录。当前不提供自动重写历史清单工具。

## 4. 状态与重试

`job.json` 的 `status` 为 `running`、`complete` 或 `failed`；操作类型另区分 `plan` 与 `train`。正常异常会写入 failed 并保留错误类型、日志和结束时间；操作系统强制终止可能保留 running，需结合日志判断，不能当作成功。

- 训练计划不产生模型；设置 SA_PLAN_ONLY=false 后重新执行创建正式训练记录。
- 训练和测试失败后不自动续写旧目录；修正参数，重新运行生成新的记录。
- 原始下载以固定 commit 归档，可以重试；预处理/索引部分失败时保留原文件与日志，选择新版本重建。
- 评分可针对同一次成功测试重复执行，每次使用新 score 目录。裁判错误未消除时结果不完整。
- 不使用“最新目录”或模糊模型扫描；目录由使用者从上一步输出明确复制。

## 5. 存储管理

可清理与实验无关的下载缓存，但不能删除正在使用的模型、父 prepared 数据、原始公开划分输入或 benchmark 版本。代码仓库只保留源码、配置和编号文档；大文件不提交。归档对外分享时应同时保留指标、来源清单、脚本副本和日志，并自行检查服务返回内容中的敏感数据。

## 6. Excel 汇总报告

运行 `scripts/tools/sh/82_benchmark_excel.sh`，在 `artifacts/reports/benchmarks/<导出记录>/` 创建 `benchmark_scores.xlsx`、`export_manifest.json` 和标准 job/脚本/日志归档。清单记录纳入与跳过的摘要、逐题源文件 SHA256、工作簿 SHA256、各表行数及文本截断/清理数量。默认只纳入完成的评分，不重新执行评分和预测来源校验。汇总范围、完整参数及字段说明见 [82 Excel 导出](82_benchmark_excel.md)。
