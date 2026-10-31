# 00 文档与入口索引

首次开展 Plain 实验，按 [21 Plain 完整实验指南](plain/21_plain_experiment_guide.md) 执行；Beacon 实验参照 [32 Beacon 完整实验指南](beacon/32_beacon_experiment_guide.md)。下面各文档用于对应阶段的参数与实现细节。

文档保留原有编号，按功能模块存放；本文件是总入口。相关 JSON 资料与对应文档放在同一目录。

| 文档目录 | 模块内容 | 对应 shell 目录 |
|---|---|---|
| `common/` | 环境、统一训练测试协议和解码 | `common/`、多个入口共用 |
| `data/` | 原始数据、预处理与长度统计 | `data/` |
| `plain/` | Plain 训练与完整实验 | `plain/` |
| `baseline/` | 原始模型和 adapter 对照 | `baseline/` |
| `beacon/` | Beacon 训练、压缩和 OOM | `beacon/` |
| `benchmark/` | OpenSeeker、在线和离线评测 | `benchmark/` |
| `report/` | HTML 数据与轨迹报告 | `report/` |
| `tools/` | 多卡、显存和环境细节 | `tools/` |
| `architecture/` | 模型架构、旧实现和蒸馏 | 多个入口共用 |
| `project/` | 迁移、映射和验证记录 | 项目级资料 |

| 文档 | 内容 |
|---|---|
| [01 环境](common/01_environment.md) | Python、依赖、离线缓存、验证边界 |
| [10 数据](data/10_data.md) | Polar 来源、规范化、长度/质量/观测筛选、产物 |
| [11 长度口径](data/11_lengths.md) | 提供的分布文件与训练序列实测长度的区别 |
| [20 Plain](plain/20_plain.md) | 一键训练、测试与配置快照 |
| [21 Plain 完整实验指南](plain/21_plain_experiment_guide.md) | OpenSeeker 训练与诊断、BrowseComp 在线及 BrowseComp-Plus / Hotpot 离线数据、生成、评分与归档 |
| [22 原始基座与 adapter 对照](baseline/22_qwen35_baseline_and_adapter.md) | 21 直接测试 Plain adapter；原始 Qwen3.5-4B 四条基准路线、参数和复读诊断 |
| [23 Plain / Beacon 解码与重复惩罚](common/23_decoding.md) | 贪心、采样、压缩后可见历史、重复惩罚、随机种子和各 benchmark 参数 |
| [24 原始 Qwen3.5 独立基线](baseline/24_qwen35_base_benchmark.md) | 独立模型加载、各基准脚本、数据与评分、完整参数和对照 |
| [30 Beacon](beacon/30_beacon.md) | 压缩范围、损失、蒸馏与训练入口 |
| [31 随机压缩](beacon/31_random_compression.md) | fixed/random 模式、候选数组、采样粒度与多卡行为 |
| [32 Beacon 完整实验指南](beacon/32_beacon_experiment_guide.md) | 按实验顺序：环境、数据、筛选、计划、训练、合并、在线/离线评测与归档 |
| [40 评测](benchmark/40_evaluation.md) | OpenSeeker 参考实现、答案提取、EM/F1 与工具后端 |
| [41 HTML](report/41_html.md) | 数据与预测的离线浏览 |
| [42 官方 Benchmark](benchmark/42_openseeker_benchmark.md) | BrowseComp、固定官方 v1/v2、Beacon native、LLM 裁判、覆盖审计、一键运行 |
| [43 离线基准](benchmark/43_offline_benchmarks.md) | BrowseComp-Plus / HotpotQA、持久化 BM25、断网构建与评测、参数和修改提示 |
| [44 离线来源](benchmark/44_offline_sources.md) | 固定官方代码、数据版本、许可证与升级规则 |
| [45 统一训练与测试](common/45_unified_agent_flow.md) | 保留原始 system、训练提示来源、离线能力适配、算法一致性、在线/离线入口、变量和修改说明 |
| [50 多卡](tools/50_multigpu.md) | torchrun、DDP、ZeRO-2/3、每卡日志、Beacon 与固定语料多卡测试 |
| [51 显存测试](tools/51_memory_probe.md) | 特定长度数据、短步数训练、峰值显存 |
| [52 四卡 A800 长轨迹 OOM](beacon/52_beacon_oom.md) | 原因、重计算修复、参数说明与修改提示 |
| [53 自包含训练配置](tools/53_self_contained_training.md) | Shell 内训练、LoRA、Beacon、DeepSpeed 参数及 plan 检查 |
| [60 架构](architecture/60_architecture.md) | 保留的核心实现、训练协议与分段损失 |
| [61 旧实现](architecture/61_legacy_implementation.md) | 保留的早期实现和算法背景 |
| [62 第一阶段蒸馏](architecture/62_stage1_distillation.md) | 第一阶段配置与训练 |
| [63 第二阶段蒸馏](architecture/63_stage2_distillation.md) | 第二阶段配置与损失 |
| [64 Qwen3.5 消融](beacon/64_qwen35_native_linear_ablation.md) | 线性状态和模型结构消融 |
| [65 压缩分析](beacon/65_qwen35_compression_analysis.md) | Qwen3.5 压缩与内存分析 |
| [66 多卡实现细节](tools/66_distributed_details.md) | 设备绑定、DeepSpeed、日志和保存约定 |
| [67 环境细节](tools/67_environment_details.md) | 安装、兼容性及服务器环境 |
| [90 迁移](project/90_migration.md) | 来源、目录重组、保留范围与入口映射 |
| [99 验证](project/99_validation.md) | 本次实际检查、结果和未实测范围 |

`61`—`67` 是保留的原算法、多卡和环境细节文档；历史实验的数字和模型配置不代表本项目新实验结果。当前工作流与默认路径以上表为准。

Shell 脚本按功能分目录，保留全局唯一的历史编号，方便文档和 Python 入口对应；编号范围中的空号留作扩展。所有命令以项目根目录为基准：

| 目录 | 用途 | 常用 shell 入口 |
|---|---|---|
| `scripts/sh/common/` | 共享环境及日志函数 | `00_common.sh`（被其他脚本 source） |
| `scripts/sh/tools/` | 总入口、测试和环境探针 | `01_test.sh`、`02_workflow.sh`、`03_check_environment.sh`、`59_check_agent_alignment.sh`、`60_searchagent_probe.sh` |
| `scripts/sh/data/` | 数据准备、筛选、证据库和审计 | `10_prepare.sh`、`11_select.sh`、`12_select_memory_probe.sh`、`13_filter_observations.sh`、`15_build_corpus.sh` |
| `scripts/sh/plain/` | Plain 训练、评测、显存和模型合并 | `20_plain_train.sh`、`21_plain_eval.sh`、`26_plain_memory_probe.sh`、`27_plain_workflow.sh`、`28_plain_4xa800.sh`、`29_plain_merge_adapter.sh` |
| `scripts/sh/beacon/` | Beacon 训练、评测、合并和显存 | `30_beacon_train.sh`、`31_beacon_eval.sh`、`35_beacon_merge_adapter.sh`、`37_beacon_memory_probe.sh`、`38_beacon_4xa800.sh` |
| `scripts/sh/report/` | HTML、指标汇总和图表 | `40_dataset_html.sh`、`41_predictions_html.sh`、`42_summarize_results.sh`、`43_export_excel.sh`、`44_plot.sh`、`45_report.sh` |
| `scripts/sh/benchmark/` | 在线与离线基准 | `46_prepare_benchmark.sh`、`47_openseeker_benchmark.sh`、`48_serve_checkpoint.sh`、`49_benchmark_env.example.sh`、`56_browsecomp_plus.sh`、`57_hotpotqa_offline.sh`、共享 `58_offline_common.sh` |
| `scripts/sh/baseline/` | 原始模型独立评测 | `70_qwen35_base_eval.sh` 总入口；71 OpenSeeker、72 在线 BrowseComp、73 离线基准 |

Plain 21 和 Beacon 31 分别独立调用通用 Python 评测器，不能通过 `KIND` 切换成另一种入口。默认训练入口在 shell 内包含训练和 DeepSpeed 配置，不读取外部基础配置，详见 [53](tools/53_self_contained_training.md)。同一模型类型内仍有便利用的转发入口：Plain 28 调用 Plain 20，Beacon 38 调用 Beacon 30；Plain 27 是按阶段分发的工作流入口。这些不会跨 Plain / Beacon 调用。

22/23、32—34 为保留的 Search-R1 入口，其数据不随本次迁移，需要另行准备。Plain 旧的 24/25 无训练 shell 入口已移除；原始 Qwen3.5 基线使用 baseline 70–73。`scripts/py/` 保留参数接口便于自动化；日常运行优先修改 shell 顶部变量。

46—49 为新增官方 Benchmark 流程：46 准备官方代码与 BrowseComp，47 在线生成及官方裁判，48 部署训练后的 Plain checkpoint，49 是私有 API 配置示例。详见 [42 说明](benchmark/42_openseeker_benchmark.md)。
