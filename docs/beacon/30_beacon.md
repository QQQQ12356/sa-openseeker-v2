# 30 Beacon 训练与评测

从数据准备到训练、合并、评测的顺序操作见 [32 完整实验指南](32_beacon_experiment_guide.md)；本文侧重参数和算法入口。

当前保留训练原始 system、测试提示词来源选择、必要的离线适配及全部入口的保留范围见 [45 统一流程](../common/45_unified_agent_flow.md)。

主入口为 `scripts/sh/beacon/30_beacon_train.sh`，使用方式与 Plain 相同。模型支持 Qwen3 和 Qwen3.5，默认 whole observation 压缩。OpenSeeker 中只有 `tool_response` 的正文进入压缩区域：工具调用 JSON、system、问题、推理和原始回答都保留。训练和 agent 评测共用序列化与 observation 区域定义。

可修改 `beacon_ratio`、LoRA、分段损失、续写损失、information/readout/output 蒸馏等。默认关闭需要完整 teacher 的额外输出蒸馏。模型结构、Qwen3.5 线性状态处理、continuation、问题记忆等已有实现和测试均保留，详见 60—65 文档。默认各主 CE 片段权重均为 1，可设置 `{}` 恢复默认。

多卡使用 DDP 或 ZeRO-2。Beacon 动态分段和压缩训练不支持 ZeRO-3，launcher 会在启动前明确拒绝。CPU activation offload 默认保留；开启 teacher 蒸馏会增加内存和显存需求。

评测用 `31_beacon_eval.sh`，它独立调用通用 Python 评测器，不调用 Plain 的 21；31 只接受 `KIND=beacon`。多卡评测每个 GPU 加载一份模型，按样本索引分片，并恢复原始顺序汇总指标；四卡设置和命令见 [50 多卡测试](../tools/50_multigpu.md#beacon-多卡测试)。`35_beacon_merge_adapter.sh` 保留适配器恢复合并入口，脚本顶部填写 TARGET 即可。`37_beacon_memory_probe.sh` 用于长样本短步数测试。本次没有运行真实模型训练或生成。

随机压缩比通过 30 内嵌 YAML 的 `beacon_compression_mode: random` 与 `beacon_compression_ratios` 开启；默认 fixed。每条训练轨迹采样一次，验证/生成仍使用 `beacon_ratio`。配置、日志与多卡规则见 [31 随机压缩](31_random_compression.md)。

训练后可用 `MODE=native RUN_DIR=... bash scripts/sh/benchmark/47_openseeker_benchmark.sh` 增加独立 BrowseComp 在线评测，保留 Beacon 压缩并使用官方 search/visit 与 LLM 裁判。不能用普通 SGLang 替代 Beacon 推理。
原有测试入口继续保留，准备依赖、完整示例与比较边界见 [42 官方 Benchmark](../benchmark/42_openseeker_benchmark.md)。

无需外部 API 的固定语料测试使用 56（BrowseComp-Plus）或 57（HotpotQA），参数顶部填 RUN_DIR 即可连接训练产物；先离线建库和持久化索引，原 21/31/47 入口保留。两套基准的运行、评分与预算差异见 [43 离线手册](../benchmark/43_offline_benchmarks.md)。56/57 也支持设置 `NUM_WORKERS` 和 `CUDA_VISIBLE_DEVICES` 按题多卡并行；每卡保留完整模型，多卡行为与分阶段命令见 [43.7.4](../benchmark/43_offline_benchmarks.md#4374-beacon-多卡测试56-与-31-的能力不同)。

## 四卡长轨迹与重计算

OpenSeeker 30/37 显式启用 `beacon_checkpoint_blocks: true`、SDPA；默认 whole 的 keep_window 为 0（整段 keep），38 提供四卡 A800 入口。默认 `beacon_layout_policy: configured` 使用脚本中的布局字段；`whole` 兼容策略会强制整段布局。blocks 重算 attention、MLP 及 Qwen3.5 reader/writer，不重放跨窗状态更新。旧顶层 `gradient_checkpointing` 仅告警，`beacon_checkpoint_loss` 只作用于 loss，两者都不能代替 blocks。Search-R1 的 34 脚本可自行开启 blocks。详细参数和 OOM 定位见 [52](52_beacon_oom.md)，多卡规则见 [50](../tools/50_multigpu.md)。
