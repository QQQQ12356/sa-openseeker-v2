# 30 Beacon 训练与评测

主入口为 `scripts/sh/30_beacon_train.sh`，使用方式与 Plain 相同。模型支持 Qwen3 和 Qwen3.5，默认 whole observation 压缩。OpenSeeker 中只有 `tool_response` 的正文进入压缩区域：工具调用 JSON、system、问题、推理和原始回答都保留。训练和 agent 评测共用序列化与 observation 区域定义。

可修改 `beacon_ratio`、LoRA、分段损失、续写损失、information/readout/output 蒸馏等。默认关闭需要完整 teacher 的额外输出蒸馏。模型结构、Qwen3.5 线性状态处理、continuation、问题记忆等已有实现和测试均保留，详见 60—65 文档。默认各主 CE 片段权重均为 1，可设置 `{}` 恢复默认。

多卡使用 DDP 或 ZeRO-2。Beacon 动态分段和压缩训练不支持 ZeRO-3，launcher 会在启动前明确拒绝。CPU activation offload 默认保留；开启 teacher 蒸馏会增加内存和显存需求。

评测用 `31_beacon_eval.sh`；多卡评测每个 GPU 加载一份模型，按样本索引分片，并恢复原始顺序汇总指标；四卡设置和命令见 [50 多卡测试](50_multigpu.md#beacon-多卡测试)。`35_beacon_merge_adapter.sh` 保留适配器恢复合并入口，脚本顶部填写 TARGET 即可。`37_beacon_memory_probe.sh` 用于长样本短步数测试。本次没有运行真实模型训练或生成。

随机压缩比通过顶部 `COMPRESSION_MODE="random"` 与 `COMPRESSION_RATIOS=(8 16 32 64)` 开启；默认 fixed。每条训练轨迹采样一次，验证/生成仍使用 `BEACON_RATIO`。配置、日志与多卡规则见 [31 随机压缩](31_random_compression.md)。

训练后可用 `MODE=native RUN_DIR=... bash scripts/sh/47_openseeker_benchmark.sh` 增加独立 BrowseComp 在线评测，保留 Beacon 压缩并使用官方 search/visit 与 LLM 裁判。不能用普通 SGLang 替代 Beacon 推理。
原有测试入口继续保留，准备依赖、完整示例与比较边界见 [42 官方 Benchmark](42_openseeker_benchmark.md)。

无需外部 API 的固定语料测试使用 56（BrowseComp-Plus）或 57（HotpotQA），参数顶部填 RUN_DIR 即可连接训练产物；先离线建库和持久化索引，原 21/31/47 入口保留。两套基准的运行、评分与预算差异见 [43 离线手册](43_offline_benchmarks.md)。56/57 也支持设置 `NUM_WORKERS` 和 `CUDA_VISIBLE_DEVICES` 按题多卡并行；每卡保留完整模型，多卡行为与分阶段命令见 [43.7.4](43_offline_benchmarks.md#4374-beacon-多卡测试56-与-31-的能力不同)。

## 四卡长轨迹与重计算

OpenSeeker 30/37现在显式启用 `beacon_checkpoint_blocks: true`、SDPA和1024-token keep分块，38提供四卡A800入口。blocks重算attention、MLP及Qwen3.5 reader/writer，不重放记忆更新，不改变whole布局。旧顶层 `gradient_checkpointing` 仅告警，`beacon_checkpoint_loss` 只作用于loss，两者都不能代替blocks。直接使用Search-R1 YAML时需要自行开启blocks。详细参数和OOM定位见 [52](52_beacon_oom.md)，多卡规则见 [50](50_multigpu.md)。
