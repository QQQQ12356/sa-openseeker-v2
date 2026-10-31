# 20 Plain 训练与评测

从数据准备到训练、OpenSeeker 测试、BrowseComp 与 Hotpot 测试和评分的完整执行顺序见 [21 Plain 完整实验指南](21_plain_experiment_guide.md)。

训练与 DeepSpeed 参数均可在各自 shell 内修改；配置来源、约束和计划审查见 [53 自包含训练配置](../tools/53_self_contained_training.md)。

21 直接加载 Plain LoRA adapter、基座模型对照和四类基准的命令与参数见 [22 对照评测指南](../baseline/22_qwen35_baseline_and_adapter.md)。47/56/57 仍需完整合并模型。

当前保留训练原始 system、测试提示词来源选择、必要的离线适配及全部入口的保留范围见 [45 统一流程](../common/45_unified_agent_flow.md)。

修改 `scripts/sh/plain/20_plain_train.sh` 顶部 Python、GPU 列表和脚本中的 YAML，然后直接执行。默认 Qwen3.5-4B、16k、LoRA、batch=1；`model_family` 支持 qwen3 / qwen35。`ACTION="plan"` 只生成实验目录、配置快照、指纹和启动命令，`train` 才加载模型训练。数据路径与 tokenizer 必须在 manifest 中匹配。

每次运行创建新的 `outputs/experiments/<name>/`，保留 `experiment.json`、`train.yaml`、`source_config.yaml`、运行日志与模型产物。根据 `train_summary.json` 的 `final_model_path` 定位推理模型；LoRA 合并与原训练器行为保持一致。计划后的代码或数据指纹改变会被拒绝，应重新创建实验计划。

多卡直接修改 `CUDA_VISIBLE_DEVICES="0,1,2,3"`，默认随可见卡数推断进程数。支持 DDP、ZeRO-2，以及仅 Plain 支持的 ZeRO-3，见 [50 多卡](../tools/50_multigpu.md)。

评测修改 `21_plain_eval.sh` 的 `RUN_DIR` 或 `MODEL_PATH`，以及 `SPLIT`、工具后端和预算后执行。21 直接调用通用 Python 评测器，不执行 Beacon shell；Beacon 的 31 同样独立。21 只接受 `KIND=plain`，31 只接受 `KIND=beacon`；旧的 `KIND=plain bash .../31_beacon_eval.sh` 应改用 21。`RUN_DIR` 可自动定位模型和测试数据。`DRY_RUN=1` 只检查并打印命令。`26_plain_memory_probe.sh` 是短步数显存实验入口，见 [51 显存测试](../tools/51_memory_probe.md)。

22/23 保留 Search-R1 训练与交互评测入口；未训练基线改用 [24 的独立 Qwen3.5 入口](../baseline/24_qwen35_base_benchmark.md)。Search-R1 数据需要另行准备。

训练后可用 47 默认 native 模式直接加载 Plain，与 Beacon 共用训练提示和工具循环；配置 API 后执行。原样官方 v1/v2 复现仍可通过 48 部署 Plain，再显式选择 47 的官方模式。
原有测试入口继续保留，准备依赖、完整示例与比较边界见 [42 官方 Benchmark](../benchmark/42_openseeker_benchmark.md)。

无需外部 API 的固定语料测试使用 56（BrowseComp-Plus）或 57（HotpotQA），参数顶部填 RUN_DIR 即可连接训练产物；先离线建库和持久化索引，原 21/31/47 入口保留。两套基准的运行、评分与预算差异见 [43 离线手册](../benchmark/43_offline_benchmarks.md)。

## Plain 长序列显存控制

四卡入口：`bash scripts/sh/plain/28_plain_4xa800.sh`；先审查配置可运行
`ACTION=plan bash scripts/sh/plain/28_plain_4xa800.sh`。模型、数据及训练参数集中在
`20_plain_train.sh`，外部 `PYTHON`、`CONDA_ENV`、`ACTION` 可覆盖默认值。
默认 ZeRO-2 + CPU optimizer offload；Plain 可用 `DISTRIBUTED_BACKEND=zero3`
启用参数分片及 CPU parameter offload。每卡 batch=1，累积4次时四卡有效 batch=16。

Plain 训练器现在对 Qwen3 / Qwen3.5、LoRA / 全参数训练统一启用以下设置。
参数放在配置顶层（20 脚本的 `training` 下），不是 `beacon` 下：

| 参数 | 默认 | 作用 |
|---|---|---|
| `gradient_checkpointing` | true | 原生 decoder 层重算，使用 non-reentrant 模式 |
| `plain_attention_backend` | sdpa | 显式选择注意力实现；实际融合内核依赖环境 |
| `plain_loss_chunk_size` | 32 | 仅监督位置执行 LM head，每次最多32个 token，可减到16或8 |
| `plain_checkpoint_loss` | true | 反向重算 LM head / CE，避免跨块保留词表 logits |
| `plain_cpu_offload_activations` | true | 用 CPU 保存 autograd 所需张量，可设 false 做对照 |
| `plain_cpu_offload_threshold` | 2048 | 序列 token 数达到阈值时卸载；0表示所有训练样本 |

分块前向在 PEFT 和分布式包装前安装，计算仍通过完整的外层模型 forward，保留
DDP/DeepSpeed 的梯度同步。带 labels 的训练和验证只返回 loss；无 labels 的推理
使用原生 forward。checkpoint 参数名称不变，合并模型不依赖训练时的 forward。
逐 token 分段权重与 labels 同步因果移位；自定义权重沿用每个 microbatch 的
有效权重和归一化，默认 CE 沿用 Trainer 在梯度累积期间的 token 数归一化。

完整轨迹仍一次经过 decoder，不压缩、不截断、不拆分因果上下文。
分块降低输出头峰值，CPU offload 降低保存激活的占用，不能消除当前算子的临时峰值；
ZeRO-2 下完整模型和单条样本的计算仍需放进单卡。CPU offload 增加主机内存和传输成本。
OOM 时每个失败 rank 写 `failure_rank<N>.json` 后退出，不自动丢弃样本或重试。
原 `beacon.beacon_loss_chunk_size` 等字段不会控制 Plain，需使用上表参数。

CPU 回归：`python -m pytest tests/test_plain_memory.py -q`，覆盖 loss / 梯度对齐、
LoRA 梯度累积、验证、保存加载和两进程 Gloo 同步；不代表四卡 A800 的实际显存测量。

Plain checkpoint 的独立 Conda 环境（`search-comp-sglang`）、SGLang 安装、架构兼容检查、单卡/多卡服务启动与联调见 [48 部署手册](48_sglang_deployment.md)。Qwen3.5 纯文本训练产物需要单独验证后端兼容性。
