# 52 四卡 A800 长轨迹 OOM：原因、修复与参数指南

当前保留训练原始 system、测试提示词来源选择、必要的离线适配及全部入口的保留范围见 [45 统一流程](45_unified_agent_flow.md)。

## 1. 结论与证据边界

适用：单机四卡 A800 80 GB、OpenSeeker、Beacon、每卡一条轨迹；重点是默认 Qwen3.5 路线，同时覆盖 Qwen3。6000 指 tokenizer 后的 token 数，字符数、原始 JSON 长度和 `max_length` 不能替代实际长度。

本次没有目标服务器异常栈、运行时最终配置或显存快照，也没有启动真实训练。因此不能断言“6000 是模型极限”，也不能给出未经测量的修复后 GB 数或保证 16k/64k 一定成功。以下是代码中已确认的显存机制及对应修复；究竟哪一个算子首先 OOM，按第 6 节定位。

1. **四卡不是一块 320 GB 显卡。** `trainer/launch.py` 启动四个进程，每个进程独立处理样本。DDP 每卡完整参数和训练状态；ZeRO-2 分片梯度、优化器状态，当前配置还卸载优化器到 CPU，但单条轨迹的激活和完整参数仍在单卡。最长样本分到任一 rank 都可能使整个作业失败。参考 [DeepSpeed ZeRO 官方说明](https://www.deepspeed.ai/tutorials/zero/)。
2. **whole 压缩发生在原文计算之后。** `BeaconMemory.step()` 给整个检索段追加 Beacon，再前向，最后才丢弃原文 KV。比例 16 并不意味着读入原文时仅用 1/16 显存。主要尺寸是当前段长度 Q、历史未压缩 keep token 与 Beacon 总数 M，以及层数，而不只是轨迹总长度。
3. **注意力仍有临时峰值。** eager 显式生成 `[B,H,Q,M+Q]` 分数和 FP32 softmax；共享因果 mask 也是二维规模。仅举例，H=32、Q=K=6000 的单个 FP32 分数矩阵约 4.29 GiB，尚未计入其他张量；这不是本项目所有模型的实际头数或峰值预测。Qwen3.5 基础配置已有 SDPA，但 Qwen3 基础配置未显式开启；此次两个 OpenSeeker shell 入口统一显式选 SDPA。SDPA 的融合内核是否可用取决于 dtype、硬件和 mask，不能将调用 SDPA 等同于必定使用 FlashAttention。
4. **压缩缓存不等于释放反向图。** Qwen3.5 除 full attention 外还执行 GatedDeltaNet 和逐 Beacon 循环 writer。最终循环状态虽小，求梯度所需的中间状态会随长度、层数及分段数增加；未启用融合线性算子时还可能使用更昂贵的回退路径。MLP 也产生较宽的中间激活。
5. **旧的 checkpoint 开关实际无效。** `beacon_trainer.py` 对顶层 `gradient_checkpointing=true` 仅告警，不执行层重计算；`beacon_checkpoint_loss=true` 仅重算 LM head / loss 分块，不覆盖 attention、MLP 和 writer。
6. **CPU activation offload 不是所有 GPU 内存的卸载。** 现有 `save_on_cpu` 作用于 autograd 保存的张量；当前算子临时结果、显式持有的缓存、参数和通信缓冲仍占 GPU。反向还需重新加载保存张量。四个 rank 同时卸载也增加主机内存、锁页内存及 PCIe 压力。
7. **蒸馏是额外成本。** OpenSeeker 30 默认关闭输出与 readout 蒸馏，而 Search-R1 基础 YAML 可能开启。teacher 在多卡 auto 下每 rank 一份 CPU 模型，不受 ZeRO 分片；all 输出蒸馏的大词表目标仍可能增加 GPU 峰值。应查看最终配置，不能只看基础 YAML。

## 2. 本次最小修复

新增 `beacon.beacon_checkpoint_blocks`，默认 false，旧 checkpoint 和直接使用基础 YAML 的行为保留。30、37 显式设置 true，38 复用 30。运行中仅在 training 且 grad enabled 时生效，推理不重算。

`BeaconCausalLMMixin._checkpoint_block` 统一调用 PyTorch `checkpoint(..., use_reentrant=False)`，用于 Qwen3/Qwen3.5 attention 和 MLP，以及 Qwen3.5 线性 reader 和 writer。状态以参数显式传入，输出返回后才更新 `BeaconMemory`；不 checkpoint 整个 `_window_hidden`，避免反向重放时重复推进窗口、错误读写最终记忆状态。没有 detach 跨窗记忆，没有截断梯度，也没有把 whole 偷换为 window。

非 reentrant checkpoint 支持嵌套 KV tuple、无梯度输入和冻结 embedding，适合 LoRA；默认保留随机数状态。代价是反向增加计算。与 CPU offload 同用时，不能把所有重算临时结果也视作已卸载，仍需单卡承受最大单块计算。参考 [PyTorch checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint)。

修改位置：`models/beacon_config.py` 定义/验证字段；`models/beacon_common.py` 提供 helper；`models/beacon_qwen3.py` 和 `models/beacon_qwen35.py` 接入无状态计算；`trainer/beacon_trainer.py` 更新旧开关提示。以上路径均相对 `search_comp/`。

## 3. 一键入口与先审查后执行

在项目根目录执行，服务器先准备好训练环境、模型及带 manifest 的数据集：

```bash
# 仅生成实验目录、source_config.yaml 和 train.yaml，不加载训练模型
ACTION=plan bash scripts/sh/38_beacon_4xa800.sh

# 用户在目标服务器自行执行真实训练
bash scripts/sh/38_beacon_4xa800.sh

# Python、GPU 和后端都可外部覆盖
PYTHON=/path/to/env/bin/python CUDA_VISIBLE_DEVICES=4,5,6,7 \
  ACTION=plan bash scripts/sh/38_beacon_4xa800.sh
```

38 只设置四卡、zero2 和 action 的默认值，委托 30；不复制另一份训练 YAML。**模型、数据、长度和全部实验参数只改 `scripts/sh/30_beacon_train.sh`。** 显存探测使用 37，参数独立，不会自动继承你对 30 的修改。30/37 现在保留外部 `PYTHON`、`CONDA_ENV` 和 `ACTION`。每次 plan/train 各建新实验；plan 后再次运行 train 会建另一目录，并非继续之前的 plan。

Search-R1 的 32/34 不读取 30 的嵌入配置。需要同样的重计算可显式传：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sh/32_beacon_searchr1_train.sh \
  --set beacon.beacon_checkpoint_blocks=true \
  --set beacon.beacon_attention_backend=sdpa \
  --set beacon.beacon_keep_window=0
```

## 4. 参数表：在哪里修改、影响什么

| 参数 | 30/38 的推荐起点 | 修改提示与边界 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 38 默认 `0,1,2,3` | 物理卡列表；每 rank 绑定列表中的相对 local rank |
| `NPROC_PER_NODE` | 不设置，自动推断 | 手动设置须匹配可见卡数；旧 shell 中遗留的值会覆盖自动推断 |
| `DISTRIBUTED_BACKEND` | `zero2` | `ddp` 用于对照；Beacon 不支持 zero3，不能靠改 JSON 绕过限制 |
| `PYTHON` / `CONDA_ENV` | 自动解析 / `search-comp-qwen3.5` | 优先可执行的 PYTHON 路径；检查实际日志输出 |
| `ACTION` | `train`，审查用 `plan` | plan 仍需可读取的数据 manifest；不是 GPU 兼容性检测 |
| `model_family` / `models` | qwen35 / Qwen3.5-4B | 与 prepared 数据 tokenizer 一致；切换模型须重做或匹配数据准备 |
| `dataset` | 30 内的数据目录 | 含 train/validation/test 及 manifest；不应只修改文件名假装改变筛选上限 |
| `training.max_length` | 16384 | 数据应提前筛选；不是降低单次算子峰值的自动切窗开关 |
| `per_device_batch_size` | 1 | Beacon 训练限制为 1；有效 batch=卡数×1×累积数 |
| `grad_accum_steps` | 4 | 四卡有效 batch=16；想保持原单卡累积4的有效 batch，四卡改1；减少累积不解决单条前向峰值 |
| `use_lora`, `lora_r`, `lora_alpha` | true,16,32 | 减少可训练参数、梯度及优化器状态；不能消除激活开销 |
| `use_bf16` | true | 不把全部模型或激活无意切回 FP32；mask/部分循环计算仍可能用 FP32 |
| `optim` | 基础为 `adamw_bnb_8bit` | zero2 JSON 自带 AdamW + CPU offload，实际优化器由 DeepSpeed 接管 |
| `gradient_checkpointing` | false / 不设置 | Beacon 中旧顶层字段不启用重算，改用下一项 |
| `beacon_checkpoint_blocks` | true | 新字段，attention/MLP/reader/writer 重算；速度和显存权衡；排查可改 false 对照 |
| `beacon_attention_backend` | sdpa | eager 仅小样本对照，长段显存明显更危险；SDPA 可能回退 |
| `beacon_keep_window` | 0 | 当前 whole SFT 入口强制整段 keep 并在归档前解析；改为正数需单独建立算法/显存对照，不能仅改 YAML 后宣称生效 |
| `beacon_information_mode` | whole | 检索段完整前向；切 window 会改变压缩语义，见第5节 |
| `beacon_window_enabled` | false | whole 时旧 window/stride 不限制检索段大小 |
| `beacon_ratio` | 16 | 改顶部 `BEACON_RATIO`，脚本会覆盖 YAML；更大比例减少 Beacon 数但影响压缩质量，原文读入长度仍在 |
| `beacon_compression_mode` | fixed | 改顶部 `COMPRESSION_MODE`；random 时最小候选比例通常生成最多 Beacon |
| `beacon_compression_ratios` | 8,16,32,64 | 改顶部数组，测试需覆盖最小比例，不能靠几个随机步推断最大显存 |
| `beacon_cpu_offload_activations` | true | 使用主机内存换 GPU 保存张量；四 rank 合计主机内存峰值需实测 |
| `beacon_cpu_offload_threshold` | 基础 YAML 为2048 | token长度阈值；改0所有训练样本开启；注意最终 train.yaml 中继承值 |
| `beacon_loss_chunk_size` | 32 | LM head分块；可改16/8降低词表logits峰值，但不影响reader前向峰值 |
| `beacon_checkpoint_loss` | true | 与 blocks互补，不可相互替代 |
| `beacon_output_distill_weight` / `beacon_readout_distill_weight` | 0 / 0 | >0 增加蒸馏路径；关闭已有蒸馏实验会改变目标，不能作为无差别修复 |
| `beacon_information_loss` | false | 保持默认目标；不要在排查内存时意外打开检索原文监督 |
| `max_train_steps` | null，probe为5 | optimizer步数；每步含多个 microbatch；只跑5步不能验证全数据 |

ZeRO-2 的 `reduce_bucket_size` / `allgather_bucket_size` 当前各为10000000元素，非字节；`overlap_comm=false` 控制额外通信缓冲。已经较保守，不应把通信桶调小作为长轨迹激活 OOM 的主要方案。`pin_memory=true` 用于 CPU optimizer offload，与 autograd activation offload 是两个机制。

## 5. 仍然 OOM 时的处理顺序

1. 先确认运行的是修改后的入口和最终配置：SDPA、blocks=true、keep_window=0、batch=1、BF16、LoRA、CPU activation offload；检查是否意外启用 teacher。确认每个 rank 不在同一物理卡，且没有其他任务占卡。
2. 若异常来自 LM head / CE，减 `beacon_loss_chunk_size`；若 keep 段前向，减 keep_window；若 reader/MLP 的单块临时张量，重计算也不能消除该块本身的空间要求。
3. 若必须保持 whole，先按真实 tokenizer 长度筛选可运行数据、降低单段输入长度或调整数据构造；不要静默截断工具响应和监督标签。更高 Beacon 比例降低后续缓存和 writer 成本，但对超长原文的第一次读取未必有效。
4. 当前 SFT 入口会强制 whole 与 keep_window=0。若研究 window，必须另行修改布局解析并建立独立对照，不能直接依靠以下旧参数覆盖当前入口：`beacon_information_mode: window`、`beacon_window_enabled: true`、`beacon_window: 1024`、`beacon_stride: 1024`、`beacon_keep_window: 1024`，ratio=16可整除窗口；`beacon_doc_region_split: false`、information_loss=false 保持显式设置。Qwen3.5 每窗提交 writer 重建的状态，和 whole 一次读取整段不同，需重新评估质量。小窗只限制当前检索输入，不代表总缓存有固定上限。
5. 若完整参数本身已经接近容量，考虑更小模型或另外实现模型/序列并行。当前 Beacon 不支持 ZeRO-3；不同 rank 的动态分段模块调用次数不同，直接启用可能导致集合通信失配。不要去掉入口保护。
6. 仅 reserved 远高于 allocated 且分配碎片明显时，再在服务器单独试 allocator 设置；碎片优化不能解决活跃张量本身超过80 GB。

## 6. 定位与验收（在目标服务器自行执行）

先保留异常栈最早出现 OOM 的 rank、请求分配字节数、allocated/reserved/free、GPU型号和占用、版本信息，以及实验目录 train.yaml/source_config.yaml/训练日志。NCCL 后续退出经常是某 rank OOM 的结果，不是根因。

按 [51 显存探测](51_memory_probe.md) 选相同长度区间和相同数据；固定随机种子、顺序、压缩比和 loss 目标，比较 blocks=false/true。先在6000附近选窄区间，再逐级增大。确认样本实际参与训练；`max_length=16384` 不意味着已经跑过16k样本。记录最大检索段长度、keep段长度和压缩段数，避免仅比较总token数。

记录每 rank 的 `cuda_memory_rank<N>.jsonl` 中 allocated/reserved峰值、耗时、loss，以及主机RSS和系统内存。allocator统计不包含全部驱动/NCCL；模型加载峰值不属于训练回调重置后的峰值。第一步前 OOM可能没有step记录，必须保留完整stderr。验收应同时满足：指定最长样本能完成前向/反向/更新、所有rank退出正常、loss有限、保存可加载；短测不代表全量数据稳定。

## 7. 本地验证与限制

本次仅执行 CPU 小模型回归和静态检查，不下载生产模型、不启动真实GPU训练。新回归覆盖 Qwen3/Qwen3.5、whole/window、多压缩区与keep分块、冻结embedding，以及CPU activation offload开/关；对比前向loss、所有参数梯度和反向后的记忆状态。现有多卡测试覆盖launcher、SDPA与保存行为。测试命令及结果见 [99 验证](99_validation.md)。

这些验证证明小模型数值行为，没有测量四卡A800的吞吐或显存，也没有验证目标服务器的DeepSpeed、FLA、causal-conv1d和CUDA组合。环境安装与排查见 [01 环境](01_environment.md)、[67 环境细节](67_environment_details.md)。
