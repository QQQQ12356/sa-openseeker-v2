# 69 训练配置参数逐项说明

本文件对应每个方法目录中的 `10_train_openseeker.sh` 和 `11_train_research.sh`。操作和 shell 变量见 [68](68_training_benchmark_guide.md)。修改所选脚本自己的 YAML，不修改已归档实验。路径或 MAX_LENGTH 等显式覆盖变量优先于内联 YAML。

Baseline 与 Plain 共享底层实现，默认区别为 baseline `use_lora: false`、Plain `use_lora: true`，以及 method/数据归档身份。Research 和 OpenSeeker 的模型参数默认一致，数据目录和 dataset_name 不同。

完整路径标明配置所属层级。表中 `auto` 由 Trainer/DeepSpeed 解析，`null` 为未指定；数值单位如 token、optimizer step、元素数在作用栏标明。列表在脚本中使用标准 YAML 缩进。`evaluation.*` 只用于训练内测试，不替代三个 benchmark 的独立 shell 参数。

## 69.1 plain 配置

| 参数路径 | 默认值 | 作用与修改提示 |
|---|---|---|
| `model_family` | `qwen35` | 模型架构族；切换后同步模型、tokenizer 和预处理。 |
| `kind` | `plain` | 底层训练/加载实现；baseline 使用 plain。 |
| `method` | `plain` | 归档实验方法身份；plain / beacon / baseline。 |
| `dataset_name` | `openseeker` | 训练数据身份；入口固定 openseeker 或 research。 |
| `data_mode` | `openseeker` | 轨迹编码协议；两种数据都使用 openseeker。 |
| `dataset` | `outputs/data/openseeker/polar_correct_qwen35_16k` | 包含 JSONL 划分和 manifest 的预处理目录。 |
| `models.qwen3` | `Qwen/Qwen3-4B-Thinking-2507` | Qwen3 基座路径；只有 model_family=qwen3 时选择。 |
| `models.qwen35` | `Qwen/Qwen3.5-4B` | Qwen3.5 基座路径；必须与预处理 tokenizer 登记一致。 |
| `training.deepspeed` | `null` | 兼容显式 DeepSpeed 配置；null 时由分布式启动器按后端选择。 |
| `training.max_length` | `16384` | 完整轨迹 token 上限；超长过滤或报错，不静默截断 |
| `training.per_device_batch_size` | `1` | 轨迹 collator 限制为 1，使用梯度累积增大 batch |
| `training.grad_accum_steps` | `4` | 有效 batch = 进程数 × 单卡 batch × 累积步数 |
| `training.learning_rate` | `5.0e-05` | 学习率；修改后重新 plan |
| `training.weight_decay` | `0.01` | 权重衰减 |
| `training.lr_scheduler_type` | `constant` | 学习率调度，如 constant / cosine / linear |
| `training.warmup_ratio` | `0.0` | Plain 预热比例；范围 0–1 |
| `training.max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `training.num_epochs` | `1` | max_train_steps 为 null 时按 epoch 训练 |
| `training.max_train_steps` | `null` | null 按 epoch；正整数覆盖 epoch，单位 optimizer step |
| `training.use_bf16` | `true` | 是否启用 BF16，硬件须支持 |
| `training.optim` | `adamw_bnb_8bit` | 单卡/DDP 优化器；DeepSpeed 自带 optimizer 时由 distributed 配置决定 |
| `training.gradient_checkpointing` | `true` | Plain 层重算；Beacon 不使用此项，使用 beacon_checkpoint_blocks |
| `training.plain_attention_backend` | `sdpa` | Plain 注意力实现；sdpa 或环境支持的其他实现 |
| `training.plain_loss_chunk_size` | `32` | Plain 分块计算监督 token loss 的块大小 |
| `training.plain_checkpoint_loss` | `true` | Plain 输出头 loss 重算 |
| `training.plain_cpu_offload_activations` | `true` | 用 CPU 内存换 GPU 激活显存 |
| `training.plain_cpu_offload_threshold` | `2048` | 达到该 token 长度才 offload；0 表示全部 |
| `training.save_freq_steps` | `200` | 周期保存间隔，单位 optimizer step |
| `training.log_freq_steps` | `5` | 兼容字段；logging_steps 已设置时以 logging_steps 为准 |
| `training.eval_steps` | `200` | 验证间隔，单位 optimizer step |
| `training.seed` | `42` | 训练随机种子 |
| `training.shuffle_train_data` | `true` | true 随机打乱；false 按 JSONL 顺序采样 |
| `training.use_lora` | `true` | true 训练 LoRA；false 为全参数，资源需求不同 |
| `training.lora_r` | `16` | LoRA 秩 |
| `training.lora_alpha` | `32` | LoRA 缩放参数 |
| `training.lora_dropout` | `0.05` | LoRA dropout |
| `training.lora_target_modules` | `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` | LoRA 目标投影模块列表；baseline 默认不开 LoRA。 |
| `training.logging_steps` | `5` | 日志间隔，优先于 log_freq_steps |
| `training.loss_segments.think_tag` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.loss_segments.think_content` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.loss_segments.tool_call_tag` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.loss_segments.tool_call_content` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.loss_segments.answer_tag` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.loss_segments.answer_content` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.loss_segments.other` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.loss_segments.search_tag` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.loss_segments.search_content` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `training.per_device_eval_batch_size` | `1` | 验证 collator 仅支持 1 |
| `training.save_total_limit` | `2` | Plain 周期 checkpoint 最大保留数；null 表示不限制 |
| `evaluation.backend` | `replay` | 训练内测试的工具后端；replay 是轨迹回放诊断，不是独立 benchmark。 |
| `evaluation.corpus` | `null` | 训练内测试的可选检索 corpus，不影响三个独立 benchmark。 |
| `evaluation.endpoint` | `null` | 训练内测试的可选工具服务地址。 |
| `evaluation.limit` | `100` | 训练内测试题数上限，与独立 benchmark LIMIT 分开。 |
| `evaluation.max_context` | `16384` | 训练内测试上下文 token 预算。 |
| `evaluation.max_new_tokens` | `8192` | 训练内测试每轮输出 token 预算。 |
| `evaluation.max_tool_tokens` | `16384` | 训练内测试 observation token 预算。 |
| `evaluation.max_turns` | `32` | 训练内测试最大交互轮数。 |
| `distributed.zero2.bf16.enabled` | `auto` | BF16 是否启用；auto 与 TrainingArguments 同步。 |
| `distributed.zero2.train_micro_batch_size_per_gpu` | `auto` | 每 GPU 每步 micro batch；auto 与训练 batch 同步。 |
| `distributed.zero2.train_batch_size` | `auto` | 全局有效 batch；auto 由进程数、micro batch 和累积步数计算。 |
| `distributed.zero2.gradient_accumulation_steps` | `auto` | DeepSpeed 梯度累积步数；auto 与训练参数同步。 |
| `distributed.zero2.gradient_clipping` | `auto` | DeepSpeed 梯度裁剪阈值；auto 与 max_grad_norm 同步。 |
| `distributed.zero2.optimizer.type` | `AdamW` | DeepSpeed optimizer 类型，默认 AdamW。 |
| `distributed.zero2.optimizer.params.lr` | `auto` | DeepSpeed optimizer 学习率；auto 与训练参数同步。 |
| `distributed.zero2.optimizer.params.betas` | `auto` | AdamW 一阶/二阶动量系数；auto 与 Trainer 默认同步。 |
| `distributed.zero2.optimizer.params.eps` | `auto` | AdamW 数值稳定常数；auto 与 Trainer 默认同步。 |
| `distributed.zero2.optimizer.params.weight_decay` | `auto` | 权重衰减 |
| `distributed.zero2.zero_optimization.stage` | `2` | ZeRO 分片阶段；2 分片优化器和梯度，3 另分片参数。 |
| `distributed.zero2.zero_optimization.offload_optimizer.device` | `cpu` | 优化器或参数卸载目的设备，cpu。 |
| `distributed.zero2.zero_optimization.offload_optimizer.pin_memory` | `true` | CPU 卸载使用锁页内存。 |
| `distributed.zero2.zero_optimization.overlap_comm` | `false` | 通信和计算是否重叠；默认 false 控制显存峰值。 |
| `distributed.zero2.zero_optimization.contiguous_gradients` | `true` | 梯度是否使用连续缓冲区。 |
| `distributed.zero2.zero_optimization.reduce_bucket_size` | `10000000` | 梯度归约 bucket 元素数，增大会影响显存峰值。 |
| `distributed.zero2.zero_optimization.allgather_bucket_size` | `10000000` | ZeRO-2 all-gather bucket 元素数。 |
| `distributed.zero2.zero_optimization.ignore_unused_parameters` | `true` | ZeRO-2 忽略未参与当前计算的参数。 |
| `distributed.zero2.zero_allow_untested_optimizer` | `true` | 允许当前优化器与 ZeRO 的组合。 |
| `distributed.zero3.bf16.enabled` | `auto` | BF16 是否启用；auto 与 TrainingArguments 同步。 |
| `distributed.zero3.train_micro_batch_size_per_gpu` | `auto` | 每 GPU 每步 micro batch；auto 与训练 batch 同步。 |
| `distributed.zero3.train_batch_size` | `auto` | 全局有效 batch；auto 由进程数、micro batch 和累积步数计算。 |
| `distributed.zero3.gradient_accumulation_steps` | `auto` | DeepSpeed 梯度累积步数；auto 与训练参数同步。 |
| `distributed.zero3.gradient_clipping` | `auto` | DeepSpeed 梯度裁剪阈值；auto 与 max_grad_norm 同步。 |
| `distributed.zero3.optimizer.type` | `AdamW` | DeepSpeed optimizer 类型，默认 AdamW。 |
| `distributed.zero3.optimizer.params.lr` | `auto` | DeepSpeed optimizer 学习率；auto 与训练参数同步。 |
| `distributed.zero3.optimizer.params.betas` | `auto` | AdamW 一阶/二阶动量系数；auto 与 Trainer 默认同步。 |
| `distributed.zero3.optimizer.params.eps` | `auto` | AdamW 数值稳定常数；auto 与 Trainer 默认同步。 |
| `distributed.zero3.optimizer.params.weight_decay` | `auto` | 权重衰减 |
| `distributed.zero3.zero_optimization.stage` | `3` | ZeRO 分片阶段；2 分片优化器和梯度，3 另分片参数。 |
| `distributed.zero3.zero_optimization.offload_optimizer.device` | `cpu` | 优化器或参数卸载目的设备，cpu。 |
| `distributed.zero3.zero_optimization.offload_optimizer.pin_memory` | `true` | CPU 卸载使用锁页内存。 |
| `distributed.zero3.zero_optimization.offload_param.device` | `cpu` | 优化器或参数卸载目的设备，cpu。 |
| `distributed.zero3.zero_optimization.offload_param.pin_memory` | `true` | CPU 卸载使用锁页内存。 |
| `distributed.zero3.zero_optimization.overlap_comm` | `false` | 通信和计算是否重叠；默认 false 控制显存峰值。 |
| `distributed.zero3.zero_optimization.contiguous_gradients` | `true` | 梯度是否使用连续缓冲区。 |
| `distributed.zero3.zero_optimization.reduce_bucket_size` | `10000000` | 梯度归约 bucket 元素数，增大会影响显存峰值。 |
| `distributed.zero3.zero_optimization.stage3_prefetch_bucket_size` | `10000000` | ZeRO-3 参数预取 bucket 元素数。 |
| `distributed.zero3.zero_optimization.stage3_param_persistence_threshold` | `10000` | ZeRO-3 小参数常驻阈值。 |
| `distributed.zero3.zero_optimization.stage3_max_live_parameters` | `100000000` | ZeRO-3 同时驻留 GPU 的参数元素数上限。 |
| `distributed.zero3.zero_optimization.stage3_max_reuse_distance` | `100000000` | ZeRO-3 参数复用距离，影响释放/缓存。 |
| `distributed.zero3.zero_optimization.stage3_gather_16bit_weights_on_model_save` | `true` | ZeRO-3 保存时聚合完整 16-bit 权重。 |
| `distributed.zero3.zero_allow_untested_optimizer` | `true` | 允许当前优化器与 ZeRO 的组合。 |

## 69.2 beacon 配置

| 参数路径 | 默认值 | 作用与修改提示 |
|---|---|---|
| `model_family` | `qwen35` | 模型架构族；切换后同步模型、tokenizer 和预处理。 |
| `kind` | `beacon` | 底层训练/加载实现；baseline 使用 plain。 |
| `method` | `beacon` | 归档实验方法身份；plain / beacon / baseline。 |
| `dataset_name` | `openseeker` | 训练数据身份；入口固定 openseeker 或 research。 |
| `data_mode` | `openseeker` | 轨迹编码协议；两种数据都使用 openseeker。 |
| `dataset` | `outputs/data/openseeker/polar_correct_qwen35_16k` | 包含 JSONL 划分和 manifest 的预处理目录。 |
| `models.qwen3` | `Qwen/Qwen3-4B-Thinking-2507` | Qwen3 基座路径；只有 model_family=qwen3 时选择。 |
| `models.qwen35` | `Qwen/Qwen3.5-4B` | Qwen3.5 基座路径；必须与预处理 tokenizer 登记一致。 |
| `training.deepspeed` | `null` | 兼容显式 DeepSpeed 配置；null 时由分布式启动器按后端选择。 |
| `training.beacon_layout_policy` | `configured` | configured 使用下方布局；whole 强制兼容整段布局 |
| `training.distill_teacher_model_name_or_path` | `null` | 蒸馏 teacher；null 使用 student 基座，启用蒸馏需额外资源 |
| `training.distill_teacher_device` | `auto` | teacher 设备；auto 或显式设备 |
| `training.learning_rate` | `5.0e-05` | 学习率；修改后重新 plan |
| `training.weight_decay` | `0.01` | 权重衰减 |
| `training.optim` | `adamw_bnb_8bit` | 单卡/DDP 优化器；DeepSpeed 自带 optimizer 时由 distributed 配置决定 |
| `training.lr_scheduler_type` | `constant` | 学习率调度，如 constant / cosine / linear |
| `training.warmup_steps` | `0` | Beacon 预热 optimizer 步数 |
| `training.max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `training.use_bf16` | `true` | 是否启用 BF16，硬件须支持 |
| `training.num_epochs` | `1` | max_train_steps 为 null 时按 epoch 训练 |
| `training.max_train_steps` | `null` | null 按 epoch；正整数覆盖 epoch，单位 optimizer step |
| `training.grad_accum_steps` | `4` | 有效 batch = 进程数 × 单卡 batch × 累积步数 |
| `training.logging_steps` | `5` | 日志间隔，优先于 log_freq_steps |
| `training.save_freq_steps` | `200` | 周期保存间隔，单位 optimizer step |
| `training.eval_steps` | `200` | 验证间隔，单位 optimizer step |
| `training.seed` | `42` | 训练随机种子 |
| `training.shuffle_train_data` | `true` | Beacon 是否打乱训练数据 |
| `training.use_lora` | `true` | true 训练 LoRA；false 为全参数，资源需求不同 |
| `training.lora_r` | `16` | LoRA 秩 |
| `training.lora_alpha` | `32` | LoRA 缩放参数 |
| `training.lora_dropout` | `0.05` | LoRA dropout |
| `training.lora_target_modules` | `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` | LoRA 目标投影模块列表；baseline 默认不开 LoRA。 |
| `training.gradient_checkpointing` | `false` | Plain 层重算；Beacon 不使用此项，使用 beacon_checkpoint_blocks |
| `training.length_diagnostics_samples` | `128` | 训练前统计样本数；0 可关闭 |
| `training.max_length` | `16384` | 完整轨迹 token 上限；超长过滤或报错，不静默截断 |
| `training.per_device_batch_size` | `1` | 轨迹 collator 限制为 1，使用梯度累积增大 batch |
| `training.log_freq_steps` | `5` | 兼容字段；logging_steps 已设置时以 logging_steps 为准 |
| `beacon.enable_beacon` | `true` | 是否启用 Beacon 压缩结构。 |
| `beacon.beacon_window` | `1024` | window 布局文档窗口 token 数 |
| `beacon.beacon_stride` | `1024` | 兼容校验字段，当前推进按 window；不得大于 window |
| `beacon.beacon_window_enabled` | `false` | configured/window 模式需 true；whole 使用 false |
| `beacon.beacon_ratio` | `16` | 固定比例及推理比例；越小 Beacon 越多 |
| `beacon.beacon_compression_mode` | `fixed` | fixed 或 random；random 按训练样本采样 |
| `beacon.beacon_compression_ratios` | `[8,16,32,64]` | 随机压缩候选正整数列表；fixed 不采样。 |
| `beacon.beacon_attn` | `full-coverage` | 当前仅支持 full-coverage |
| `beacon.beacon_attention_backend` | `sdpa` | Beacon 注意力实现，默认 sdpa |
| `beacon.beacon_param` | `q k v` | 独立 Beacon 投影，可选 q k v o 的空格组合 |
| `beacon.beacon_embed_init` | `eos` | Beacon token 嵌入初始化来源；默认 eos。 |
| `beacon.beacon_sink_size` | `0` | 兼容字段；当前不使用 |
| `beacon.beacon_attend_prev` | `true` | 兼容字段；当前固定保留历史 Beacon |
| `beacon.beacon_pos` | `append` | append 或 intersect |
| `beacon.eval_beacon_ratio` | `null` | 兼容字段；推理使用 beacon_ratio |
| `beacon.beacon_linear_writer_rank` | `128` | 原生线性注意力 memory writer 的低秩维度。 |
| `beacon.beacon_preserve_native_linear_state` | `false` | 是否保留原生线性注意力状态；属于架构消融选项。 |
| `beacon.beacon_question_memory_v1` | `false` | 是否启用问题记忆分支，默认关闭。 |
| `beacon.beacon_question_max_tokens` | `128` | 问题记忆分支最多保留的问题 token 数。 |
| `beacon.beacon_readout_distill_weight` | `0.0` | readout 蒸馏权重，0 表示关闭 |
| `beacon.beacon_output_distill_weight` | `0.0` | 输出蒸馏权重；>0 加载 teacher |
| `beacon.beacon_output_distill_temperature` | `1.0` | 输出分布蒸馏温度；只在蒸馏启用时生效。 |
| `beacon.beacon_output_distill_strategy` | `all` | 输出蒸馏 token 选择策略；all / prefix / stratified；后两者按每段预算选取 token。 |
| `beacon.beacon_output_distill_tokens` | `8` | 启用 token 选择策略时的 token 数量参数。 |
| `beacon.beacon_output_distill_doc_weighting` | `false` | 是否按文档分布加权输出蒸馏。 |
| `beacon.beacon_output_distill_doc_epsilon` | `0.1` | 文档加权平滑项，避免零权重。 |
| `beacon.beacon_doc_region_split` | `false` | 按子文档拆分；OpenSeeker 不支持 true |
| `beacon.beacon_information_mode` | `whole` | whole / window；policy=configured 时生效 |
| `beacon.beacon_information_loss` | `false` | 开启续写监督需正 continuation 权重 |
| `beacon.beacon_ce_loss_weight` | `1.0` | assistant 交叉熵权重 |
| `beacon.beacon_continuation_loss_weight` | `0.1` | 续写 loss 权重；实际生效还需 information_loss 开启 |
| `beacon.beacon_continuation_tokens` | `4` | 续写监督每处的 token 数；须启用 information_loss。 |
| `beacon.beacon_continuation_per_beacon` | `false` | 是否为每个 Beacon 分别计算续写监督。 |
| `beacon.beacon_keep_window` | `0` | keep 段上限；0 整段；policy=configured 时生效 |
| `beacon.beacon_loss_chunk_size` | `32` | Beacon 监督 loss 分块大小 |
| `beacon.beacon_checkpoint_loss` | `true` | Beacon 输出头 loss 重算 |
| `beacon.beacon_checkpoint_blocks` | `true` | 重算无状态 attention/MLP/writer，不重放状态更新 |
| `beacon.beacon_cpu_offload_activations` | `true` | 用 CPU 内存换 GPU 激活显存 |
| `beacon.beacon_cpu_offload_threshold` | `2048` | 达到该 token 长度才 offload |
| `beacon.loss_segments.think_tag` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `beacon.loss_segments.think_content` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `beacon.loss_segments.tool_call_tag` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `beacon.loss_segments.tool_call_content` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `beacon.loss_segments.answer_tag` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `beacon.loss_segments.answer_content` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `beacon.loss_segments.other` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `beacon.loss_segments.search_tag` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `beacon.loss_segments.search_content` | `1.0` | 该 assistant 片段的逐 token loss 权重；1 不调整，0 不监督；不会给工具 observation 加标签。 |
| `evaluation.backend` | `replay` | 训练内测试的工具后端；replay 是轨迹回放诊断，不是独立 benchmark。 |
| `evaluation.corpus` | `null` | 训练内测试的可选检索 corpus，不影响三个独立 benchmark。 |
| `evaluation.endpoint` | `null` | 训练内测试的可选工具服务地址。 |
| `evaluation.limit` | `100` | 训练内测试题数上限，与独立 benchmark LIMIT 分开。 |
| `evaluation.max_context` | `16384` | 训练内测试上下文 token 预算。 |
| `evaluation.max_new_tokens` | `8192` | 训练内测试每轮输出 token 预算。 |
| `evaluation.max_tool_tokens` | `16384` | 训练内测试 observation token 预算。 |
| `evaluation.max_turns` | `32` | 训练内测试最大交互轮数。 |
| `distributed.zero2.bf16.enabled` | `auto` | BF16 是否启用；auto 与 TrainingArguments 同步。 |
| `distributed.zero2.train_micro_batch_size_per_gpu` | `auto` | 每 GPU 每步 micro batch；auto 与训练 batch 同步。 |
| `distributed.zero2.train_batch_size` | `auto` | 全局有效 batch；auto 由进程数、micro batch 和累积步数计算。 |
| `distributed.zero2.gradient_accumulation_steps` | `auto` | DeepSpeed 梯度累积步数；auto 与训练参数同步。 |
| `distributed.zero2.gradient_clipping` | `auto` | DeepSpeed 梯度裁剪阈值；auto 与 max_grad_norm 同步。 |
| `distributed.zero2.optimizer.type` | `AdamW` | DeepSpeed optimizer 类型，默认 AdamW。 |
| `distributed.zero2.optimizer.params.lr` | `auto` | DeepSpeed optimizer 学习率；auto 与训练参数同步。 |
| `distributed.zero2.optimizer.params.betas` | `auto` | AdamW 一阶/二阶动量系数；auto 与 Trainer 默认同步。 |
| `distributed.zero2.optimizer.params.eps` | `auto` | AdamW 数值稳定常数；auto 与 Trainer 默认同步。 |
| `distributed.zero2.optimizer.params.weight_decay` | `auto` | 权重衰减 |
| `distributed.zero2.zero_optimization.stage` | `2` | ZeRO 分片阶段；2 分片优化器和梯度，3 另分片参数。 |
| `distributed.zero2.zero_optimization.offload_optimizer.device` | `cpu` | 优化器或参数卸载目的设备，cpu。 |
| `distributed.zero2.zero_optimization.offload_optimizer.pin_memory` | `true` | CPU 卸载使用锁页内存。 |
| `distributed.zero2.zero_optimization.overlap_comm` | `false` | 通信和计算是否重叠；默认 false 控制显存峰值。 |
| `distributed.zero2.zero_optimization.contiguous_gradients` | `true` | 梯度是否使用连续缓冲区。 |
| `distributed.zero2.zero_optimization.reduce_bucket_size` | `10000000` | 梯度归约 bucket 元素数，增大会影响显存峰值。 |
| `distributed.zero2.zero_optimization.allgather_bucket_size` | `10000000` | ZeRO-2 all-gather bucket 元素数。 |
| `distributed.zero2.zero_optimization.ignore_unused_parameters` | `true` | ZeRO-2 忽略未参与当前计算的参数。 |
| `distributed.zero2.zero_allow_untested_optimizer` | `true` | 允许当前优化器与 ZeRO 的组合。 |

## 69.3 常用修改与约束

- 有效 batch = 进程数 × per_device_batch_size × grad_accum_steps。完整轨迹 collator 要求单卡 batch 为 1；增加累积步数扩大有效 batch。
- 优先使用 max_train_steps 控制小规模检查；设为 null 时按 num_epochs。当前修改没有实际执行这些检查训练。
- Plain/baseline 的 gradient_checkpointing 与 Beacon 的 beacon_checkpoint_blocks 是不同实现；不要直接把 Plain 开关复制成 Beacon 的重算设置。
- Beacon 开启随机压缩时确保候选非空、正整数、无重复；推理仍使用 beacon_ratio。比较实验需记录实际设置。
- 开启蒸馏需 teacher 权重及额外显存；默认权重为 0。详细架构和蒸馏约束见 architecture/60、62、63 文档。
- 全参数 baseline 不使用 lora_r/alpha/dropout/target_modules；保留这些兼容字段不会启用 LoRA。全参数训练需要相应 GPU 内存和优化器状态资源。
- 改用 model_family=qwen3 时同时修改 MODEL_PATH 和预处理 tokenizer；脚本导出的 MODEL_PATH 默认指向 Qwen3.5，不会随 YAML 的 model_family 自动切换。
- ZeRO-3 仅用于 Plain/baseline；Beacon 使用 ZeRO-2 或 DDP。单卡不会因为脚本默认 backend=zero2 就自动启用多卡训练。
