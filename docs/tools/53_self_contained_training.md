# 53 自包含训练脚本：参数、DeepSpeed 和实验计划

Plain 和 Beacon 的训练参数、LoRA、损失、Beacon 配置及 DeepSpeed 选项都写在相应 shell 脚本中。日常修改时编辑脚本，不需要打开或修改 `configs/train/*.yaml`、`ds_zero*.json`。模型权重、tokenizer 与数据仍需事先准备；脚本不能让训练器支持尚未实现的功能。

## 53.1 入口与配置位置

| 目的 | 编辑哪个脚本 | 运行入口 | 说明 |
|---|---|---|---|
| Plain OpenSeeker 正式训练 | `scripts/sh/plain/20_plain_train.sh` | 20；四卡 28 委托 20 | 脚本内 `training`、`distributed` |
| Plain 显存试验 | `scripts/sh/plain/26_plain_memory_probe.sh` | 26 | 配置独立，不继承 20 |
| Beacon OpenSeeker 正式训练 | `scripts/sh/beacon/30_beacon_train.sh` | 30；四卡 38 委托 30 | 脚本内 `training`、`beacon`、`distributed` |
| Beacon 显存试验 | `scripts/sh/beacon/37_beacon_memory_probe.sh` | 37 | 配置独立，不继承 30 |
| Plain Search-R1 | `scripts/sh/plain/22_plain_searchr1_train.sh` | 22 | 脚本内平铺训练配置、`distributed` |
| Beacon Search-R1 | `scripts/sh/beacon/34_beacon_searchr1_train_impl.sh` | 32 或 34 | 32 委托 34；脚本内平铺配置、`beacon`、`distributed` |

20/26/30/37 的 YAML 位于 `cat > "$CONFIG_FILE" <<'YAML'` 和 `YAML` 之间；22/34 位于相同 heredoc 形式的临时配置中。直接修改脚本中的 YAML，保留缩进。Python/conda、GPU 和 `ACTION` 在脚本顶部或环境变量中设置。20/30 与同模型的 28/38 使用同一份参数；26/37 各有自己的配置，需分别修改。

22/34 仍兼容以首个位置参数传旧外部 YAML，用于旧实验；**不传位置参数**才使用脚本内自包含配置。旧的独立 Python/config 入口也保留兼容性；本文“不依赖外部 config”适用于上表的默认 shell 入口。

## 53.2 OpenSeeker：先 plan 再训练

```bash
export CONDA_ENV=search-comp-qwen35-cu126
export PYTHON="$(command -v python)"
# 编辑 20 或 30 的 dataset、model_family、models、training；Beacon 再检查 beacon。
CUDA_VISIBLE_DEVICES=0 ACTION=plan bash scripts/sh/plain/20_plain_train.sh
CUDA_VISIBLE_DEVICES=0 ACTION=plan bash scripts/sh/beacon/30_beacon_train.sh

# 四卡 plan 示例；NPROC 必须与可见 GPU 数匹配。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 DISTRIBUTED_BACKEND=zero2 ACTION=plan \
  bash scripts/sh/plain/28_plain_4xa800.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 DISTRIBUTED_BACKEND=zero2 ACTION=plan \
  bash scripts/sh/beacon/38_beacon_4xa800.sh
```

plan 不加载权重、不开启训练，创建新的实验目录并打印 `run_dir`、`train_command` 和 `next`。检查：

- `source_config.yaml`：脚本中完整模型、训练、Beacon、`distributed` 和评测配置的快照。
- `train.yaml`：数据路径、训练参数和实际 Beacon 布局。多卡 DeepSpeed 配置作为启动命令的 `--set deepspeed={...}` 内联传递，不依赖 `configs/train/ds_zero*.json`。
- `experiment.json`：原始配置、数据与代码指纹；包含内联 `distributed` 映射。
- plan 的 `train_command`：确定进程数和所选 DeepSpeed 字段。训练结束或失败后，以 `models/model/resolved_config.json` 和日志确认最终生效值。

使用刚才的 plan 训练：

```bash
RUN_DIR=/absolute/path/from/plan
# 使用与 plan 相同的 GPU、进程数和 DISTRIBUTED_BACKEND；不是中断恢复。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 DISTRIBUTED_BACKEND=zero2 \
  "$PYTHON" scripts/py/51_experiment.py train --run-dir "$RUN_DIR"
```

`train --run-dir` 从已审查的配置开始；模型目录已有产物时拒绝覆盖。代码/数据改变后重新 plan。修改 shell 后不会改变已存档的 run；要使用新设置必须重新 plan。直接运行 20/28/30/38 会创建**新的** run 并训练，不是继续此前计划。

`dataset` 必须是现有 prepared 目录，包含 manifest 及 split；`models` 中所选模型的身份字符串必须与 manifest tokenizer 键一致。数据过滤和计长见 [10](../data/10_data.md)；环境见 [01](../common/01_environment.md)。

## 53.3 训练参数修改

| `training` 参数 | 典型默认 | 修改说明 |
|---|---|---|
| `max_length` | 16384 | 完整轨迹上限；不得小于 prepared 上限，训练不自动截断 |
| `per_device_batch_size` | 1 | 当前轨迹 collator 只支持 1；其他值在计划/训练时报错 |
| `grad_accum_steps` | 4 | 有效 batch = 实际进程数 × 1 × 梯度累积 |
| `learning_rate` / `weight_decay` | 5e-5 / 0.01 | 学习率和权重衰减；在独立 run 比较 |
| `lr_scheduler_type` | constant | 如 constant、cosine、linear；要符合安装的 Transformers |
| `warmup_ratio` / `warmup_steps` | Plain 0 / Beacon 0 | Plain 使用 ratio；Beacon 使用 steps，不要混用 |
| `num_epochs` / `max_train_steps` | 1 / null | 正整数 max_train_steps 覆盖 epoch，按 optimizer steps |
| `use_bf16` / `optim` | true / adamw_bnb_8bit | 单卡/DDP 的精度和优化器；DeepSpeed optimizer 见 53.4 |
| `use_lora` / `lora_r` / `lora_alpha` / `lora_dropout` | true / 16 / 32 / 0.05 | LoRA 或全参数训练；后者显存要求明显不同 |
| `lora_target_modules` | 注意力及 MLP 投影 | 显式列表；加入 Qwen3.5 线性层模块前先确认实际架构 |
| `save_freq_steps` / `eval_steps` / `logging_steps` | 200 / 200 / 5 | optimizer step 间隔；`log_freq_steps` 是兼容回退 |
| `per_device_eval_batch_size` | Plain 1；Beacon 固定 1 | collator 限制，不能通过配置设大 |
| `save_total_limit` | Plain 2 | Plain 的周期 checkpoint 保留数量；Beacon 使用自己的保存回调 |
| `loss_segments` | Plain 每项 1 | `training.loss_segments`，调整 assistant 分段逐 token 权重 |
| `shuffle_train_data` | true | Plain 和 Beacon 均可选随机或顺序采样 |
| `seed` | 42 | 训练随机种子；数据准备 seed 独立记录 |

Plain 内存参数 `gradient_checkpointing`、`plain_attention_backend`、`plain_loss_chunk_size`、`plain_checkpoint_loss`、`plain_cpu_offload_activations`、`plain_cpu_offload_threshold` 均在脚本内，详见 [20](../plain/20_plain.md)。Beacon 使用自身的 `beacon_checkpoint_blocks`、loss 分块和 CPU offload；顶层 `gradient_checkpointing` 不能替代它们。

训练器内部仍固定协议性约束（例如单轨迹 collator、label mask、Beacon 动态模型不支持 ZeRO-3），不能仅添加任意 Transformers 字段就宣称它会被读取。脚本中每个字段的注释、训练器实际读取以及 plan 的最终结果共同决定行为；需要新训练算法或未支持参数时仍要改代码。

## 53.4 分布式配置直接在脚本修改

`distributed.zero2` 和 Plain 的 `distributed.zero3` 是完整的 DeepSpeed 字典。各字段写在训练脚本 heredoc 中，例如：

```yaml
distributed:
  zero2:
    bf16:
      enabled: auto
    train_micro_batch_size_per_gpu: auto
    train_batch_size: auto
    gradient_accumulation_steps: auto
    gradient_clipping: auto
    optimizer:
      type: AdamW
      params:
        lr: auto
        betas: auto
        eps: auto
        weight_decay: auto
    zero_optimization:
      stage: 2
      offload_optimizer:
        device: cpu
        pin_memory: true
      overlap_comm: false
      contiguous_gradients: true
      reduce_bucket_size: 10000000
      allgather_bucket_size: 10000000
      ignore_unused_parameters: true
    zero_allow_untested_optimizer: true
```

按实际资源编辑脚本的字典；此处示例与现有默认相同。`auto` 由 DeepSpeed/Trainer 结合训练参数解析；想固定值应同时确保与实际进程数、batch、累积步数及精度一致。`stage` 必须与 `zero2`/`zero3` 名称一致，错误在启动前拒绝。Plain 的 zero3 同样可在脚本修改参数 offload、bucket 和保存设置；Beacon 禁止 zero3。选择多进程 `ddp` 不使用 `distributed` 中的 DeepSpeed 字典，且 `training.deepspeed` 必须为 null，否则会报冲突。

| 运行变量 | 默认/行为 |
|---|---|
| CUDA_VISIBLE_DEVICES | 20/26/30/37 默认 0；28/38 默认 0,1,2,3 |
| NPROC_PER_NODE | 留空按可见 GPU 数推断；旧环境中已有值会覆盖推断 |
| DISTRIBUTED_BACKEND | 多进程默认 zero2；可选 ddp、zero2，Plain 可选 zero3 |
| ACTION | OpenSeeker shell 默认 train；plan 只创建配置及命令 |

单进程默认不会因 `DISTRIBUTED_BACKEND=zero2` 自动启用 DeepSpeed。多进程 zero2/zero3 下，脚本内字典通过命令行内联为 `deepspeed`；DeepSpeed 配置含 optimizer 时，`setup_training` 使用其 AdamW 和 CPU offload，覆盖单卡的 `optim=adamw_bnb_8bit`。最终以 `resolved_config.json` 和运行日志为准。用户自选外部 YAML 的遗留入口若没有 `distributed`，仍按旧逻辑读取 `configs/train/ds_zero*.json`；这不影响上表默认自包含入口。

## 53.5 Beacon 压缩与布局

30/37 的所有 Beacon 字段都在 `beacon:` 下，修改压缩比只需修改这一处：

```yaml
beacon:
  beacon_ratio: 16
  beacon_compression_mode: fixed       # fixed / random
  beacon_compression_ratios: [8, 16, 32, 64]
```

random 每条训练样本采样候选，验证和生成使用 `beacon_ratio`。候选须非空、正整数且不重复；详见 [31](../beacon/31_random_compression.md)。30 与 37 各自配置，互不继承。34 的 Search-R1 兼容 shell 环境覆盖也仍可用；其额外 `--set` 优先。

`training.beacon_layout_policy` 默认为 `configured`，使用脚本 `beacon:` 中的实际布局；当前默认仍是 `whole`、窗口关闭和 `beacon_keep_window: 0`，与旧基线相同。设置为 `whole` 可主动采用旧版强制兼容布局，该策略会覆盖模式、窗口、子文档和 keep_window。旧配置未提供 policy 时也沿用 `whole`，以保护已有实验。

如果研究分窗，在新实验中保留 `configured`，同时正确设置 `beacon_information_mode: window`、`beacon_window_enabled: true`、窗口/比例、keep_window 等；`BeaconConfig` 会检查必要约束。当前 `beacon_stride` 是兼容校验字段，窗口推进实际使用 `beacon_window`；`beacon_sink_size`、`beacon_attend_prev`、`eval_beacon_ratio` 等兼容字段并非可自由修改的有效算法开关。OpenSeeker 不支持子文档拆分，`beacon_doc_region_split: true` 会报错；Qwen3 不支持 Qwen3.5 的线性 question memory。分窗和 whole 的记忆行为不同，不能将其视为单纯显存开关，需分别测量质量和性能。

`beacon_output_distill_weight>0` 会加载 teacher，`beacon_information_loss` 和 continuation 权重共同决定续写分支；默认是否实际开启，以完整 `beacon` 配置和门控为准。蒸馏需额外资源，详见 [30](../beacon/30_beacon.md)、[62](../architecture/62_stage1_distillation.md)、[63](../architecture/63_stage2_distillation.md)。

## 53.6 Search-R1 自包含训练

```bash
# 只打印脚本内解析配置和启动命令，不验证数据存在、不启动训练。
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 ACTION=plan \
  bash scripts/sh/plain/22_plain_searchr1_train.sh
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 ACTION=plan \
  bash scripts/sh/beacon/32_beacon_searchr1_train.sh
```

22 的数据路径可直接改 heredoc `train_data_path`，也可设置 `SEARCHR1_DATA=/path/to/file`；命令末尾 `--set train_data_path=...` 的优先级更高。34 的 `model_name_or_path`、`train_data_path` 等同样直接改 heredoc，也支持其列出的环境覆盖与末尾 `--set`。Search-R1 输入是独立协议，不可直接混用 OpenSeeker prepared 数据。

22/32 的 plan 会打印临时配置内容和命令，但退出后临时文件被删除，**不能复制 plan 中的临时路径直接启动**；检查后以相同环境重新运行 `ACTION=train`。实际训练器会在模型输出目录保存解析后的 `resolved_config.json`。`ACTION=train` 会执行实际训练，不应作为 dry-run。

## 53.7 显存试验和检查清单

26/37 的默认数据是约 64K 的显存样本，不比默认 16K 轻。先根据目标长度选择数据、修改对应脚本 dataset/max_length，单独 plan；真实 probe 会训练并更新参数。发生 OOM 不应静默截断，查看 failure 文件、日志、GPU 和主机内存。参考 [51](51_memory_probe.md)、[52](../beacon/52_beacon_oom.md)。

检查配置时确认：所选 tokenizer 身份和最大长度、训练数据与划分、有效 batch、训练步数、LoRA 目标、分段损失、实际 Beacon 布局/压缩比、teacher 门控、DeepSpeed 字典与主机内存。四卡时单条轨迹仍不等于四卡显存简单相加。当前环境可能没有默认 conda 环境或生产模型，必须在目标服务器验证依赖和 GPU；本文没有启动真实训练。
