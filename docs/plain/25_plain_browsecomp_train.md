# 25 BrowseComp-Plus SFT Plain 训练与参数说明

本入口是 `scripts/sh/plain/25_plain_browsecomp_train.sh`，使用 BrowseComp-Plus SFT 已处理轨迹训练 Plain。数据先通过 [14 数据预处理](../data/14_browsecomp_sft.md) 独立准备；脚本只校验和读取 prepared 数据、生成训练配置、启动实验，不调用预处理脚本或 `20_plain_train.sh`。模型训练实现沿用现有 Plain 训练器。

## 25.1 最短运行步骤

在项目根目录的 Bash 中执行，使用已有环境和本地模型：

```bash
export PYTHON=/data/miniconda3/envs/search-comp-qwen35-cu126/bin/python
export MODEL_PATH=/data/models/Qwen3.5-4B
export DATASET_DIR=outputs/data/browsecomp-qwen35-sft/prepared_16k
export MAX_LENGTH=16384

# 数据阶段：只需成功运行一次；输出目录必须不存在。
bash scripts/sh/data/14_prepare_browsecomp.sh

# 训练阶段：每条命令独立运行，不再处理数据。
ACTION=plan bash scripts/sh/plain/25_plain_browsecomp_train.sh
ACTION=train bash scripts/sh/plain/25_plain_browsecomp_train.sh
```

路径是示例，请替换为服务器实际路径。已有 prepared 数据时直接从训练阶段开始。默认 `ACTION=plan`，不会训练。`ACTION=train` 每次创建新实验；不是自动复用上次 plan，也不是断点恢复。若要启动已经检查过的 plan，在相同代码、数据和 GPU 设置下执行 `"$PYTHON" scripts/py/51_experiment.py train --run-dir /absolute/path/to/run`，详见 [53](../tools/53_self_contained_training.md)。

## 25.2 在哪里修改参数

1. 顶部“运行环境”：Python、conda 环境、可见卡、分布式后端和 ACTION。
2. “可选训练覆盖”：`DATASET_DIR`、`MODEL_PATH`、`MAX_LENGTH` 默认空值，非空时覆盖 YAML。
3. `cat > "$CONFIG_FILE" <<'YAML'` 到 `YAML` 之间：完整模型、训练、评测和 DeepSpeed 配置，直接修改并保留 YAML 缩进。

三项可选覆盖的优先级高于 YAML。只想直接改 YAML 时，先 `unset DATASET_DIR MODEL_PATH MAX_LENGTH`。修改脚本不会改变已经归档的实验。该入口不接收额外命令行参数，`LEARNING_RATE=...`、`SEED=...` 不会覆盖训练配置；学习率和训练种子应修改 `training.learning_rate`、`training.seed`。

全部 Shell 变量、默认值和错误提示见 [14.4](../data/14_browsecomp_sft.md#144-脚本变量优先级与修改提示)。临时 YAML 通过 `CONFIG_FILE` 保存，退出时清理；不要手动设置 `CONFIG_FILE` 或修改中间文件来调参。

## 25.3 模型、数据和训练参数

| 字段 | 默认 | 含义与修改提示 |
|---|---|---|
| `model_family` | `qwen35` | 选 `models` 中的模型；切换 qwen3 时重新用对应 tokenizer 准备数据 |
| `dataset_name` | `browsecomp-plus` | 实验目录的数据集标识，生成 `qwen35_plain_browsecomp-plus_L...`；与 data_mode 协议独立 |
| `kind` / `data_mode` | `plain` / `openseeker` | 算法及轨迹协议，入口会校验；数据来源为 BrowseComp SFT，协议仍为 OpenSeeker |
| `dataset` | `outputs/data/browsecomp-qwen35-sft/prepared_16k` | 已有 prepared 目录，包含 manifest、train、validation、test 文件 |
| `models.qwen35` / `models.qwen3` | `Qwen/Qwen3.5-4B` / `Qwen/Qwen3-4B-Thinking-2507` | 只使用当前模型族字段；与预处理 tokenizer 的身份字符串一致 |
| `training.max_length` | `16384` | 完整轨迹上限，非压缩后的长度；不得靠此参数重新过滤 prepared 数据 |
| `per_device_batch_size` / `per_device_eval_batch_size` | `1` / `1` | 轨迹 collator 只支持 1，增大有效 batch 用梯度累积 |
| `grad_accum_steps` | `4` | 有效 batch = 实际进程数 × 单卡 batch × 累积步数 |
| `learning_rate` / `weight_decay` | `5e-5` / `0.01` | 学习率和权重衰减，每次修改创建新实验比较 |
| `lr_scheduler_type` / `warmup_ratio` | `constant` / `0.0` | 调度器及预热比例；改 cosine/linear 时结合总训练步数检查 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `num_epochs` / `max_train_steps` | `1` / `null` | 正整数 max_train_steps 覆盖 epoch，单位 optimizer step |
| `use_bf16` | `true` | 使用 BF16，需硬件支持 |
| `optim` | `adamw_bnb_8bit` | 单卡/DDP 优化器；DeepSpeed 选择内联 optimizer 时由它接管 |
| `seed` / `shuffle_train_data` | `42` / `true` | 训练随机种子和打乱开关；与数据预处理 `SEED` 独立 |
| `save_freq_steps` / `eval_steps` | `200` / `200` | 保存和验证间隔，单位 optimizer step；本来源验证集为空，不计算验证 loss |
| `logging_steps` / `log_freq_steps` | `5` / `5` | 日志间隔，logging_steps 优先，后者仅兼容回退 |
| `save_total_limit` | `2` | 周期 checkpoint 保留上限；null 不限制 |
| `use_lora` | `true` | false 为全参数训练，需重新评估显存 |
| `lora_r` / `lora_alpha` / `lora_dropout` | `16` / `32` / `0.05` | LoRA 秩、缩放和 dropout |
| `lora_target_modules` | q/k/v/o 与 gate/up/down 投影 | 按真实模型模块名修改，勿仅凭名称推测新增模块 |

## 25.4 显存和监督参数

| 字段 | 默认 | 含义与修改提示 |
|---|---|---|
| `gradient_checkpointing` | `true` | Transformer 层重算，用计算换显存 |
| `plain_attention_backend` | `sdpa` | 注意力实现；替换前确认模型和环境支持 |
| `plain_loss_chunk_size` | `32` | 监督 token 的输出头 loss 分块；减小通常降低峰值显存、增加调度开销 |
| `plain_checkpoint_loss` | `true` | 输出头 loss 重算 |
| `plain_cpu_offload_activations` | `true` | 将满足阈值的激活转到 CPU；增加主机内存和传输开销 |
| `plain_cpu_offload_threshold` | `2048` | 达到该 token 长度才 offload；0 表示全部 |
| `training.loss_segments` | 所有分段均 `1.0` | assistant 分段逐 token 权重；system/user/observation 不参与 CE |

分段键为 `think_tag`、`think_content`、`tool_call_tag`、`tool_call_content`、`answer_tag`、`answer_content`、`other`、`search_tag`、`search_content`。1.0 保持原权重；调整时检查真实轨迹标签形式和有效监督 token，不能通过设置权重让 observation 成为 assistant 监督。Plain 保留完整序列，不做 Beacon 压缩。

## 25.5 多卡、DeepSpeed 和评测配置

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
DISTRIBUTED_BACKEND=zero2 ACTION=train \
  bash scripts/sh/plain/25_plain_browsecomp_train.sh
```

`DISTRIBUTED_BACKEND=ddp` 使用 DDP；`zero2` / `zero3` 选择本脚本 `distributed` 下完整字典。`training.deepspeed: null` 是单卡默认。ZeRO-2 默认 CPU optimizer offload；ZeRO-3 另加 CPU parameter offload。`auto` 由 launcher/DeepSpeed 根据训练参数解析，通常保留；若修改 bucket、预取、参数保留阈值等字段，参考 [53.4](../tools/53_self_contained_training.md#534-分布式配置直接在脚本修改) 并检查 plan 的启动命令。

`evaluation` 块包含 replay backend、corpus/endpoint、limit、max_context、max_new_tokens、max_tool_tokens、max_turns，默认分别为 replay、null/null、100、16384、8192、16384、32。这是实验中的评测配置，不会让训练脚本自动运行 BrowseComp-Plus benchmark；修改训练 max_length 也不会自动修改这些评测预算。独立评测使用 `scripts/sh/plain/56_plain_browsecomp_plus.sh`，完整参数见 [59](../benchmark/59_llm_judge_matrix.md)。

## 25.6 产物与检查

查看命令打印的 run_dir，其中 `source_config.yaml` 记录覆盖后的完整配置，`train.yaml` 记录解析后的训练配置，`experiment.json` 记录数据和代码指纹。训练日志、退出码、模型和 adapter 按现有 experiment 流程保存。缺 manifest、tokenizer 不匹配、长度不兼容等错误先修正数据/配置再重跑，不修改 manifest 绕过验证。

本来源没有独立金标准 val/test；训练 loss 不能代替 benchmark 成绩。示范回答来源、工具能力适配与数据重叠边界见 [14.5](../data/14_browsecomp_sft.md#145-产物追溯与评测边界)。
