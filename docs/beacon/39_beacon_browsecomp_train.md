# 39 BrowseComp-Plus SFT Beacon 训练与参数说明

本入口是 `scripts/sh/beacon/39_beacon_browsecomp_train.sh`。数据先由 [14 数据预处理](../data/14_browsecomp_sft.md) 独立准备；39 只读已有 prepared 数据，包含自己的完整训练、Beacon 和分布式配置，直接启动 experiment，不调用 14、25 或 `30_beacon_train.sh`。Plain 与 Beacon 使用同一份未压缩长度过滤结果，Beacon 仅压缩 observation 正文。

## 39.1 分阶段运行

```bash
export PYTHON=/data/miniconda3/envs/search-comp-qwen35-cu126/bin/python
export MODEL_PATH=/data/models/Qwen3.5-4B
export DATASET_DIR=outputs/data/browsecomp-qwen35-sft/prepared_16k
export MAX_LENGTH=16384

# 已有 prepared 数据时跳过此步；目录必须尚不存在。
bash scripts/sh/data/14_prepare_browsecomp.sh

# 默认 plan；需要训练时显式设置 train。
ACTION=plan bash scripts/sh/beacon/39_beacon_browsecomp_train.sh
ACTION=train bash scripts/sh/beacon/39_beacon_browsecomp_train.sh
```

示例绝对路径须替换为本机路径。plan 和 train 各自创建独立实验；train 不自动使用先前 plan。要使用已检查的计划，以相同环境运行 `"$PYTHON" scripts/py/51_experiment.py train --run-dir /absolute/path/to/run`，见 [53](../tools/53_self_contained_training.md)。数据缺失时脚本报错，不自动预处理。

## 39.2 修改位置与变量

顶部设置 Python/conda、GPU、分布式后端、ACTION。`DATASET_DIR`、`MODEL_PATH`、`MAX_LENGTH` 默认留空，非空时才覆盖内联 YAML。全部变量、默认值和修改提示见 [14.4](../data/14_browsecomp_sft.md#144-脚本变量优先级与修改提示)。不传位置参数；数据变量 INPUT/WORKERS/SEED/OFFLINE 不控制训练，训练 seed 修改 YAML 的 `training.seed`。

完整参数位于 `cat > "$CONFIG_FILE" <<'YAML'` 与 `YAML` 之间。环境覆盖优先于 YAML，只编辑 YAML 时可先 `unset DATASET_DIR MODEL_PATH MAX_LENGTH`。修改 30 不会改变 39，修改脚本不会更新已归档实验。`kind: beacon`、`data_mode: openseeker` 为该入口固定路线；BrowseComp SFT 使用现有 OpenSeeker 轨迹协议。

## 39.3 共有训练参数

| 字段 | 默认 | 含义与修改提示 |
|---|---|---|
| `model_family` / `models` | qwen35 / Qwen3.5-4B（另有 qwen3 候选） | 选择模型，tokenizer 身份必须与 prepared manifest 一致；换模型族重新准备数据 |
| `dataset_name` | `browsecomp-plus` | 实验目录的数据集标识，生成 `qwen35_beacon_browsecomp-plus_L...`；不改变 OpenSeeker 轨迹协议 |
| `dataset` | `outputs/data/browsecomp-qwen35-sft/prepared_16k` | 已有 prepared 数据，与 Plain 可共用 |
| `max_length` / `per_device_batch_size` | `16384` / `1` | 完整未压缩轨迹长度；collator 仅支持单轨迹 batch |
| `learning_rate` / `weight_decay` | `5e-5` / `0.01` | 学习率及权重衰减 |
| `optim` / `use_bf16` | `adamw_bnb_8bit` / `true` | 单卡/DDP 优化器与精度；DeepSpeed 内联 optimizer 可接管 |
| `lr_scheduler_type` / `warmup_steps` | `constant` / `0` | Beacon 按 optimizer step 预热，不使用 Plain warmup_ratio |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `num_epochs` / `max_train_steps` | `1` / `null` | 正整数 max_train_steps 覆盖 epoch |
| `grad_accum_steps` | `4` | 有效 batch = 实际进程数 × 1 × 累积步数 |
| `logging_steps` / `log_freq_steps` | `5` / `5` | logging_steps 优先，log_freq_steps 仅兼容回退 |
| `save_freq_steps` / `eval_steps` | `200` / `200` | optimizer step 间隔；空 validation 不计算验证 loss |
| `seed` / `shuffle_train_data` | `42` / `true` | 训练随机种子、打乱开关，与预处理 SEED 独立 |
| `use_lora` / `lora_r` / `lora_alpha` / `lora_dropout` | `true` / `16` / `32` / `0.05` | LoRA 参数；全参数训练需另评估资源 |
| `lora_target_modules` | q/k/v/o、gate/up/down 投影 | 按模型实际模块名设置 |
| `gradient_checkpointing` | `false` | Beacon 使用自身 beacon_checkpoint_blocks，不通过此字段重放状态更新 |
| `length_diagnostics_samples` | `128` | 训练前长度统计样本数；0 关闭 |
| `beacon_layout_policy` | `configured` | 使用下方 Beacon 布局；whole 强制兼容整段布局 |
| `distill_teacher_model_name_or_path` / `distill_teacher_device` | `null` / `auto` | 启用输出蒸馏时使用的 teacher 模型和设备；null 使用 student 基座 |

## 39.4 压缩布局与模型参数（`beacon` 块）

| 字段 | 默认 | 含义与修改提示 |
|---|---|---|
| `enable_beacon` | `true` | 启用 Beacon |
| `beacon_information_mode` / `beacon_window_enabled` | `whole` / `false` | 默认整段 observation 布局；切 window 时同步修改开关、窗口及布局策略，先 plan 校验 |
| `beacon_window` / `beacon_stride` | `1024` / `1024` | 窗口 token 数及兼容 stride；stride 不得大于 window，当前按 window 推进 |
| `beacon_ratio` | `16` | 固定训练及推理压缩比例，越小 Beacon 越多 |
| `beacon_compression_mode` | `fixed` | fixed 固定比例；random 按训练样本采样候选比例 |
| `beacon_compression_ratios` | `[8,16,32,64]` | random 的正整数候选，非空且不重复；推理仍使用 beacon_ratio |
| `beacon_attn` / `beacon_attention_backend` | `full-coverage` / `sdpa` | 当前支持的覆盖形式与注意力实现 |
| `beacon_param` / `beacon_embed_init` | `q k v` / `eos` | 独立投影与初始化；投影可选 q/k/v/o 的空格组合 |
| `beacon_pos` | `append` | append 或 intersect，变更后检查布局 |
| `beacon_sink_size` / `beacon_attend_prev` | `0` / `true` | 兼容字段，当前不使用 sink，固定保留历史 Beacon |
| `eval_beacon_ratio` | `null` | 兼容字段；实际推理使用 beacon_ratio |
| `beacon_keep_window` | `0` | keep 段上限；0 整段，由配置布局策略解析 |
| `beacon_linear_writer_rank` | `128` | 线性 writer 的秩 |
| `beacon_preserve_native_linear_state` | `false` | 原生线性状态保留开关；变更前核对对应模型实现 |
| `beacon_question_memory_v1` / `beacon_question_max_tokens` | `false` / `128` | 问题记忆开关与 token 上限 |
| `beacon_doc_region_split` | `false` | OpenSeeker 协议不支持 true，保持 false |

不要单独改压缩比例后声称允许更长原始轨迹；入库和训练 max_length 都使用压缩前长度。布局由 `configure_sft_layout` 与 `BeaconConfig` 校验，训练前检查归档中的解析结果。

## 39.5 损失、蒸馏与显存参数

| 字段 | 默认 | 含义与修改提示 |
|---|---|---|
| `beacon_ce_loss_weight` | `1.0` | assistant CE 总权重 |
| `loss_segments` | 各项 `1.0` | assistant 分段 token 权重，键及监督边界见 [25.4](../plain/25_plain_browsecomp_train.md#254-显存和监督参数)；Beacon 字段位于 beacon 块 |
| `beacon_information_loss` | `false` | 续写监督开关；启用时需正 continuation 权重 |
| `beacon_continuation_loss_weight` / `beacon_continuation_tokens` / `beacon_continuation_per_beacon` | `0.1` / `4` / `false` | 续写权重、token 数、是否逐 Beacon；默认 information_loss 关闭，因此续写损失未启用 |
| `beacon_readout_distill_weight` | `0.0` | readout 蒸馏权重；0 关闭 |
| `beacon_output_distill_weight` | `0.0` | 输出蒸馏权重；正值需加载 teacher，增加资源需求 |
| `beacon_output_distill_temperature` / `beacon_output_distill_strategy` | `1.0` / `all` | 输出蒸馏温度与策略；使用实现支持的策略值 |
| `beacon_output_distill_tokens` | `8` | 输出蒸馏 token 参数，作用取决于所选策略 |
| `beacon_output_distill_doc_weighting` / `beacon_output_distill_doc_epsilon` | `false` / `0.1` | 文档权重开关和稳定项 |
| `beacon_loss_chunk_size` / `beacon_checkpoint_loss` | `32` / `true` | 输出头 loss 分块和重算 |
| `beacon_checkpoint_blocks` | `true` | 重算无状态 attention/MLP/writer，不重放状态更新 |
| `beacon_cpu_offload_activations` / `beacon_cpu_offload_threshold` | `true` / `2048` | 激活转移到 CPU 的开关和长度阈值，用主机内存与传输换显存 |

## 39.6 分布式、评测和归档

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
DISTRIBUTED_BACKEND=zero2 ACTION=train \
  bash scripts/sh/beacon/39_beacon_browsecomp_train.sh
```

多进程选择 ddp/zero2，Beacon 不支持 ZeRO-3。ZeRO-2 全部参数在本脚本 `distributed.zero2`，默认 CPU optimizer offload；`auto` 由 launcher/DeepSpeed 解析，通常保留。单卡 `training.deepspeed` 默认 null。具体 DeepSpeed 字段和修改原则见 [53.4](../tools/53_self_contained_training.md#534-分布式配置直接在脚本修改)。

`evaluation` 块默认 replay、corpus/endpoint 为 null、limit=100、max_context=16384、max_new_tokens=8192、max_tool_tokens=16384、max_turns=32。它们随实验归档，不会自动运行 BrowseComp-Plus benchmark，也不随训练长度变量自动更新。实际 benchmark 使用 `scripts/sh/beacon/56_beacon_browsecomp_plus.sh`，见 [59](../benchmark/59_llm_judge_matrix.md)。

检查输出 run_dir 内的 `source_config.yaml`、`train.yaml`、`experiment.json`，以及实际训练日志和最终解析配置；模型与 adapter 按现有 experiment 流程保存。空 validation 不作为模型质量结论，独立评测及示范数据重叠边界见 [14.5](../data/14_browsecomp_sft.md#145-产物追溯与评测边界)。
