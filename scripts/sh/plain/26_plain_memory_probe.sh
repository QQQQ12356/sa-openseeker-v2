#!/usr/bin/env bash
# 长样本显存测试；准备好 12 的数据后执行：bash scripts/sh/plain/26_plain_memory_probe.sh
set -euo pipefail

# ===== 运行环境 =====
# 留空时按 common.sh 的规则选择 Python；也可填当前服务器的 Python 绝对路径。
PYTHON="${PYTHON:-}"
CONDA_ENV="${CONDA_ENV:-search-comp-qwen35-cu126}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# 设置可见卡列表，例如 0,1,2,3；NPROC_PER_NODE 默认随可见卡数自动推断。
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-zero2}"  # ddp / zero2；plain 另支持 zero3
# export NPROC_PER_NODE=4
ACTION="${ACTION:-train}"                       # train：生成新实验并训练；plan：只生成配置

source "$(dirname "${BASH_SOURCE[0]}")/../common/00_common.sh"
PY=$(resolve_python)
if [[ "$ACTION" != "train" && "$ACTION" != "plan" ]]; then
  echo "ACTION 必须是 train 或 plan" >&2
  exit 2
fi

# 临时配置会归档到实验目录的 source_config.yaml；结束后删除临时文件。
CONFIG_FILE=$(mktemp "${TMPDIR:-/tmp}/openseeker-train.XXXXXX")
trap 'rm -f -- "$CONFIG_FILE"' EXIT

# ===== 实验参数：直接修改下面的 YAML =====
# 路径相对于项目根目录，也可以使用绝对路径；所有训练与分布式参数写在此脚本。
cat > "$CONFIG_FILE" <<'YAML'
# 可选 qwen3 / qwen35；切换后确认数据按对应 tokenizer 计长。
model_family: qwen35
kind: plain
data_mode: openseeker
# prepared 目录；训练/验证/测试与 manifest 自动从这里读取。
dataset: outputs/data/openseeker/polar_memory_probe
models:
  qwen3: Qwen/Qwen3-4B-Thinking-2507
  qwen35: Qwen/Qwen3.5-4B
# 训练参数：此处为完整配置，不读取外部基础 YAML。
training:
  # 单卡默认关闭 DeepSpeed；可将 null 替换为完整字典。多卡由 distributed 选择。
  deepspeed: null
  max_length: 65536  # 完整轨迹 token 上限；超长过滤或报错，不静默截断
  per_device_batch_size: 1  # 轨迹 collator 限制为 1，使用梯度累积增大 batch
  grad_accum_steps: 1  # 有效 batch = 进程数 × 单卡 batch × 累积步数
  learning_rate: 5.0e-05  # 学习率；修改后重新 plan
  weight_decay: 0.01  # 权重衰减
  lr_scheduler_type: constant  # 学习率调度，如 constant / cosine / linear
  warmup_ratio: 0.0  # Plain 预热比例；范围 0–1
  max_grad_norm: 1.0  # 梯度裁剪阈值
  num_epochs: 1  # max_train_steps 为 null 时按 epoch 训练
  max_train_steps: 5  # null 按 epoch；正整数覆盖 epoch，单位 optimizer step
  use_bf16: true  # 是否启用 BF16，硬件须支持
  optim: adamw_bnb_8bit  # 单卡/DDP 优化器；DeepSpeed 自带 optimizer 时由 distributed 配置决定
  gradient_checkpointing: true  # Plain 层重算；Beacon 不使用此项，使用 beacon_checkpoint_blocks
  plain_attention_backend: sdpa  # Plain 注意力实现；sdpa 或环境支持的其他实现
  plain_loss_chunk_size: 32  # Plain 分块计算监督 token loss 的块大小
  plain_checkpoint_loss: true  # Plain 输出头 loss 重算
  plain_cpu_offload_activations: true  # 用 CPU 内存换 GPU 激活显存
  plain_cpu_offload_threshold: 2048  # 达到该 token 长度才 offload；0 表示全部
  save_freq_steps: 200  # 周期保存间隔，单位 optimizer step
  log_freq_steps: 1  # 兼容字段；logging_steps 已设置时以 logging_steps 为准
  eval_steps: 200  # 验证间隔，单位 optimizer step
  seed: 42  # 训练随机种子
  shuffle_train_data: true  # true 随机打乱；false 按 JSONL 顺序采样
  use_lora: true  # true 训练 LoRA；false 为全参数，资源需求不同
  lora_r: 16  # LoRA 秩
  lora_alpha: 32  # LoRA 缩放参数
  lora_dropout: 0.05  # LoRA dropout
  # LoRA 目标模块；按模型架构检查名称后调整。
  lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
  logging_steps: 1  # 日志间隔，优先于 log_freq_steps
  # 每个 assistant 分段的逐 token 权重；1 表示不调整。
  loss_segments:
    think_tag: 1.0
    think_content: 1.0
    tool_call_tag: 1.0
    tool_call_content: 1.0
    answer_tag: 1.0
    answer_content: 1.0
    other: 1.0
    search_tag: 1.0
    search_content: 1.0
  per_device_eval_batch_size: 1  # 验证 collator 仅支持 1
  save_total_limit: 2  # Plain 周期 checkpoint 最大保留数；null 表示不限制
evaluation:
  backend: replay
  corpus: null
  endpoint: null
  limit: 100
  max_context: 16384
  max_new_tokens: 8192
  max_tool_tokens: 16384
  max_turns: 32
# 多卡 DeepSpeed 参数全部在这里：使用 DISTRIBUTED_BACKEND 选择。
distributed:
  # ZeRO-2，Plain / Beacon 均支持。
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
        weight_decay: auto  # 权重衰减
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
  # ZeRO-3，仅 Plain 支持。
  zero3:
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
        weight_decay: auto  # 权重衰减
    zero_optimization:
      stage: 3
      offload_optimizer:
        device: cpu
        pin_memory: true
      offload_param:
        device: cpu
        pin_memory: true
      overlap_comm: false
      contiguous_gradients: true
      reduce_bucket_size: 10000000
      stage3_prefetch_bucket_size: 10000000
      stage3_param_persistence_threshold: 10000
      stage3_max_live_parameters: 100000000
      stage3_max_reuse_distance: 100000000
      stage3_gather_16bit_weights_on_model_save: true
    zero_allow_untested_optimizer: true
YAML

# 防止误把 Plain 入口改成其他算法或数据协议。
"$PY" - "$CONFIG_FILE" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding='utf-8') as source:
    spec = yaml.safe_load(source)
if spec.get('kind') != 'plain' or spec.get('data_mode') != 'openseeker':
    raise SystemExit('此入口仅支持 kind=plain、data_mode=openseeker；其他路线使用对应脚本')
PY

echo "[openseeker] Python: $PY"
echo "[openseeker] GPU: $CUDA_VISIBLE_DEVICES; action: $ACTION"
# experiment.py 自动创建新实验，保存最终 train.yaml，并记录训练日志和退出码。
# 无需手动 plan 或复制 run_dir。
"$PY" scripts/py/51_experiment.py "$ACTION" --config "$CONFIG_FILE"
