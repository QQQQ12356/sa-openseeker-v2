#!/usr/bin/env bash
# Qwen3 / Qwen3.5 纯文本 SFT 训练（标准 Trainer + tqdm 可视化，无 Beacon、不压缩）。
# 数据：Search-R1 官方 messages 轨迹（与 beacon_qwen35_whole_searchr1.yaml 使用
# 同一份文件，但这里不挂任何压缩参数，可直接对比压缩 vs 不压缩）。
# 训练器：search_comp.trainer.plain_sft_trainer
set -euo pipefail
source "$(dirname "$0")/../common/00_common.sh"
PYTHON_BIN=$(resolve_python)

# 默认训练参数和 DeepSpeed 均在下方；可选首个位置参数指定旧外部 YAML。
# ACTION=plan 仅打印启动命令和解析配置，不启动训练。
ACTION="${ACTION:-train}"
if [[ "$ACTION" != train && "$ACTION" != plan ]]; then echo 'ACTION 必须是 train 或 plan' >&2; exit 2; fi
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG=$1
  shift
else
  CONFIG=$(mktemp "${TMPDIR:-/tmp}/searchr1-train.XXXXXX")
  trap 'rm -f -- "$CONFIG"' EXIT
  cat > "$CONFIG" <<'YAML'
# 自包含模式禁止回退到外部 DeepSpeed 配置。
self_contained_training: true
data_mode: searchr1
model_name_or_path: Qwen/Qwen3.5-4B
train_data_path: outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl
val_data_path: null
max_length: 8192  # 完整轨迹 token 上限；超长过滤或报错，不静默截断
per_device_batch_size: 1  # 轨迹 collator 限制为 1，使用梯度累积增大 batch
grad_accum_steps: 8  # 有效 batch = 进程数 × 单卡 batch × 累积步数
learning_rate: 5.0e-05  # 学习率；修改后重新 plan
weight_decay: 0.01  # 权重衰减
lr_scheduler_type: constant  # 学习率调度，如 constant / cosine / linear
warmup_ratio: 0.0  # Plain 预热比例；范围 0–1
max_grad_norm: 1.0  # 梯度裁剪阈值
num_epochs: 1  # max_train_steps 为 null 时按 epoch 训练
max_train_steps: null  # null 按 epoch；正整数覆盖 epoch，单位 optimizer step
use_bf16: true  # 是否启用 BF16，硬件须支持
optim: adamw_bnb_8bit  # 单卡/DDP 优化器；DeepSpeed 自带 optimizer 时由 distributed 配置决定
gradient_checkpointing: true  # Plain 层重算；Beacon 不使用此项，使用 beacon_checkpoint_blocks
plain_attention_backend: sdpa  # Plain 注意力实现；sdpa 或环境支持的其他实现
plain_loss_chunk_size: 32  # Plain 分块计算监督 token loss 的块大小
plain_checkpoint_loss: true  # Plain 输出头 loss 重算
plain_cpu_offload_activations: true  # 用 CPU 内存换 GPU 激活显存
plain_cpu_offload_threshold: 2048  # 达到该 token 长度才 offload；0 表示全部
save_freq_steps: 1000  # 周期保存间隔，单位 optimizer step
log_freq_steps: 5  # 兼容字段；logging_steps 已设置时以 logging_steps 为准
eval_steps: 1000  # 验证间隔，单位 optimizer step
seed: 42  # 训练随机种子
shuffle_train_data: true  # true 随机打乱；false 按 JSONL 顺序采样
use_lora: true  # true 训练 LoRA；false 为全参数，资源需求不同
lora_r: 16  # LoRA 秩
lora_alpha: 32  # LoRA 缩放参数
lora_dropout: 0.05  # LoRA dropout
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj
exp_name: qwen35_plain_sft_v1
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
per_device_eval_batch_size: 1  # 验证 collator 仅支持 1
save_total_limit: 2  # Plain 周期 checkpoint 最大保留数；null 表示不限制
# 逐 token 分段权重：不需要时保持 1。
loss_segments:
  think_tag: 1.0
  think_content: 1.0
  search_tag: 1.0
  search_content: 1.0
  tool_call_tag: 1.0
  tool_call_content: 1.0
  answer_tag: 1.0
  answer_content: 1.0
  other: 1.0
output_dir: outputs
# 单卡默认关闭 DeepSpeed；可把 null 替换为完整字典。多卡由 distributed 选择。
deepspeed: null
YAML
fi
DATA_OVERRIDES=()
if [[ -n ${SEARCHR1_DATA:-} ]]; then
  ENCODED_DATA=$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$SEARCHR1_DATA")
  DATA_OVERRIDES=(--set "train_data_path=$ENCODED_DATA")
fi
SEARCHR1_DATA=$("$PYTHON_BIN" - "$CONFIG" "${DATA_OVERRIDES[@]}" "$@" <<'PY'
import argparse, sys
from search_comp.utils.runtime import load_yaml_config
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--set', action='append', default=[])
args, _ = p.parse_known_args(sys.argv[2:])
print(load_yaml_config(sys.argv[1], args.set)['train_data_path'])
PY
)

echo "[plain-sft] 训练数据: $SEARCHR1_DATA"
if [[ "$ACTION" == train && ! -f "$SEARCHR1_DATA" ]]; then
  cat >&2 <<EOF
[plain-sft] 未找到 $SEARCHR1_DATA（Search-R1 SFT 轨迹约 60MB，不在本仓库内）。
[plain-sft] 先下载：
  mkdir -p outputs/data/searchr1
  huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train \\
      --include '*instruct-sft.jsonl' --local-dir outputs/data/searchr1
[plain-sft] 或复用本地已有文件（避免重复下载）：
  mkdir -p outputs/data/searchr1
  ln -s /path/to/qwen3-4b-instruct-sft.jsonl $SEARCHR1_DATA
EOF
  exit 1
fi

if [[ "$ACTION" == plan ]]; then
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$PYTHON_BIN" -u -m search_comp.trainer.launch --dry-run \
    --module search_comp.trainer.plain_sft_trainer --config "$CONFIG" "${DATA_OVERRIDES[@]}" "$@"
  exit 0
fi
mkdir -p outputs/data outputs/models
echo "[plain-sft] 开始训练（标准 Trainer，可视化到终端）"
LOG_PATH=${LOG_PATH:-$(new_log_path plain_train)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$PYTHON_BIN" -u -m search_comp.trainer.launch --module search_comp.trainer.plain_sft_trainer --config "$CONFIG" "${DATA_OVERRIDES[@]}" "$@"
echo "[plain-sft] 训练完成"
