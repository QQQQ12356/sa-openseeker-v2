#!/usr/bin/env bash
# 修改下方参数后执行：bash scripts/sh/beacon/30_beacon_train.sh
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
kind: beacon
data_mode: openseeker
# prepared 目录；训练/验证/测试与 manifest 自动从这里读取。
dataset: outputs/data/openseeker/polar_correct_qwen35_16k
models:
  qwen3: Qwen/Qwen3-4B-Thinking-2507
  qwen35: Qwen/Qwen3.5-4B
# 训练参数：此处为完整配置，不读取外部基础 YAML。
training:
  # 单卡默认关闭 DeepSpeed；可将 null 替换为完整字典。多卡由 distributed 选择。
  deepspeed: null
  # 默认 configured：使用下方 beacon 布局；whole 强制兼容整段布局。
  beacon_layout_policy: configured  # configured 使用下方布局；whole 强制兼容整段布局
  distill_teacher_model_name_or_path: null  # 蒸馏 teacher；null 使用 student 基座，启用蒸馏需额外资源
  distill_teacher_device: auto  # teacher 设备；auto 或显式设备
  learning_rate: 5.0e-05  # 学习率；修改后重新 plan
  weight_decay: 0.01  # 权重衰减
  optim: adamw_bnb_8bit  # 单卡/DDP 优化器；DeepSpeed 自带 optimizer 时由 distributed 配置决定
  lr_scheduler_type: constant  # 学习率调度，如 constant / cosine / linear
  warmup_steps: 0  # Beacon 预热 optimizer 步数
  max_grad_norm: 1.0  # 梯度裁剪阈值
  use_bf16: true  # 是否启用 BF16，硬件须支持
  num_epochs: 1  # max_train_steps 为 null 时按 epoch 训练
  max_train_steps: null  # null 按 epoch；正整数覆盖 epoch，单位 optimizer step
  grad_accum_steps: 4  # 有效 batch = 进程数 × 单卡 batch × 累积步数
  logging_steps: 5  # 日志间隔，优先于 log_freq_steps
  save_freq_steps: 200  # 周期保存间隔，单位 optimizer step
  eval_steps: 200  # 验证间隔，单位 optimizer step
  seed: 42  # 训练随机种子
  shuffle_train_data: true  # Beacon 是否打乱训练数据
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
  gradient_checkpointing: false  # Plain 层重算；Beacon 不使用此项，使用 beacon_checkpoint_blocks
  length_diagnostics_samples: 128  # 训练前统计样本数；0 可关闭
  max_length: 16384  # 完整轨迹 token 上限；超长过滤或报错，不静默截断
  per_device_batch_size: 1  # 轨迹 collator 限制为 1，使用梯度累积增大 batch
  log_freq_steps: 5  # 兼容字段；logging_steps 已设置时以 logging_steps 为准
# Beacon 算法、损失、内存参数；whole 布局约束见下方说明。
beacon:
  enable_beacon: true
  beacon_window: 1024  # window 布局文档窗口 token 数
  beacon_stride: 1024  # 兼容校验字段，当前推进按 window；不得大于 window
  beacon_window_enabled: false  # configured/window 模式需 true；whole 使用 false
  beacon_ratio: 16  # 固定比例及推理比例；越小 Beacon 越多
  beacon_compression_mode: fixed  # fixed 或 random；random 按训练样本采样
  # random 的非空、无重复正整数候选；固定推理使用 beacon_ratio。
  beacon_compression_ratios:
  - 8
  - 16
  - 32
  - 64
  beacon_attn: full-coverage  # 当前仅支持 full-coverage
  beacon_attention_backend: sdpa  # Beacon 注意力实现，默认 sdpa
  beacon_param: q k v  # 独立 Beacon 投影，可选 q k v o 的空格组合
  beacon_embed_init: eos
  beacon_sink_size: 0  # 兼容字段；当前不使用
  beacon_attend_prev: true  # 兼容字段；当前固定保留历史 Beacon
  beacon_pos: append  # append 或 intersect
  eval_beacon_ratio: null  # 兼容字段；推理使用 beacon_ratio
  beacon_linear_writer_rank: 128
  beacon_preserve_native_linear_state: false
  beacon_question_memory_v1: false
  beacon_question_max_tokens: 128
  beacon_readout_distill_weight: 0.0  # readout 蒸馏权重，0 表示关闭
  beacon_output_distill_weight: 0.0  # 输出蒸馏权重；>0 加载 teacher
  beacon_output_distill_temperature: 1.0
  beacon_output_distill_strategy: all
  beacon_output_distill_tokens: 8
  beacon_output_distill_doc_weighting: false
  beacon_output_distill_doc_epsilon: 0.1
  beacon_doc_region_split: false  # 按子文档拆分；OpenSeeker 不支持 true
  beacon_information_mode: whole  # whole / window；policy=configured 时生效
  beacon_information_loss: false  # 开启续写监督需正 continuation 权重
  beacon_ce_loss_weight: 1.0  # assistant 交叉熵权重
  beacon_continuation_loss_weight: 0.1  # 续写 loss 权重；实际生效还需 information_loss 开启
  beacon_continuation_tokens: 4
  beacon_continuation_per_beacon: false
  beacon_keep_window: 0  # keep 段上限；0 整段；policy=configured 时生效
  beacon_loss_chunk_size: 32  # Beacon 监督 loss 分块大小
  beacon_checkpoint_loss: true  # Beacon 输出头 loss 重算
  beacon_checkpoint_blocks: true  # 重算无状态 attention/MLP/writer，不重放状态更新
  beacon_cpu_offload_activations: true  # 用 CPU 内存换 GPU 激活显存
  beacon_cpu_offload_threshold: 2048  # 达到该 token 长度才 offload
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
YAML

# 在创建实验前校验 Beacon 参数；实际布局策略由训练配置统一解析。
"$PY" - "$CONFIG_FILE" <<'PY'
from pathlib import Path
import sys
import yaml
from search_comp.models.beacon_config import BeaconConfig, configure_sft_layout

spec = yaml.safe_load(Path(sys.argv[1]).read_text())
if spec.get('kind') != 'beacon' or spec.get('data_mode') != 'openseeker':
    raise SystemExit('此入口仅支持 kind=beacon、data_mode=openseeker；其他路线使用对应脚本')
cfg = dict(data_mode=spec['data_mode'], beacon_layout_policy=spec['training'].get('beacon_layout_policy', 'whole'),
           beacon=dict(spec['beacon']))
configure_sft_layout(cfg)
BeaconConfig.from_dict(cfg['beacon'])
PY

echo "[openseeker] Python: $PY"
echo "[openseeker] GPU: $CUDA_VISIBLE_DEVICES; action: $ACTION"
# experiment.py 自动创建新实验，保存最终 train.yaml，并记录训练日志和退出码。
# 无需手动 plan 或复制 run_dir。
"$PY" scripts/py/51_experiment.py "$ACTION" --config "$CONFIG_FILE"
