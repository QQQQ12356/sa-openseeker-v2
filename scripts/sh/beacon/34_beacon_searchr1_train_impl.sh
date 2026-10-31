#!/usr/bin/env bash
# Qwen3 / Qwen3.5 Beacon 训练 —— Search-R1 官方 SFT 轨迹（messages 格式，无需构建数据）。
# 只压缩 <information> 检索文档块；损失只在 assistant 生成片段上计算。
set -euo pipefail
source "$(dirname "$0")/../common/00_common.sh"
# PyTorch 2.7.0+cu126 自带 cuDNN 9.5。若系统里另有旧版 cuDNN 通过
# LD_LIBRARY_PATH 抢先加载，会覆盖 PyTorch 自带版本并导致运行时崩溃。
# 需要屏蔽时导出要剔除的目录（多个用 ':' 分隔），例如：
#   export SHADOWED_LD_PATHS=/opt/cudnn9.0/lib
# 未设置时不改动 LD_LIBRARY_PATH。
if [[ -n ${SHADOWED_LD_PATHS:-} && -n ${LD_LIBRARY_PATH:-} ]]; then
  IFS=: read -r -a _ld_library_paths <<< "$LD_LIBRARY_PATH"
  _kept_ld_library_paths=()
  for _ld_library_path in "${_ld_library_paths[@]}"; do
    if [[ ":${SHADOWED_LD_PATHS}:" == *":${_ld_library_path}:"* ]]; then
      echo "[beacon-searchr1] 已从 LD_LIBRARY_PATH 剔除: $_ld_library_path"
      continue
    fi
    _kept_ld_library_paths+=("$_ld_library_path")
  done
  LD_LIBRARY_PATH=$(IFS=:; printf '%s' "${_kept_ld_library_paths[*]:-}")
  export LD_LIBRARY_PATH
fi
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
model_name_or_path: Qwen/Qwen3.5-4B
distill_teacher_model_name_or_path: null  # 蒸馏 teacher；null 使用 student 基座，启用蒸馏需额外资源
distill_teacher_device: auto  # teacher 设备；auto 或显式设备
data_mode: searchr1
# configured 使用 beacon 中的布局配置；whole 强制兼容整段布局。
beacon_layout_policy: configured  # whole 固定兼容布局；configured 使用 beacon 中的显式布局
train_data_path: outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl
learning_rate: 5.0e-05  # 学习率；修改后重新 plan
weight_decay: 0.01  # 权重衰减
optim: adamw_bnb_8bit  # 单卡/DDP 优化器；DeepSpeed 自带 optimizer 时由 distributed 配置决定
lr_scheduler_type: constant  # 学习率调度，如 constant / cosine / linear
warmup_steps: 0  # Beacon 预热 optimizer 步数
max_grad_norm: 1.0  # 梯度裁剪阈值
use_bf16: true  # 是否启用 BF16，硬件须支持
num_epochs: 1  # max_train_steps 为 null 时按 epoch 训练
max_train_steps: 2000  # null 按 epoch；正整数覆盖 epoch，单位 optimizer step
grad_accum_steps: 4  # 有效 batch = 进程数 × 单卡 batch × 累积步数
logging_steps: 5  # 日志间隔，优先于 log_freq_steps
save_freq_steps: 1000  # 周期保存间隔，单位 optimizer step
eval_steps: 10000  # 验证间隔，单位 optimizer step
val_data_path: null
seed: 42  # 训练随机种子
shuffle_train_data: true  # Beacon 是否打乱训练数据
exp_name: beacon_qwen35_whole_searchr1
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
gradient_checkpointing: false  # Plain 层重算；Beacon 不使用此项，使用 beacon_checkpoint_blocks
length_diagnostics_samples: 1000  # 训练前统计样本数；0 可关闭
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
  beacon_output_distill_weight: 0.1  # 输出蒸馏权重；>0 加载 teacher
  beacon_output_distill_temperature: 1.0
  beacon_output_distill_strategy: all
  beacon_output_distill_tokens: 8
  beacon_output_distill_doc_weighting: false
  beacon_output_distill_doc_epsilon: 0.1
  beacon_doc_region_split: false  # 按子文档拆分；OpenSeeker 不支持 true
  beacon_information_mode: whole  # whole / window；需 policy=configured 才能更改
  beacon_information_loss: false  # 开启续写监督需正 continuation 权重
  beacon_ce_loss_weight: 1.0  # assistant 交叉熵权重
  beacon_continuation_loss_weight: 0.1  # 续写 loss 权重；实际生效还需 information_loss 开启
  beacon_continuation_tokens: 4
  beacon_continuation_per_beacon: false
  beacon_keep_window: 0  # keep 段上限；0 整段；需 policy=configured 才生效
  beacon_loss_chunk_size: 32  # Beacon 监督 loss 分块大小
  beacon_checkpoint_loss: true  # Beacon 输出头 loss 重算
  beacon_checkpoint_blocks: false  # 重算无状态 attention/MLP/writer，不重放状态更新
  beacon_cpu_offload_activations: true  # 用 CPU 内存换 GPU 激活显存
  beacon_cpu_offload_threshold: 2048  # 达到该 token 长度才 offload
  loss_segments:
    think_tag:
      enabled: true
      weight: 1.0
    think_content:
      enabled: true
      weight: 1.0
    search_tag:
      enabled: true
      weight: 1.0
    search_content:
      enabled: true
      weight: 1.0
    answer_tag:
      enabled: true
      weight: 1.0
    answer_content:
      enabled: true
      weight: 1.0
    other:
      enabled: true
      weight: 1.0
    tool_call_tag:
      enabled: true
      weight: 1.0
    tool_call_content:
      enabled: true
      weight: 1.0
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
output_dir: outputs
# 单卡默认关闭 DeepSpeed；可把 null 替换为完整字典。多卡由 distributed 选择。
deepspeed: null
YAML
fi
# 可用环境变量覆盖（未设置则沿用 YAML）；额外 --set 参数优先级最高。
# MODEL_NAME_OR_PATH: Qwen3 默认模型，或 Qwen/Qwen3.5-4B / 本地 checkpoint。
# INFORMATION_MODE: whole=整个 information 一次前向；window=原有分窗。
# BEACON_POS: append=末尾追加全部 beacon；intersect=每 ratio 个原文 token 插入。
# INFORMATION_LOSS: true/false，是否计算 information 续写 CE。
# CONTINUATION_WEIGHT: 续写损失份额，启用时必须 >0。
# DOC_REGION_SPLIT: true/false，whole 必须 false；window 可按子文档拆区。
# WINDOW_ENABLED: false 暂停旧窗口参数（需 whole），true 恢复。
# BEACON_RATIO / BEACON_WINDOW / BEACON_STRIDE: 压缩率 / 历史模式窗口大小。
# 修改后可直接运行；32 入口也会委托本脚本。
COMPRESSION_MODE="${COMPRESSION_MODE:-}"  # 空：沿用 YAML；或 fixed / random
COMPRESSION_RATIOS=()                     # 空：沿用 YAML；例如 (8 16 32 64)
BEACON_OVERRIDES=()
if [[ -n "$COMPRESSION_MODE" ]]; then
  BEACON_OVERRIDES+=(--set "beacon.beacon_compression_mode=$COMPRESSION_MODE")
fi
if [[ ${#COMPRESSION_RATIOS[@]} -gt 0 ]]; then
  COMPRESSION_RATIOS_JSON=$(IFS=,; printf '[%s]' "${COMPRESSION_RATIOS[*]}")
  BEACON_OVERRIDES+=(--set "beacon.beacon_compression_ratios=$COMPRESSION_RATIOS_JSON")
fi
# SHUFFLE_TRAIN_DATA=true 随机打乱，false 按 JSONL 原始顺序。
for pair in 'MODEL_NAME_OR_PATH:model_name_or_path' \
            'DISTILL_TEACHER_MODEL_NAME_OR_PATH:distill_teacher_model_name_or_path' \
            'OUTPUT_DISTILL_WEIGHT:beacon.beacon_output_distill_weight' \
            'OUTPUT_DISTILL_STRATEGY:beacon.beacon_output_distill_strategy' \
            'SHUFFLE_TRAIN_DATA:shuffle_train_data' \
            'INFORMATION_MODE:beacon.beacon_information_mode' \
            'BEACON_POS:beacon.beacon_pos' \
            'INFORMATION_LOSS:beacon.beacon_information_loss' \
            'CONTINUATION_WEIGHT:beacon.beacon_continuation_loss_weight' \
            'DOC_REGION_SPLIT:beacon.beacon_doc_region_split' \
            'BEACON_RATIO:beacon.beacon_ratio' \
            'WINDOW_ENABLED:beacon.beacon_window_enabled' \
            'BEACON_WINDOW:beacon.beacon_window' \
            'BEACON_STRIDE:beacon.beacon_stride'; do
  variable=${pair%%:*}
  key=${pair#*:}
  if [[ -n ${!variable:-} ]]; then
    BEACON_OVERRIDES+=(--set "$key=${!variable}")
  fi
done
if [[ "$ACTION" == plan ]]; then
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$PYTHON_BIN" -u -m search_comp.trainer.launch --dry-run \
    --module search_comp.trainer.beacon_trainer --config "$CONFIG" "${BEACON_OVERRIDES[@]}" "$@"
  exit 0
fi
mkdir -p outputs/models
echo "[beacon-searchr1] 训练数据: Search-R1 messages 轨迹（默认 outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl）"
LOG_PATH=${LOG_PATH:-$(new_log_path beacon_searchr1_train)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$PYTHON_BIN" -u -m search_comp.trainer.launch --module search_comp.trainer.beacon_trainer --config "$CONFIG" "${BEACON_OVERRIDES[@]}" "$@"
echo "[beacon-searchr1] 训练完成"
