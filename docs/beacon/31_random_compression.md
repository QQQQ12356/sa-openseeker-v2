# 31 固定与随机压缩比训练

当前保留训练原始 system、测试提示词来源选择、必要的离线适配及全部入口的保留范围见 [45 统一流程](../common/45_unified_agent_flow.md)。

Beacon 支持固定压缩比与随机压缩比，Qwen3/Qwen3.5、OpenSeeker/Search-R1 共用同一实现。默认 `fixed`，保持已有训练行为。

## 一键脚本

修改 `scripts/sh/beacon/30_beacon_train.sh` 顶部变量后直接运行：

```bash
COMPRESSION_MODE="random"          # fixed / random
BEACON_RATIO=16                     # fixed 训练，以及验证/生成的固定压缩比
COMPRESSION_RATIOS=(8 16 32 64)     # random 的候选数组
```

`37_beacon_memory_probe.sh` 有相同变量；`34_beacon_searchr1_train_impl.sh` 也支持这组控制，32 入口会委托它。34 的模式/数组默认为空，沿用 YAML；填写顶部变量后覆盖 YAML，固定比例可以通过既有 `BEACON_RATIO` 覆盖。候选必须是非空、无重复的正整数数组；不是给每个候选配置权重。

30/37 会把顶部变量写入临时 YAML，校验后归档到实验目录的 `source_config.yaml`、`train.yaml` 和 `experiment.json`。`ACTION="plan"` 只校验和生成配置、启动命令，`train` 才启动训练。配置快照用于确认实际设置。

直接编辑配置时使用以下字段：

```yaml
beacon:
  beacon_ratio: 16
  beacon_compression_mode: random
  beacon_compression_ratios: [8, 16, 32, 64]
```

`configs/openseeker.yaml` 及现有 Beacon 基础配置都显式提供这些字段。旧 checkpoint 没有新增字段时默认使用 fixed。

## 采样规则

每条训练轨迹、每次有 labels 的训练前向，从候选数组中等概率独立抽取一个值；允许相邻样本抽到相同值。同一前向的所有 observation 和窗口共用该值。梯度累积包含多个 micro-batch 时，每个 micro-batch 分别抽取；重新访问同一样本会重新抽取。不按单个 token、工具段或 optimizer step 采样。

无压缩区、不带 labels 的前向、`model.eval()` 的验证、prefill 和生成均使用 `beacon_ratio`，不消耗压缩采样随机数。训练时采样不会改写 `beacon_ratio` 或保存的配置，验证/生成可以采用不在候选数组里的固定值。

压缩长度为 `ceil(observation_tokens / 采样比例)`。实际比例同步用于 Beacon 数量、append/intersect 布局和按 Beacon 续写目标，避免出现数量与目标错位。模型参数结构不因比例改变，所以可在同一训练过程中切换。

当前训练器使用 whole 模式，候选不要求整除 observation 长度；尾部不足一个比例单位仍保留一个 Beacon。保留的 window 模式要求所有候选都满足窗口整除及 continuation 布局约束，配置校验会在开始训练前逐个检查。

## 随机性与多卡

采样使用 PyTorch 全局 CPU RNG，受训练的 `seed` 控制；固定随机状态与调用顺序可复现。多卡每个 rank 本地采样，不增加采样同步通信，不要求各卡本次比例不同或相同。已有 DDP unused-parameter 处理和 ZeRO-2 启动方式保持不变。只保存模型/adapter 不包含完整训练随机状态，不能据此保证断点后的随机序列完全一致。

启动布局日志会显示模式、候选和验证/生成的比例。random 模式的训练日志额外提供 `compression_ratio_mean`，表示当前 rank 在该日志窗口内各 micro-batch 所用比例的算术平均值；它不是某次实际抽到的候选，也不是全局多卡平均值。该字段独立于四项损失分量，不参与损失求和，验证前向不计入该窗口。

## 显存测试

比例越小，通常需要保留越多 Beacon。寻找长样本显存边界时先固定最小候选比例测试；短时间随机训练可能没有覆盖最小比例，不能据此推断最坏情况下的显存上限。正式随机训练仍可使用 37 的同组变量；每卡显存日志沿用既有回调。修改比例不改变预处理 token 长度，不需要重新准备数据。

## 验证

`tests/test_beacon_random_compression.py` 覆盖配置校验、种子复现、旧配置兼容、Qwen3/Qwen3.5 的 append/intersect 实际前向与反向等价性、跨观测共享比例、验证/缓存不采样，以及 LoRA 包装后的日志均值。另通过双进程 CPU/Gloo 的 tiny 模型检查不同压缩布局下的梯度同步与保存；不代表已在目标 GPU 服务器实测训练。
