# 50 多卡训练与测试

原始提示词复用与数据格式说明见 [45 统一流程](../common/45_unified_agent_flow.md)；多卡不会改变提示词或工具协议。

四卡 A800 OpenSeeker Beacon 可直接执行（模型与数据先在30中修改）：

```bash
ACTION=plan bash scripts/sh/beacon/38_beacon_4xa800.sh  # 只归档，不训练
bash scripts/sh/beacon/38_beacon_4xa800.sh              # 在目标服务器执行训练
```

38 默认四卡 ZeRO-2，复用 30 的配置，不维护第二份参数。30/37 启用无状态模块重计算 `beacon_checkpoint_blocks` 和 SDPA，默认 whole 且 `keep_window=0`；采用 `beacon_layout_policy=configured` 可在脚本内设置其他受支持布局。完整 OOM 分析及参数说明见 [52 长轨迹 OOM](../beacon/52_beacon_oom.md)。不能把四卡 80 GB 当成单条样本可用 320 GB。

多卡机制保留在 `search_comp/trainer/launch.py`、`utils/distributed.py` 和训练器中。修改主训练 shell 脚本顶部变量后直接执行：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export DISTRIBUTED_BACKEND="zero2"
# export NPROC_PER_NODE=4
```

可见卡多于一张时，launcher 生成 `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=N --module ...`。每卡一个进程，按 rank 绑定设备。当前 launcher 为单机多卡；不把它描述成完整多机调度。

| backend | Plain | Beacon | 行为 |
|---|---|---|---|
| ddp | 支持 | 支持 | 各卡完整模型，分布式梯度同步 |
| zero2 | 支持 | 支持 | 多卡默认，分片梯度/优化器，CPU optimizer offload |
| zero3 | 支持 | 拒绝 | Plain 参数/优化器分片及 CPU offload |

默认训练入口的 ZeRO 配置在 shell 的 `distributed` 段；启动器内联传递所选配置，不读取外部 JSON。旧外部配置入口仍保留兼容。训练器在 DeepSpeed 下交给 Trainer 放置模型，避免 device_map=auto 与每卡进程冲突。只有 rank 0 写共用模型/汇总，显存诊断按 rank 独立文件写入。单卡默认不启用 DeepSpeed；需要时可在脚本的 `training.deepspeed` 内直接填写配置字典。详见 [53](53_self_contained_training.md)。

全局有效 batch = `per_device_batch_size × 进程数 × grad_accum_steps`。显存实验固定每卡 batch=1；增加 GPU 数不保证单条最长样本能装入单卡，尤其 DDP/ZeRO-2 不分片模型参数本体。选择零冗余阶段和 offload 时同时考虑 CPU 内存/传输带宽。

本次保留并运行 CPU/Gloo 相关回归，不进行真实 GPU 多卡训练，实际显存、吞吐、NCCL 和 CPUAdam 能否编译应在目标服务器验证。更详细的已有分布式设计见 [66 多卡细节](66_distributed_details.md)。

## Beacon 多卡测试

`31_beacon_eval.sh` 支持每张 GPU 一个完整模型副本，按样本索引分片并在结束后恢复原始顺序、汇总 EM/F1。这里增加的是题目吞吐，每张卡仍需放下完整 Beacon 模型及单条题目的推理状态；不使用训练的 DDP/ZeRO 或 torchrun。

使用 shell 入口时，修改 `scripts/sh/beacon/31_beacon_eval.sh` 顶部：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_WORKERS=4
RUN_DIR="outputs/experiments/your-beacon-run"
LIMIT=100
DRY_RUN=1
```

设置 `DRY_RUN=1` 执行 `bash scripts/sh/beacon/31_beacon_eval.sh` 审查解析出的模型、数据与命令，确认后设置 `DRY_RUN=0` 再执行。31 的顶部标量支持通过命令前的同名环境变量覆盖。`NUM_WORKERS` 须为正整数，且不超过显式列出的互异 GPU 数；例如可见卡 `2,3,6,7`、workers=4 时，各子进程分别只看到其中一张卡。`LIMIT` 是总测试题数，先取前 N 题，再按索引分给 workers，不是每卡各取 N 题。

需要从命令行传参时，也可直接调用其 Python 入口；以下示例沿用默认 replay 诊断后端，替换模型和数据路径，使用尚不存在的输出文件：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/py/50_evaluate.py \
  --model /models/beacon-merged --kind beacon \
  --data outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl \
  --data-split test --backend replay --num-workers 4 --limit 100 \
  --max-context 16384 --max-new-tokens 8192 \
  --max-tool-tokens 16384 --max-turns 32 \
  --output outputs/results/beacon_4gpu_test_001.jsonl
```

replay 只在生成动作与历史记录匹配时返回观测，用于动作复现诊断。要允许新查询，可选择 `--backend corpus --corpus /path/to/corpus.jsonl`，或 `--backend http --endpoint http://127.0.0.1:8080/tool`，其余多卡参数不变；语料来源和工具服务须与实验目的相符。各后端与指标解释见 [40 评测](../benchmark/40_evaluation.md)。

最终预测旁生成 `.metrics.json`；`<预测文件名去掉.jsonl>.workers/` 保存 `shard_<rank>.jsonl` 和 `worker_<rank>.log`，可定位各卡进度或失败原因。某个 worker 失败时整体报错，不把部分成功的分片当成完整结果。

**56/57 也支持多 worker 测试**，可通过环境变量设置，无需修改脚本顶部：

```bash
ACTION=run CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_WORKERS=4 \
MODEL_PATH=/models/beacon-merged KIND=beacon LIMIT=8 \
OUTPUT_DIR=outputs/results/bcp_beacon_4gpu_001 \
  bash scripts/sh/benchmark/56_browsecomp_plus.sh
```

该命令要求已完成固定语料准备与索引构建。每个 worker 绑定一张 GPU，独立加载完整模型；生成完毕后，父进程按原顺序合并并校验题目和来源，失败则停止其他 workers，不继续评分。`LIMIT=8` 是总题数；全量设 `LIMIT=-1`。裁判仍逐题评分，使用父进程可见卡自动分配权重。完整 all/run/score 示例、worker 日志和显存边界见 [43.7.4](../benchmark/43_offline_benchmarks.md#4374-beacon-多卡测试56-与-31-的能力不同)。本次通过真实子进程与替代生成器验证调度，尚未运行真实 GPU 模型测试。
