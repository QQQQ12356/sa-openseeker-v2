# 50 多卡训练

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

ZeRO 配置保留在 `configs/train/ds_zero2.json`、`ds_zero3.json`。训练器在 DeepSpeed 下交给 Trainer 放置模型，避免 device_map=auto 与每卡进程冲突。只有 rank 0 写共用模型/汇总，显存诊断按 rank 独立文件写入。单卡路径保持普通 Python 启动；若单卡需要 DeepSpeed，应在基础配置中显式设置 deepspeed。

全局有效 batch = `per_device_batch_size × 进程数 × grad_accum_steps`。显存实验固定每卡 batch=1；增加 GPU 数不保证单条最长样本能装入单卡，尤其 DDP/ZeRO-2 不分片模型参数本体。选择零冗余阶段和 offload 时同时考虑 CPU 内存/传输带宽。

本次保留并运行 CPU/Gloo 相关回归，不进行真实 GPU 多卡训练，实际显存、吞吐、NCCL 和 CPUAdam 能否编译应在目标服务器验证。更详细的已有分布式设计见 [66 多卡细节](66_distributed_details.md)。
