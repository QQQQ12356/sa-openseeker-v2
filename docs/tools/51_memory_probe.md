# 51 特定长度显存上限实验

当前保留原始 system 与数据兼容说明见 [45 统一流程](../common/45_unified_agent_flow.md)。12/26/37 默认使用 `polar_memory_probe`；原 v3 数据可复用。

本次仅提供数据和执行入口，没有执行显存训练探测。流程：

1. 10 完成全量规范化与长度实测。
2. 修改 `12_select_memory_probe.sh` 的闭区间、EXACT_LENGTH、TOKENIZER、CORRECTNESS、OBSERVATIONS，运行生成数据。默认选 64512—65536 token、Correct、含 observation；SPLIT_MODE=train，全部写入 `polar_memory_probe/train.jsonl`，validation/test 为空。
3. 检查 manifest 中实际样本数和数据 HTML；精确长度可能为零。区间应足够窄，且保持所有条件一致再比较 Plain 与 Beacon。
4. 修改 `26_plain_memory_probe.sh` 或 `37_beacon_memory_probe.sh` 的 GPU、后端、模型、max_length、压缩比和 offload 参数。默认 max_length=65536、batch=1、梯度累积=1、5 个 optimizer steps，不加载 teacher。先改 ACTION=plan 审查，再改 train 执行。
5. 比较实验目录 `models/model/cuda_memory_rank<N>.jsonl`：记录各 rank 每步的当前 allocated/reserved 和峰值 allocated/reserved，单位 bytes。保留 train.yaml、日志、所选数据的 manifest，便于复现。

训练集单独导出时，实验计划允许空 validation 并关闭验证；这些记录仅用于训练显存，不形成正式测试分数。长区间样本按训练器顺序/采样训练，短测只代表实际经过的样本；若需要严格测试单条样本，优先用 EXACT_LENGTH 或缩窄区间，并检查样本数量。不要把 max_length 配置值当作已测样本的真实最大长度。

CUDA 诊断由公共回调完成：开始训练时重置分配器峰值，每步记录，结束再记录。它统计 PyTorch allocator，不包含全部 NCCL/驱动/其他进程显存；OOM 发生在第一步之前可能没有 step 记录，此时保留异常日志并结合 nvidia-smi。模型加载期峰值不在重置后的训练峰值中。通过几步训练不能证明整个数据集或全部压缩配置都可稳定运行。

要更换 32k/64k/128k：同时修改 12 的 MIN/MAX/EXACT、OUTPUT、26/37 的 dataset 与 max_length，输出目录使用新名称，不覆盖历史实验。多卡行为和 backend 限制见 50。

随机压缩训练可在 37 顶部设置 `COMPRESSION_MODE` / `COMPRESSION_RATIOS`。显存上限测试应先 fixed 使用最小候选比例，再测试 random；短步数随机采样不保证覆盖最坏比例。详见 [31 随机压缩](../beacon/31_random_compression.md)。

## 重计算修复的对照方法

37现在显式设置 `beacon_checkpoint_blocks=true`、`beacon_attention_backend=sdpa`、`beacon_keep_window=1024`；顶层gradient_checkpointing不代替blocks。可直接使用 `CUDA_VISIBLE_DEVICES=0,1,2,3 ACTION=plan bash scripts/sh/beacon/37_beacon_memory_probe.sh` 审查。实际probe需另行运行，plan不测显存。

对同一数据和种子，只修改37中的blocks为false/true，各保存独立实验，记录所有rank显存、主机RSS和每步耗时。offload、压缩比例、loss目标均保持一致；先用最小随机候选比例固定测试。若第一步前OOM，保留完整异常栈，回调可能尚无step记录。6000附近失败应先测窄长度区间，不必从默认64k开始。不要将window与whole的结果视作纯性能对照，因为会改变压缩状态提交方式。详见 [52 长轨迹OOM](../beacon/52_beacon_oom.md)。
