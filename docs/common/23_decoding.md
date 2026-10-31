# 23 Plain / Beacon 测试解码与重复惩罚

本指南说明在 OpenSeeker（21/31）、在线 BrowseComp native（47）、离线 BrowseComp-Plus（56）和 HotpotQA（57）中调整 Plain 与 Beacon 的生成方式。原始基座入口 70 使用独立脚本和相同的通用评测器，也支持这些变量。训练、tokenizer、工具逻辑和参考答案不因此改变。

## 23.1 参数与默认值

| 环境变量 | Python 参数 | 默认 | 意义 |
|---|---|---|---|
| DECODING | --decoding | greedy | `greedy` 每步选最高概率 token；`sample` 按概率采样（当前仅 Plain） |
| REPETITION_PENALTY | --repetition-penalty | 1.0 | 有限正数；1 不改变 logits，>1 惩罚已出现 token，0～1 反向鼓励重复 |
| TEMPERATURE | --temperature | 1.0 | 有限正数；仅 sample 生效；较低值使分布更集中 |
| TOP_P | --top-p | 1.0 | `(0,1]`；仅 sample 生效，1 不进行核采样截断；采样时 top-k 显式设为 0 |
| GENERATION_SEED | --generation-seed | 42 | 非负整数且不超过 2147483647；仅 sample 用于生成随机数，与裁判种子分开 |

默认保留原来的贪心、重复惩罚 1.0；不会更改历史结果。`TEMPERATURE`、`TOP_P` 和生成种子在 greedy 下不改变选择，但仍写入配置以便审计。采样模式下，每题使用生成种子和问题文本的 SHA256 派生随机种子，避免 worker 分片或题目顺序改变导致同题沿用其他题的随机序列；同一问题文本得到同一初始随机序列。不同硬件、CUDA 内核、软件和模型权重仍可能产生差异，不能保证逐位复现。

重复惩罚使用与 Transformers 相同的 token 级正负 logits 调整，并不只针对重复句子。Plain 基于完整提示和已生成 token；Beacon 仅针对**未被 Beacon 压缩的输入 token 以及已生成 token**。具体是 system、问题、工具调用和观察包装等 keep 部分，以及后续生成的 token；被压缩的工具观察正文不以原 token 保留，不参与惩罚，Beacon 记忆本身没有可对应的普通词表 ID。若某 token 同时在压缩正文和 keep 部分出现，仍会因 keep 部分而被惩罚。Qwen3.5 线性层即使将 keep 信息融合到循环状态，惩罚范围仍按未压缩的输入 token ID 记录，而非尝试从状态反解 token。增量缓存复用时会持续保留可见 token 集合；开始新历史时重置。

惩罚可能影响 system、问题和可见上下文中已出现的名称、JSON 和工具标记；系数过高可能损害正确引用或工具调用。它是诊断生成策略的变量，不证明训练已修复，也不保证解决复读。建议先用 `1.0` 与 `1.05` 或 `1.1` 比较，再根据轨迹决定；这些数值不是保证最优的配置。

Beacon 为支持增量缓存，会维护可见 token ID 集合；读取 GPU token ID 可能产生设备同步。非默认惩罚每步还要处理该集合对应的 logits，因此可能增加推理耗时。具体开销取决于上下文和设备，需在目标服务器测量。

Plain native 支持贪心、采样与重复惩罚（含 21 直接挂载的 Plain adapter）；Beacon native 支持贪心和重复惩罚，仍不通过本入口暴露采样。47 的 official-v1/v2 使用官方服务端生成策略，不支持本参数覆盖；非默认覆盖会报错，不会默默假装生效。47/56/57 的裁判推理有独立参数，本参数不改变裁判。

## 23.2 OpenSeeker：比较基座、adapter 和合并模型

```bash
RUN_DIR=/absolute/path/to/plain_run
BASE=/models/Qwen3.5-4B
ADAPTER="$RUN_DIR/models/model/final"

# 原基座、相同训练题、相同 replay；先审查。21 的 dry-run 不执行 Python 参数校验或加载模型。
TARGET=openseeker ACTION=run BASE_CHECKPOINT="$BASE" RUN_DIR="$RUN_DIR" \
SPLIT=train LIMIT=8 BACKEND=replay DECODING=greedy REPETITION_PENALTY=1.0 DRY_RUN=1 \
  bash scripts/sh/baseline/70_qwen35_base_eval.sh

# 相同模型/数据，贪心 + 适度重复惩罚；输出单独文件。
RUN_DIR="$RUN_DIR" MODEL_PATH="$ADAPTER" BASE_MODEL_PATH="$BASE" \
SPLIT=train LIMIT=8 BACKEND=replay DECODING=greedy REPETITION_PENALTY=1.1 \
  bash scripts/sh/plain/21_plain_eval.sh

# 核采样作为另一个独立对照：
RUN_DIR="$RUN_DIR" MODEL_PATH="$ADAPTER" BASE_MODEL_PATH="$BASE" \
SPLIT=train LIMIT=8 BACKEND=replay DECODING=sample REPETITION_PENALTY=1.0 \
TEMPERATURE=0.8 TOP_P=0.95 GENERATION_SEED=42 \
  bash scripts/sh/plain/21_plain_eval.sh
```

`RUN_DIR`/`MODEL_PATH`/`BASE_MODEL_PATH`/`TOKENIZER_PATH` 的完整规则见 [22](../baseline/22_qwen35_baseline_and_adapter.md)。replay 只有生成动作与记录完全相同才返回工具观察，不适合作为自由搜索成绩；若研究后续工具循环，统一设置 `BACKEND=corpus CORPUS=/path/to/corpus.jsonl` 或 `BACKEND=http ENDPOINT=...`。建议先比较第一轮 assistant：若基座与 adapter 均在第一次工具调用前复读，工具返回不是该轮的直接触发因素。

## 23.3 在线 BrowseComp 与离线 benchmark

```bash
# 47 native 生成和在线裁判；真实运行调用 API。
RUN_DIR="$RUN_DIR" MODE=native KIND=plain LIMIT=2 \
DECODING=greedy REPETITION_PENALTY=1.1 \
ENV_FILE="$HOME/.config/searchagent/benchmark.env.sh" \
  bash scripts/sh/plain/47_plain_browsecomp.sh

# 56 的待测模型采样；评分 JUDGE_TEMPERATURE、SEED 与生成参数独立。
RUN_DIR="$RUN_DIR" KIND=plain ACTION=run LIMIT=8 \
DECODING=sample TEMPERATURE=0.8 TOP_P=0.95 GENERATION_SEED=42 \
REPETITION_PENALTY=1.05 bash scripts/sh/plain/56_plain_browsecomp_plus.sh

# 57 贪心并使用重复惩罚。
RUN_DIR="$RUN_DIR" KIND=plain ACTION=run LIMIT=8 \
DECODING=greedy REPETITION_PENALTY=1.1 \
  bash scripts/sh/plain/57_plain_hotpotqa.sh

# Beacon 同样可在 31/47/56/57 使用惩罚；需使用已合并 Beacon 模型。
RUN_DIR=/absolute/path/to/beacon_run KIND=beacon SPLIT=test LIMIT=8 \
BACKEND=replay DECODING=greedy REPETITION_PENALTY=1.1 \
  bash scripts/sh/beacon/31_beacon_eval.sh
```

上述 benchmark 在生成前需准备数据、索引、已合并模型和对应服务，步骤见 [21](../plain/21_plain_experiment_guide.md)、[42](../benchmark/42_openseeker_benchmark.md)、[43](../benchmark/43_offline_benchmarks.md)。47 的 MODE=official-v1/v2 不接受非默认覆盖；47/56/57 不支持直接加载 adapter。21 的 LIMIT 为正数，47/56/57 用 -1 表示全量。评分阶段读取已有预测，不能用新解码参数改变已经生成的回答。

使用原始模型时，将相同的参数添加到 `TARGET=... ACTION=run BASE_CHECKPOINT=... bash scripts/sh/baseline/70_qwen35_base_eval.sh`；70 不使用训练后模型。跨模型对比时确保数据、模型 tokenizer、提示、工具、预算和解码一致。

## 23.4 检查输出与比较

每条 native 预测的 `generation_config` 保存 `decoding`、`do_sample`、`repetition_penalty`、`repetition_scope`、`temperature`、`top_p` 和 `generation_seed`；21/31 的指标 JSON 及 47/56/57 的 manifest 保存命令参数。`repetition_scope` 的 `beacon_visible_tokens` 表示上述压缩后可见 token 范围，Plain 的 `full_input_and_generated` 覆盖完整输入和已生成 token。原始基座、adapter、不同解码策略分别写新输出，保留所有题目与失败分母。

先看 `status`、`generated_tokens`、`messages`、`events`：

- `generation_limit`：本轮未在预算内完成，检查是否在单轮 think 中循环，或正常长推理被截断；不要仅通过增大上限解决。
- `turn_limit`：检查是否在多轮重复调用 search/visit。
- `invalid_call` / `tool_error`：检查工具格式与后端；较强惩罚也可能影响工具 JSON。
- `completed`：仍要核验答案质量、工具轨迹和评分；仅结束生成并不代表答对。

比较单一变量：先固定 greedy，比较 penalty 1.0 和 1.05/1.1；再固定其他参数比较 sample。对采样至少记录种子，正式统计需要多个种子和足够题数。先用相同问题的小样本进行诊断，再用独立结果目录扩展到验证/测试集；训练题表现不能代替泛化成绩。

各方案独立 Benchmark 入口、完整变量表、配置优先级与路径迁移见 [46 脚本手册](../benchmark/46_benchmark_scripts.md)。
