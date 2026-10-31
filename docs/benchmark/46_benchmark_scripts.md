# 46 各方案独立 Benchmark 脚本与参数手册

本次仅调整 Shell 入口、分发和文档，不修改训练算法、Python 推理流程、数据协议或评分方法。未启动真实训练。OpenSeeker 留出集、在线 BrowseComp、固定语料 BrowseComp-Plus 和 HotpotQA 是不同评测；结果不能混为同一项指标。

## 46.1 入口与目录职责

| 任务 | Plain | Beacon | 原始 Qwen3.5 Baseline |
|---|---|---|---|
| OpenSeeker 留出集 | `plain/21_plain_eval.sh` | `beacon/31_beacon_eval.sh` | `baseline/71_qwen35_openseeker.sh` |
| 在线 BrowseComp | `plain/47_plain_browsecomp.sh` | `beacon/47_beacon_browsecomp.sh` | `baseline/72_qwen35_browsecomp.sh` |
| BrowseComp-Plus | `plain/56_plain_browsecomp_plus.sh` | `beacon/56_beacon_browsecomp_plus.sh` | `baseline/74_qwen35_browsecomp_plus.sh` |
| HotpotQA | `plain/57_plain_hotpotqa.sh` | `beacon/57_beacon_hotpotqa.sh` | `baseline/75_qwen35_hotpotqa.sh` |

表中路径均相对 `scripts/sh/`。脚本有序号前缀；同一 benchmark 在 Plain/Beacon 使用相同编号，并以方案名区分。Plain 和 Beacon 各有完整配置入口，直接调用公共 Python 工作流，不互相转发，也不转发到 benchmark 下的评测入口。Baseline 74/75 固定数据集并调用本目录 73 的原始模型配置与参数映射；73 保留兼容旧命令，70 总入口已转向 74/75。

`benchmark/` 只保留共享内容：46 下载在线基准及官方代码、49 私有 API 配置示例、58 离线参数到 CLI 的映射。Plain 模型服务入口迁到 `plain/48_serve_checkpoint.sh`。已有 Python 47/48/52 的文件名和命令行接口保持不变。

## 46.2 一键运行与阶段

以下命令在仓库根目录执行；Shell 会先定位仓库，因此使用绝对脚本路径也可从其他目录启动。先安装项目依赖并准备本地完整模型。示例中的 `/models/...`、`/runs/...`、`/private/...` 均需替换。

```bash
# 在线数据准备（会联网，单独执行一次）
bash scripts/sh/benchmark/46_prepare_benchmark.sh

# 在线 BrowseComp：选择一个方案执行；ENV_FILE 使用 49 示例填写私有服务配置
RUN_DIR=/runs/plain ENV_FILE=/private/api.env bash scripts/sh/plain/47_plain_browsecomp.sh
RUN_DIR=/runs/beacon ENV_FILE=/private/api.env bash scripts/sh/beacon/47_beacon_browsecomp.sh
BASE_CHECKPOINT=/models/Qwen3.5-4B ENV_FILE=/private/api.env bash scripts/sh/baseline/72_qwen35_browsecomp.sh

# 离线数据：download 唯一允许联网的阶段；各方案共用同一个 ROOT_DIR
ACTION=download bash scripts/sh/plain/57_plain_hotpotqa.sh
ACTION=prepare bash scripts/sh/plain/57_plain_hotpotqa.sh
ACTION=index bash scripts/sh/plain/57_plain_hotpotqa.sh

# 已准备数据和索引后，一条命令推理；HotpotQA 的 run 不自动评分
ACTION=run RUN_DIR=/runs/plain bash scripts/sh/plain/57_plain_hotpotqa.sh
ACTION=run RUN_DIR=/runs/beacon bash scripts/sh/beacon/57_beacon_hotpotqa.sh
BASE_CHECKPOINT=/models/Qwen3.5-4B bash scripts/sh/baseline/75_qwen35_hotpotqa.sh

# 离线全流程（不包含 download）：prepare → index → run → score
ACTION=all RUN_DIR=/runs/plain bash scripts/sh/plain/57_plain_hotpotqa.sh
ACTION=all RUN_DIR=/runs/beacon JUDGE_MODEL=/models/Qwen3-32B \
  bash scripts/sh/beacon/56_beacon_browsecomp_plus.sh
ACTION=all BASE_CHECKPOINT=/models/Qwen3.5-4B JUDGE_MODEL=/models/Qwen3-32B \
  bash scripts/sh/baseline/74_qwen35_browsecomp_plus.sh

# 单独评分：必须指定预测路径；OUTPUT_DIR 使用尚不存在的新目录
ACTION=score PREDICTIONS=/results/predictions.jsonl OUTPUT_DIR=/results/new-score \
  bash scripts/sh/plain/57_plain_hotpotqa.sh

# 只解析配置：不下载数据、不加载模型、不调用工具或裁判、不训练
DRY_RUN=1 ACTION=run RUN_DIR=/runs/beacon bash scripts/sh/beacon/57_beacon_hotpotqa.sh
DRY_RUN=1 bash scripts/sh/plain/47_plain_browsecomp.sh
DRY_RUN=1 bash scripts/sh/baseline/74_qwen35_browsecomp_plus.sh
```

Plain 的 `27_plain_workflow.sh` 支持 `TARGET=browsecomp|browsecomp-plus|hotpotqa`，继续使用 `ACTION` 选择阶段，已改为本目录独立入口。Baseline `70_qwen35_base_eval.sh` 同理。Beacon 直接运行表中的脚本即可。

Plain/Beacon 离线脚本默认 `ACTION=all`，Baseline 默认 `ACTION=run`，与原有行为一致；建议命令中显式写阶段。在线 Plain/Beacon 默认生成并评分；设置 `PREDICTIONS` 后仅评分。Baseline 在线通过 `ACTION=run|score` 区分。BrowseComp-Plus 的 `all/score` 需要本地 `JUDGE_MODEL`；HotpotQA 使用官方 EM/F1，无需裁判。`RETRIEVAL_ONLY=1` 只用于检索诊断，不能报告答案准确率。

## 46.3 配置优先级和修改位置

1. 单次实验优先在命令前写 `变量=值`；长期默认值可修改相应独立脚本顶部 `${VAR:-默认值}` 中的默认值。Baseline 两个离线入口共享的默认值在 73 顶部修改，也可在 74/75 中增加 `export VAR="${VAR:-默认值}"` 为某个数据集单独配置。
2. 在线脚本的 `ENV_FILE` 在顶部默认值赋值之后加载，文件中直接赋值会覆盖已有变量；49 示例采用 `${VAR:-默认值}`，已有非空环境变量优先。密钥留在仓库外，不提交真实配置。
3. 命令末尾支持额外 Python CLI 参数，重复标量参数通常最后一个生效；推荐使用同名 Shell 环境变量，避免两处设置冲突。不要通过 `--kind`、`--mode` 或 `--model-origin` 改变入口对应的实验方案。
4. Plain/Beacon `KIND` 分别固定为 plain/beacon；错误环境变量立即报错。Beacon 仅支持 `MODE=native` 和 greedy，保留压缩逻辑；Plain 支持 native 以及通过已部署模型服务执行 official-v1/v2。官方模式使用上游提示及解码，不接受自定义 native 解码配置。
5. `RUN_DIR` 指训练实验目录，自动定位最终模型及提示；`MODEL_PATH` 可覆盖模型路径，`PROMPT_DATA` 可覆盖提示来源。模型必须是完整、已合并 checkpoint，adapter 先用各方案合并脚本处理。Baseline 使用 `BASE_CHECKPOINT` 和专用原始模型加载器，`RUN_DIR` 仅用于数据/提示。
6. 离线 `ROOT_DIR` 供三组共享，存储原始数据、标准化数据和索引。改变语料、语料子集或 `TITLE_WEIGHT` 时用新 ROOT_DIR 并重新 prepare/index。输出使用新 OUTPUT_DIR；默认目录由原 Python 流程选择（通常在 run/evaluation 或 outputs/results），Baseline 按数据集与时间放到 outputs/results/qwen35_base 下。DRY_RUN 输出可确认实际路径。
7. `LIMIT=-1` 为 benchmark 全量，正数取前 N 题；OpenSeeker 21/31/71 的限制以各自说明为准，不能套用 -1。`CORPUS_LIMIT` 仅用于小规模检查。修改模型、提示、检索和生成预算后，记录参数再比较结果。

## 46.4 脚本变量逐项说明

下表从独立脚本的显式默认值整理。详细协议与数据格式见 [42 在线评测](42_openseeker_benchmark.md)、[43 离线评测](43_offline_benchmarks.md)；API 获取与 URL 格式见 `scripts/sh/benchmark/49_benchmark_env.example.sh`。

### 在线 Plain（Beacon 除 KIND 外相同，且限定 native）

配置源：`scripts/sh/plain/47_plain_browsecomp.sh`。

| 变量 | 默认值 | 说明与修改提示 |
|---|---|---|
| `PYTHON` | `空` | Python 可执行文件的绝对路径；留空自动寻找环境。 |
| `CONDA_ENV` | `search-comp-qwen35-cu126` | 自动寻找的 Conda 环境名；PYTHON 显式可执行路径优先。 |
| `MODE` | `native` | native 本地加载；official-v1/v2 请求已部署 Plain 服务。 |
| `RUN_DIR` | `空` | 训练实验目录；Baseline 仅选择提示/数据。 |
| `MODEL_PATH` | `空` | 完整本地模型路径，覆盖 RUN_DIR 自动选择。 |
| `PROMPT_DATA` | `空` | 原训练 JSONL 的 system/schema 提示来源；不使用评测答案构造提示。 |
| `KIND` | `plain` | 固定模型方案，Plain=plain、Beacon=beacon。 |
| `DATA` | `outputs/data/benchmarks/browsecomp.jsonl` | 问题数据；留空使用该基准默认原始文件。 |
| `BENCHMARK` | `BrowseComp` | 在线裁判使用的数据集名称。 |
| `OUTPUT_DIR` | `空` | 本次结果的新目录；留空自动选择，禁止覆盖旧结果。 |
| `PREDICTIONS` | `空` | 已有预测 JSONL；离线 score 阶段必须设置。 |
| `WORKERS` | `60` | 官方在线生成并发；native 不按题并行。 |
| `JUDGE_WORKERS` | `20` | 在线裁判并发。 |
| `LIMIT` | `-1` | -1 全量；正整数取前 N 题。 |
| `MAX_TOKENS` | `32768` | 在线单次生成预算。 |
| `TOOL_COUNT` | `200` | 工具调用预算；在线 native 循环由 MAX_TURNS 控制。 |
| `MAX_CONTEXT` | `131072` | 逻辑历史 token 预算；不同于 Beacon 压缩后的 KV 长度。 |
| `MAX_TURNS` | `200` | agent 循环最大轮数。 |
| `MAX_TOOL_TOKENS` | `32768` | 工具返回 token 总预算。 |
| `DECODING` | `greedy` | greedy 或 sample；Beacon 仅 greedy。 |
| `REPETITION_PENALTY` | `1.0` | 重复惩罚；1.0 不惩罚，须为正数。 |
| `TEMPERATURE` | `1.0` | sample 生成温度；greedy 不使用。 |
| `TOP_P` | `1.0` | sample 核采样概率阈值。 |
| `GENERATION_SEED` | `42` | 生成随机种子，与裁判 SEED 独立。 |
| `DRY_RUN` | `0` | 0 实际执行；1 只解析并打印配置。 |
| `ENV_FILE` | `空` | 仓库外私有 Shell API 配置文件。 |

### 离线 BrowseComp-Plus（Plain/Beacon 共用同名变量）

配置源：`scripts/sh/plain/56_plain_browsecomp_plus.sh`。

| 变量 | 默认值 | 说明与修改提示 |
|---|---|---|
| `ACTION` | `all` | download/prepare/index/run/score/all；all 不包含 download。 |
| `ROOT_DIR` | `outputs/data/benchmarks/browsecomp-plus` | 该数据集共享原始数据、预处理数据和索引目录。 |
| `PYTHON` | `空` | Python 可执行文件的绝对路径；留空自动寻找环境。 |
| `CONDA_ENV` | `search-comp-qwen35-cu126` | 自动寻找的 Conda 环境名；PYTHON 显式可执行路径优先。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可见 GPU 编号列表，例如 0,1；只影响实际运行。 |
| `NUM_WORKERS` | `1` | 每卡一个完整模型副本，按题并行，不超过可见 GPU 数。 |
| `RUN_DIR` | `空` | 训练实验目录；Baseline 仅选择提示/数据。 |
| `MODEL_PATH` | `空` | 完整本地模型路径，覆盖 RUN_DIR 自动选择。 |
| `PROMPT_DATA` | `空` | 原训练 JSONL 的 system/schema 提示来源；不使用评测答案构造提示。 |
| `KIND` | `plain` | 固定模型方案，Plain=plain、Beacon=beacon。 |
| `DATA` | `空` | 问题数据；留空使用该基准默认原始文件。 |
| `CORPUS` | `空` | 独立完整语料路径；留空使用默认。 |
| `DECRYPTED` | `0` | 1 表示提供已解密的 BrowseComp-Plus 官方问题 JSONL。 |
| `CORPUS_LIMIT` | `空` | 留空全语料；正数限制语料，仅用于小规模检查。 |
| `OFFICIAL_ONLY` | `0` | download 阶段 1 只安装代码自带的官方参考文件。 |
| `TITLE_WEIGHT` | `1.0` | FTS5 BM25 标题权重；改动需换目录重建索引。 |
| `VERIFY_INDEX` | `0` | 1 加载时额外验证整个索引 SHA256。 |
| `TOPK` | `5` | 每次搜索返回文档数量。 |
| `SNIPPET_TOKENS` | `512` | 每条搜索结果正文 token 上限，使用待测模型 tokenizer。 |
| `ALLOW_VISIT` | `0` | 1 允许读取本地全文；BCP 默认 0，Hotpot 默认 1。 |
| `READ_TOKENS` | `4096` | 一次读取本地文档的 token 上限。 |
| `TOOL_COUNT` | `200` | 工具调用预算；在线 native 循环由 MAX_TURNS 控制。 |
| `MAX_TURNS` | `200` | agent 循环最大轮数。 |
| `MAX_CONTEXT` | `131072` | 逻辑历史 token 预算；不同于 Beacon 压缩后的 KV 长度。 |
| `MAX_NEW_TOKENS` | `8192` | 离线单次模型生成 token 上限。 |
| `MAX_TOOL_TOKENS` | `32768` | 工具返回 token 总预算。 |
| `DECODING` | `greedy` | greedy 或 sample；Beacon 仅 greedy。 |
| `REPETITION_PENALTY` | `1.0` | 重复惩罚；1.0 不惩罚，须为正数。 |
| `TEMPERATURE` | `1.0` | sample 生成温度；greedy 不使用。 |
| `TOP_P` | `1.0` | sample 核采样概率阈值。 |
| `GENERATION_SEED` | `42` | 生成随机种子，与裁判 SEED 独立。 |
| `LIMIT` | `-1` | -1 全量；正整数取前 N 题。 |
| `OUTPUT_DIR` | `空` | 本次结果的新目录；留空自动选择，禁止覆盖旧结果。 |
| `PREDICTIONS` | `空` | 已有预测 JSONL；离线 score 阶段必须设置。 |
| `JUDGE_MODEL` | `空` | BrowseComp-Plus 本地裁判完整权重目录。 |
| `RETRIEVAL_ONLY` | `0` | 1 只报告检索诊断，不报告答案 accuracy。 |
| `JUDGE_MAX_TOKENS` | `4096` | 裁判生成 token 上限。 |
| `JUDGE_MAX_CONTEXT` | `32768` | 裁判上下文 token 上限。 |
| `JUDGE_TEMPERATURE` | `0.7` | 裁判采样温度。 |
| `JUDGE_TOP_P` | `0.8` | 裁判核采样阈值。 |
| `JUDGE_TOP_K` | `20` | 裁判 top-k。 |
| `SEED` | `42` | 裁判随机种子。 |
| `DRY_RUN` | `0` | 0 实际执行；1 只解析并打印配置。 |

### 离线 HotpotQA（Plain/Beacon 共用同名变量）

配置源：`scripts/sh/plain/57_plain_hotpotqa.sh`。

| 变量 | 默认值 | 说明与修改提示 |
|---|---|---|
| `ACTION` | `all` | download/prepare/index/run/score/all；all 不包含 download。 |
| `ROOT_DIR` | `outputs/data/benchmarks/hotpotqa` | 该数据集共享原始数据、预处理数据和索引目录。 |
| `PYTHON` | `空` | Python 可执行文件的绝对路径；留空自动寻找环境。 |
| `CONDA_ENV` | `search-comp-qwen35-cu126` | 自动寻找的 Conda 环境名；PYTHON 显式可执行路径优先。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可见 GPU 编号列表，例如 0,1；只影响实际运行。 |
| `NUM_WORKERS` | `1` | 每卡一个完整模型副本，按题并行，不超过可见 GPU 数。 |
| `RUN_DIR` | `空` | 训练实验目录；Baseline 仅选择提示/数据。 |
| `MODEL_PATH` | `空` | 完整本地模型路径，覆盖 RUN_DIR 自动选择。 |
| `PROMPT_DATA` | `空` | 原训练 JSONL 的 system/schema 提示来源；不使用评测答案构造提示。 |
| `KIND` | `plain` | 固定模型方案，Plain=plain、Beacon=beacon。 |
| `DATA` | `空` | 问题数据；留空使用该基准默认原始文件。 |
| `CORPUS` | `空` | 独立完整语料路径；留空使用默认。 |
| `CORPUS_LIMIT` | `空` | 留空全语料；正数限制语料，仅用于小规模检查。 |
| `OFFICIAL_ONLY` | `0` | download 阶段 1 只安装代码自带的官方参考文件。 |
| `TITLE_WEIGHT` | `1.0` | FTS5 BM25 标题权重；改动需换目录重建索引。 |
| `VERIFY_INDEX` | `0` | 1 加载时额外验证整个索引 SHA256。 |
| `TOPK` | `5` | 每次搜索返回文档数量。 |
| `SNIPPET_TOKENS` | `512` | 每条搜索结果正文 token 上限，使用待测模型 tokenizer。 |
| `ALLOW_VISIT` | `1` | 1 允许读取本地全文；BCP 默认 0，Hotpot 默认 1。 |
| `READ_TOKENS` | `4096` | 一次读取本地文档的 token 上限。 |
| `TOOL_COUNT` | `200` | 工具调用预算；在线 native 循环由 MAX_TURNS 控制。 |
| `MAX_TURNS` | `200` | agent 循环最大轮数。 |
| `MAX_CONTEXT` | `131072` | 逻辑历史 token 预算；不同于 Beacon 压缩后的 KV 长度。 |
| `MAX_NEW_TOKENS` | `8192` | 离线单次模型生成 token 上限。 |
| `MAX_TOOL_TOKENS` | `32768` | 工具返回 token 总预算。 |
| `DECODING` | `greedy` | greedy 或 sample；Beacon 仅 greedy。 |
| `REPETITION_PENALTY` | `1.0` | 重复惩罚；1.0 不惩罚，须为正数。 |
| `TEMPERATURE` | `1.0` | sample 生成温度；greedy 不使用。 |
| `TOP_P` | `1.0` | sample 核采样概率阈值。 |
| `GENERATION_SEED` | `42` | 生成随机种子，与裁判 SEED 独立。 |
| `LIMIT` | `-1` | -1 全量；正整数取前 N 题。 |
| `OUTPUT_DIR` | `空` | 本次结果的新目录；留空自动选择，禁止覆盖旧结果。 |
| `PREDICTIONS` | `空` | 已有预测 JSONL；离线 score 阶段必须设置。 |
| `DRY_RUN` | `0` | 0 实际执行；1 只解析并打印配置。 |

### Baseline 离线共享配置（74/75 固定 TARGET）

配置源：`scripts/sh/baseline/73_qwen35_offline.sh`。

| 变量 | 默认值 | 说明与修改提示 |
|---|---|---|
| `PYTHON` | `空` | Python 可执行文件的绝对路径；留空自动寻找环境。 |
| `CONDA_ENV` | `search-comp-qwen35-cu126` | 自动寻找的 Conda 环境名；PYTHON 显式可执行路径优先。 |
| `TARGET` | `hotpotqa` | Baseline 数据集；74 固定 browsecomp-plus、75 固定 hotpotqa。 |
| `ACTION` | `run` | download/prepare/index/run/score/all；all 不包含 download。 |
| `BASE_CHECKPOINT` | `Qwen/Qwen3.5-4B` | 原始 Qwen3.5 模型路径；run/all 需要本地完整权重。 |
| `RUN_DIR` | `空` | 训练实验目录；Baseline 仅选择提示/数据。 |
| `PROMPT_DATA` | `空` | 原训练 JSONL 的 system/schema 提示来源；不使用评测答案构造提示。 |
| `ROOT_DIR` | `outputs/data/benchmarks/$TARGET` | 该数据集共享原始数据、预处理数据和索引目录。 |
| `DATA` | `空` | 问题数据；留空使用该基准默认原始文件。 |
| `CORPUS` | `空` | 独立完整语料路径；留空使用默认。 |
| `CORPUS_LIMIT` | `空` | 留空全语料；正数限制语料，仅用于小规模检查。 |
| `OFFICIAL_ONLY` | `0` | download 阶段 1 只安装代码自带的官方参考文件。 |
| `TITLE_WEIGHT` | `1.0` | FTS5 BM25 标题权重；改动需换目录重建索引。 |
| `VERIFY_INDEX` | `0` | 1 加载时额外验证整个索引 SHA256。 |
| `TOPK` | `5` | 每次搜索返回文档数量。 |
| `SNIPPET_TOKENS` | `512` | 每条搜索结果正文 token 上限，使用待测模型 tokenizer。 |
| `ALLOW_VISIT` | `$DEFAULT_VISIT` | 1 允许读取本地全文；BCP 默认 0，Hotpot 默认 1。 |
| `READ_TOKENS` | `4096` | 一次读取本地文档的 token 上限。 |
| `TOOL_COUNT` | `200` | 工具调用预算；在线 native 循环由 MAX_TURNS 控制。 |
| `MAX_TURNS` | `200` | agent 循环最大轮数。 |
| `MAX_CONTEXT` | `131072` | 逻辑历史 token 预算；不同于 Beacon 压缩后的 KV 长度。 |
| `MAX_NEW_TOKENS` | `8192` | 离线单次模型生成 token 上限。 |
| `MAX_TOOL_TOKENS` | `32768` | 工具返回 token 总预算。 |
| `LIMIT` | `-1` | -1 全量；正整数取前 N 题。 |
| `NUM_WORKERS` | `1` | 每卡一个完整模型副本，按题并行，不超过可见 GPU 数。 |
| `OUTPUT_DIR` | `空` | 本次结果的新目录；留空自动选择，禁止覆盖旧结果。 |
| `PREDICTIONS` | `空` | 已有预测 JSONL；离线 score 阶段必须设置。 |
| `DECRYPTED` | `0` | 1 表示提供已解密的 BrowseComp-Plus 官方问题 JSONL。 |
| `RETRIEVAL_ONLY` | `0` | 1 只报告检索诊断，不报告答案 accuracy。 |
| `JUDGE_MODEL` | `空` | BrowseComp-Plus 本地裁判完整权重目录。 |
| `JUDGE_MAX_TOKENS` | `4096` | 裁判生成 token 上限。 |
| `JUDGE_MAX_CONTEXT` | `32768` | 裁判上下文 token 上限。 |
| `JUDGE_TEMPERATURE` | `0.7` | 裁判采样温度。 |
| `JUDGE_TOP_P` | `0.8` | 裁判核采样阈值。 |
| `JUDGE_TOP_K` | `20` | 裁判 top-k。 |
| `SEED` | `42` | 裁判随机种子。 |
| `DECODING` | `greedy` | greedy 或 sample；Beacon 仅 greedy。 |
| `REPETITION_PENALTY` | `1.0` | 重复惩罚；1.0 不惩罚，须为正数。 |
| `TEMPERATURE` | `1.0` | sample 生成温度；greedy 不使用。 |
| `TOP_P` | `1.0` | sample 核采样概率阈值。 |
| `GENERATION_SEED` | `42` | 生成随机种子，与裁判 SEED 独立。 |
| `DRY_RUN` | `0` | 0 实际执行；1 只解析并打印配置。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可见 GPU 编号列表，例如 0,1；只影响实际运行。 |

### Plain 模型服务

配置源：`scripts/sh/plain/48_serve_checkpoint.sh`。完整服务安装与兼容性说明见 [48 部署手册](../plain/48_sglang_deployment.md)。

| 变量 | 默认值 | 说明与修改提示 |
|---|---|---|
| `PYTHON` | `空` | Python 可执行文件的绝对路径；留空自动寻找环境。 |
| `CONDA_ENV` | `search-comp-sglang` | 自动寻找的 Conda 环境名；PYTHON 显式可执行路径优先。 |
| `RUN_DIR` | `空` | 训练实验目录；Baseline 仅选择提示/数据。 |
| `MODEL_PATH` | `空` | 完整本地模型路径，覆盖 RUN_DIR 自动选择。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可见 GPU 编号列表，例如 0,1；只影响实际运行。 |
| `HOST` | `127.0.0.1` | 监听地址；跨节点设 0.0.0.0，客户端填真实节点名/IP。 |
| `TP_SIZE` | `1` | Plain 服务张量并行卡数。 |
| `PORT` | `30018` | Plain completion 服务端口。 |
| `CONTEXT_LENGTH` | `131072` | Plain 服务最大上下文长度。 |
| `DRY_RUN` | `0` | 0 实际执行；1 只解析并打印配置。 |

## 46.5 迁移清单与后续维护

| 旧入口（已移除） | 新入口 |
|---|---|
| `benchmark/47_openseeker_benchmark.sh` | `plain/47_plain_browsecomp.sh` 或 `beacon/47_beacon_browsecomp.sh` |
| `benchmark/56_browsecomp_plus.sh` | `plain/56_plain_browsecomp_plus.sh` 或 `beacon/56_beacon_browsecomp_plus.sh` |
| `benchmark/57_hotpotqa_offline.sh` | `plain/57_plain_hotpotqa.sh` 或 `beacon/57_beacon_hotpotqa.sh` |
| `benchmark/48_serve_checkpoint.sh` | `plain/48_serve_checkpoint.sh` |
| `baseline/73_qwen35_offline.sh`（仍兼容） | `baseline/74_qwen35_browsecomp_plus.sh`、`baseline/75_qwen35_hotpotqa.sh` |

旧的 benchmark 路径不保留执行转发，批处理、个人笔记和集群作业请更新路径。文档中的公共离线数据准备示例采用 Plain 入口，仅执行 download/prepare/index 时不会加载 Plain 模型；Beacon 入口相同阶段生成兼容的共享数据。

新增参数时：在各相关方案脚本顶部添加带默认值的变量及用途注释；离线公共 CLI 映射只改 benchmark/58；Baseline 离线映射改 baseline/73。需要改变实际推理/评分功能时再修改 Python 工作流，并同步本表和对应协议文档。不要把测试入口重新放回 benchmark 目录。
