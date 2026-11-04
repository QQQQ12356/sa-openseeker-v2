# 68 两数据集、三种方法与独立测试评分

完整的数据下载、HTML 检查和整体分布到训练/测试/评分顺序，分别见 [70 OpenSeeker](../data/70_openseeker_experiment_guide.md) 与 [71 Research](../data/71_research_experiment_guide.md)。本文件负责独立脚本和参数查阅。

本手册是当前独立脚本与参数参考。所有命令在 **Linux 项目根目录**执行；Windows 只修改代码和进行不加载模型的检查。没有在本次修改中运行预处理、真实训练、真实推理或真实裁判。

## 68.1 实验定义与脚本矩阵

训练数据只有 OpenSeeker 和 Research。`data_mode: openseeker` 是两者共享的轨迹编码协议，`dataset_name` 才是数据身份。OpenSeeker 的 Polar/Zephyr/combined 是同一数据路线的来源选择。Research 先通过独立适配器转换角色，保留其研究指令、search/visit schema 和示范答案，不把示范答案当成独立 benchmark 标注。

| 方法 | 配置与训练实现 | 评测加载 |
|---|---|---|
| plain | 现有 Plain SFT Trainer，默认 `use_lora: true` | 现有 Plain checkpoint 加载器 |
| beacon | 现有 Beacon Trainer，保留压缩及蒸馏配置 | Beacon checkpoint 加载器 |
| baseline | 标准全参数 SFT，复用无 Beacon 的 Plain Trainer，`use_lora: false` | Plain checkpoint 加载器 |

Baseline 新训练入口用于满足每种方法均训练两种数据的实验矩阵。其底层 `kind: plain`，实验归档另写 `method: baseline`，实验目录名称也包含 baseline。**全参数 baseline 和 Plain LoRA 的可训练参数量不同**；若要比较纯压缩机制，需在各训练脚本中明确统一训练策略并记录。现有 `baseline/70`–`75` 保留“未训练原始 Qwen3.5”含义，新 baseline 测试不使用这些入口。

下表中的文件在 `scripts/sh/plain/`、`scripts/sh/beacon/`、`scripts/sh/baseline/` **分别独立存在**。没有 METHOD/TARGET 总调度入口；用户修改所选脚本自己的参数。相同 Python 计算逻辑和参数映射可内部复用。

| 文件名 | 固定任务 | 输入 | 输出 |
|---|---|---|---|
| `10_train_openseeker.sh` | OpenSeeker 训练 | prepared 目录 | 新实验目录及模型 |
| `11_train_research.sh` | Research 训练 | prepared 目录 | 新实验目录及模型 |
| `40_test_browsecomp.sh` | 在线 BrowseComp，只生成 | RUN_DIR 或模型+PROMPT_DATA、问题、工具服务 | `result_tool200.jsonl`、对齐结果、coverage |
| `41_score_browsecomp.sh` | BrowseComp，只评分 | 原问题、预测、在线裁判服务 | `summary.metrics.json`、逐题裁判日志 |
| `42_test_browsecomp_plus.sh` | 本地 BrowseComp-Plus，只生成 | 训练模型、原提示词、prepared+索引 | `predictions.jsonl`、manifest |
| `43_score_browsecomp_plus.sh` | BrowseComp-Plus，只评分 | prepared 问题、预测、本地裁判 | `summary.metrics.json`、逐题答案/检索/引用评分 |
| `44_test_hotpotqa.sh` | 本地 HotpotQA，只生成 | 训练模型、原提示词、prepared+索引 | `predictions.jsonl`、manifest |
| `45_score_hotpotqa.sh` | HotpotQA，只评分 | prepared 问题、预测、本地裁判 | 官方答案/支持事实指标、LLM judge |

同一训练数据集生成的模型通过同一 RUN_DIR 运行三个测试入口。不要让 Research 训练模型使用 OpenSeeker 的训练 JSONL 作为 PROMPT_DATA。对另外两种方法重复相同步骤时，直接替换目录名即可。

## 68.2 数据准备与训练

先准备 Linux 训练环境和本地 tokenizer/模型缓存，参见 [01 环境](../common/01_environment.md)。不会自动从 Windows 路径迁移模型或训练数据。

```bash
# OpenSeeker：先编辑数据脚本，选择输入、输出、模型 tokenizer、长度和筛选条件。
bash scripts/sh/data/10_prepare.sh

# Research：只读本地角色导出文件，不下载数据。
INPUT=/data/research.jsonl MODEL_PATH=/models/Qwen3.5-4B \
  DATASET_DIR=outputs/data/research/prepared_16k \
  bash scripts/sh/data/09_prepare_research.sh

# 查看最终配置与命令。plan 需要上一步数据存在，不加载模型、不开始训练。
DATASET_DIR=outputs/data/research/prepared_16k MODEL_PATH=/models/Qwen3.5-4B \
  ACTION=plan bash scripts/sh/plain/11_train_research.sh

# 正式训练会创建另一个新的实验目录。只有在 Linux 用户自行执行本命令才开始。
DATASET_DIR=outputs/data/research/prepared_16k MODEL_PATH=/models/Qwen3.5-4B \
  CUDA_VISIBLE_DEVICES=0,1,2,3 DISTRIBUTED_BACKEND=zero2 \
  bash scripts/sh/plain/11_train_research.sh
```

| 训练 shell 变量 | 默认或规则 | 作用与修改提示 |
|---|---|---|
| `PYTHON` | 留空自动查找 | 指定 Linux 可执行 Python 绝对路径；应存在且可执行 |
| `CONDA_ENV` | `search-comp-qwen35-cu126` | 仅帮助查找 Python，不负责安装环境 |
| `CUDA_VISIBLE_DEVICES` | `0` | 单卡或逗号分隔的可见 GPU；多卡按可见卡数推断进程数 |
| `DISTRIBUTED_BACKEND` | `zero2` | `ddp`/`zero2`；Plain 和 baseline 另支持 `zero3` |
| `NPROC_PER_NODE` | 自动推断 | 显式覆盖每节点训练进程数，不应超过可见 GPU 数 |
| `ACTION` | `train` | `plan` 只生成配置与命令；`train` 创建新实验并启动训练 |
| `DATASET_NAME` | 入口固定 | 10 固定 openseeker，11 固定 research；不要从环境切换数据身份 |
| `DATASET_DIR` | 入口对应 prepared 目录 | 含 train/validation/test.jsonl、manifest.json；不会自动预处理 |
| `MODEL_PATH` | `Qwen/Qwen3.5-4B` | 与预处理登记的 tokenizer 身份相同；离线服务器推荐本地路径 |
| `MAX_LENGTH` | `16384` | 完整轨迹 token 上限；需大于等于预处理上限，超长不静默截断 |
| `CONFIG_FILE`、`PY` | 内部生成 | 临时 YAML 路径与解析后的 Python；不作为用户覆盖变量 |

每个训练文件内置完整 YAML，不依赖外部基础配置。逐项参数见 [69](69_training_parameters.md)。同名环境变量只覆盖上述显式支持的字段；例如 `LEARNING_RATE=...` **不会**覆盖 YAML，学习率必须修改所选训练文件。优先级为：脚本完整 YAML → 脚本导出的模型/数据/长度覆盖 → experiment 解析及归档。实验输出包括 `source_config.yaml`、`train.yaml`、`experiment.json`、代码和数据指纹、训练日志、events.jsonl。改动代码、数据或参数后创建新实验，不复用旧 plan。

预处理 Research 的变量详见 [09](../data/09_research_sft.md)。其中 `INPUT` 指原始 JSONL；`WORKERS` 为 CPU 分词进程数；`SEED` 控制按问题分组划分；`VAL_FRACTION`、`TEST_FRACTION` 默认各 0.02，可设 0；`OBSERVATIONS=with/without/any` 决定观测筛选。完整序列超长时整条过滤。验证/测试划分只供训练内部检查，不能代替三项独立 benchmark。

## 68.3 Benchmark 准备与格式一致性

在线 BrowseComp 先执行 `bash scripts/sh/benchmark/46_prepare_benchmark.sh`，固定官方版本与问题文件。生成需要在线 search/visit 及摘要服务；详细服务变量见 [49](../benchmark/49_local_aux_models.md) 和 `scripts/sh/benchmark/49_benchmark_env.example.sh`。新生成入口不要求 SCORER 服务。

两个离线 benchmark 必须先下载、规范化及建立索引。准备阶段继续使用原有独立 benchmark 脚本，例如：

```bash
# BrowseComp-Plus，仅准备独立测试数据和固定 corpus。
ACTION=download bash scripts/sh/plain/56_plain_browsecomp_plus.sh
ACTION=prepare bash scripts/sh/plain/56_plain_browsecomp_plus.sh
ACTION=index bash scripts/sh/plain/56_plain_browsecomp_plus.sh

# HotpotQA，完整 dev 与官方 Wikipedia 摘要语料。
ACTION=download bash scripts/sh/plain/57_plain_hotpotqa.sh
ACTION=prepare bash scripts/sh/plain/57_plain_hotpotqa.sh
ACTION=index bash scripts/sh/plain/57_plain_hotpotqa.sh
```

其余方法共用同一份 benchmark 资产即可，不需重复下载或建立索引。自备文件、解密、完整语料与索引校验见 [43](../benchmark/43_offline_benchmarks.md)。新 42/44 测试入口固定 run，不自动下载、不自动建索引。

提示词解析优先级：显式 `PROMPT_DATA` → RUN_DIR 的 `experiment.json.training_config.train_data_path`。新入口要求至少提供其中之一，避免默默使用内置 Polar 提示。自动读取训练 JSONL 时核对归档 SHA256；文件改动后拒绝评测。显式 PROMPT_DATA 是用户提供的覆盖，需自行保证属于当前实验。必须使用**规范化的 train.jsonl**，直接传 Research 原始角色导出会报错。

所有行的有效 system+tools 必须一致；混合模板数据会拒绝任意挑选第一行。此时应将同一提示词配置的规范化样本单独写成 JSONL，并显式指定。读取提示词只使用 system/schema，不使用训练问题、答案或 observation。

| Benchmark | 对原训练格式的必要调整 |
|---|---|
| BrowseComp | 原 system/schema 与原生轨迹序列化，在线 search/visit；不改成官方不同模板 |
| BrowseComp-Plus | 保留原研究指令及工具参数 schema；工具说明改为本地 corpus；默认 search-only；最终回答追加 `[docid]` 引用要求 |
| HotpotQA | 保留原研究指令及工具参数 schema；search/visit 限定本地 corpus；追加 `<answer>{"answer":"简短答案","sp":[["标题",0]]}</answer>` 要求 |

Hotpot 的句子编号从 0 开始；不能确认支持句时 `sp: []`，不得从 gold 反推支持事实。BCP 开启 visit 属于扩展协议，需在报告中记录。模型生成、工具调用解析和 token 监督协议复用原有实现，Research 不另造推理模板。

## 68.4 三项测试与独立评分示例

```bash
# 使用一个明确的训练实验；三项测试均从这里取得模型与原始训练提示词。
export RUN_DIR=/experiments/qwen35_plain_research_run

ENV_FILE=/private/benchmark.env \
  OUTPUT_DIR=outputs/results/plain/research/bc_run \
  bash scripts/sh/plain/40_test_browsecomp.sh
ENV_FILE=/private/benchmark.env \
  PREDICTIONS=outputs/results/plain/research/bc_run/result_tool200.jsonl \
  OUTPUT_DIR=outputs/results/plain/research/bc_score \
  bash scripts/sh/plain/41_score_browsecomp.sh

OUTPUT_DIR=outputs/results/plain/research/bcp_run \
  bash scripts/sh/plain/42_test_browsecomp_plus.sh
PREDICTIONS=outputs/results/plain/research/bcp_run/predictions.jsonl \
  JUDGE_MODEL=/models/Qwen3-32B OUTPUT_DIR=outputs/results/plain/research/bcp_score \
  bash scripts/sh/plain/43_score_browsecomp_plus.sh

OUTPUT_DIR=outputs/results/plain/research/hotpot_run \
  bash scripts/sh/plain/44_test_hotpotqa.sh
PREDICTIONS=outputs/results/plain/research/hotpot_run/predictions.jsonl \
  JUDGE_MODEL=/models/Qwen3-32B OUTPUT_DIR=outputs/results/plain/research/hotpot_score \
  bash scripts/sh/plain/45_score_hotpotqa.sh
```

每次运行使用新 OUTPUT_DIR，旧结果不覆盖。不同方法、数据集、benchmark、阶段建议分别命名，避免结果混用。以上命令对 OpenSeeker 模型也适用，但必须更换对应 RUN_DIR。单独使用模型时同时提供 `MODEL_PATH=/models/merged` 与 `PROMPT_DATA=/data/prepared/train.jsonl`；不要把 LoRA adapter 路径当成已合并的离线 checkpoint。

## 68.5 测试脚本变量表

三个方法同名新入口的变量含义相同。下表是设置入口，而不是总调度接口。

| 变量 | 默认/范围 | 含义与修改提示 |
|---|---|---|
| `RUN_DIR` | 空，推荐必填 | 对应训练实验目录，自动取最终模型和提示词 |
| `MODEL_PATH` | 空 | 显式模型覆盖；脱离 RUN_DIR 时必须提供 PROMPT_DATA |
| `PROMPT_DATA` | 空 | 原始有效训练提示的规范化 JSONL，只取 system/tools |
| `KIND` | 方法固定 | Beacon 为 beacon，其余为 plain；不能用于切换方法 |
| `MODE` | 在线新入口固定 native | 保证使用训练协议 |
| `DATA` | BrowseComp 默认问题 JSONL；离线默认 ROOT_DIR/raw | 原始问题；新离线 run 实际读取 ROOT_DIR/prepared，改 DATA 后须先重新 prepare |
| `BENCHMARK` | 入口固定 | 对应的测试集，不用它切换到其他 benchmark |
| `ROOT_DIR` | `outputs/data/benchmarks/<benchmark>` | 离线资产、prepared 和索引目录 |
| `OUTPUT_DIR` | 自动新目录 | 所有预测和 manifest 存放位置，不得已存在 |
| `PREDICTIONS` | 测试必须为空 | 新测试入口拒绝已有预测；已有预测交对应评分脚本 |
| `LIMIT` | `-1` | -1 全量，正整数前 N 题；测试与评分保持一致 |
| `CUDA_VISIBLE_DEVICES` | 离线默认 0 | 推理 GPU；在线沿用环境 |
| `NUM_WORKERS` | `1` | 离线每卡完整模型副本，按题并行，不超过可见卡数 |
| `WORKERS` | `60` | 上游官方兼容参数；新 native 在线入口不按此值并行题目 |
| `MAX_CONTEXT` | `131072` | 序列化历史的逻辑预算，不是 Beacon 压缩后物理 KV 长度 |
| `MAX_TOKENS` | 在线 `32768` | 在线生成预算 |
| `MAX_NEW_TOKENS` | 离线 `8192` | 离线每轮生成预算 |
| `MAX_TURNS` | `200` | 原生工具交互轮次上限 |
| `MAX_TOOL_TOKENS` | `32768` | 工具 observation 预算 |
| `TOOL_COUNT` | `200` | 在线结果文件名的上游兼容参数；离线限制工具调用数；原生轮次由 MAX_TURNS 约束 |
| `TOPK` | `5` | 离线 BM25 每次查询返回文档数 |
| `SNIPPET_TOKENS` | `512` | 每篇搜索结果正文上限，以待测 tokenizer 计数 |
| `ALLOW_VISIT` | BCP 0，Hotpot 1 | 1 可读本地全文，0 只有搜索；不能访问外网 |
| `READ_TOKENS` | `4096` | 本地 visit 文本 token 上限 |
| `TITLE_WEIGHT` | `1.0` | BM25 标题权重；更改后换 ROOT_DIR 并重建索引 |
| `VERIFY_INDEX` | `0` | 1 加载时额外完整 SHA256 核验 |
| `DECODING` | `greedy` | Plain/baseline 支持 sample，Beacon 使用 greedy |
| `REPETITION_PENALTY` | `1.0` | 1 不惩罚，>1 抑制重复 |
| `TEMPERATURE`、`TOP_P` | `1.0`、`1.0` | 仅生成 sample 生效 |
| `GENERATION_SEED` | `42` | 生成随机种子，独立于裁判 SEED |
| `TOOL_CALL_RETRIES` | `1` | 每题格式恢复总额度；0 严格失败 |
| `TOOL_POLICY` | 在线 `stable` | 在线 visit 的恢复策略，兼容可设 official |
| `ENV_FILE` | 在线空 | 私有 search/visit/summary 服务 shell 文件，执行前加载；不要提交密钥 |
| `DRY_RUN` | `0` | 1 仅打印解析后的参数，不加载模型或请求服务 |
| `PYTHON`、`CONDA_ENV` | 与训练一致 | 选择 Linux Python |

兼容离线测试脚本仍列出 `CORPUS`、`CORPUS_LIMIT`、`DECRYPTED`、`OFFICIAL_ONLY`，只在旧 download/prepare/index 阶段生效，新 run 使用现有 prepared/index。`CORPUS_LIMIT` 正数属于 smoke，不能声称完整 corpus 结果。测试中保留的 JUDGE_*、SEED、LLM_JUDGE 参数只为底层参数映射兼容；run 不评分，正式设置请放到独立 score 入口。新测试入口 ACTION 固定为 run。

## 68.6 评分参数、指标和失败口径

| 变量 | 默认/范围 | 作用 |
|---|---|---|
| `PREDICTIONS` | 必填 | 生成阶段输出文件；评分不加载待测模型 |
| `DATA` | 在线 BrowseComp JSONL | 与测试相同的问题和参考答案 |
| `ROOT_DIR` | 离线 benchmark 目录 | 使用校验过的 prepared 问题及官方评分代码 |
| `OUTPUT_DIR` | 自动新目录 | 评分产物目录，不与生成目录重用 |
| `LIMIT` | `-1` | 必须与测试选取一致；缺失题不自动排除 |
| `JUDGE_MODEL` | 离线必填 | 本地裁判权重目录，不是待测模型目录 |
| `ENV_FILE` | 在线空 | 设置 SCORER_URLS、SCORER_API_KEY、SCORER_MODEL_NAME |
| `SCORER_URLS` | 在线必填 | 裁判 API 地址；沿用 ChatJudge 的现有协议 |
| `SCORER_API_KEY`、`SCORER_MODEL_NAME` | 在线必填 | 裁判密钥及服务中的模型名称 |
| `JUDGE_WORKERS` | 在线 1 | 裁判请求并发数 |
| `JUDGE_MAX_TOKENS` | 在线 1024、离线 4096 | 裁判输出预算 |
| `JUDGE_MAX_CONTEXT` | 离线 32768 | 裁判提示+输出总预算，超限报错而不是截断 |
| `JUDGE_TIMEOUT` | 在线 120 秒 | 单次请求超时 |
| `JUDGE_ATTEMPTS` | 3 | 每题最多总尝试次数；有效负判不重试 |
| `JUDGE_ENABLE_THINKING` | 0 | 0 关闭裁判思考，1 开启 |
| `SKIP_JUDGE_PREFLIGHT` | 0 | 默认先做正负例校准；1 跳过应在报告中说明 |
| `JUDGE_TEMPERATURE` | 离线 0 | 0 贪心；正数启用采样 |
| `JUDGE_TOP_P`、`JUDGE_TOP_K` | 离线 0.8、20 | 仅裁判采样生效 |
| `SEED` | 离线 42 | 裁判随机种子，与生成种子无关 |
| `CUDA_VISIBLE_DEVICES` | 离线 0 | 裁判 GPU，独立评分可使用与待测模型相同的 GPU |
| `DRY_RUN`、`PYTHON`、`CONDA_ENV` | 同前 | 仅打印计划或选择解释器 |

| Benchmark | EM / F1 | LLM judge | 其他指标 |
|---|---|---|---|
| BrowseComp | 补充答案指标，`em`、`f1`，0–1 | 原官方提示+稳定裁判，`llm_judge_accuracy` 等于全题分母 accuracy | coverage、官方裁判成功率 |
| BrowseComp-Plus | 补充答案指标，生成 final answer 去 `[docid]` 引用标记，再计算 EM/token F1 | 原官方提示+本地裁判，`llm_judge_accuracy` 等于 accuracy | evidence/gold recall、citation、calibration |
| HotpotQA | 原官方答案 `em`、`f1`，保留 yes/no 等特例 | 答案语义等价补充指标 `llm_judge_accuracy` | 原官方 supporting fact / joint 指标 |

BrowseComp 两项补充 EM/F1 使用已有英文大小写、标点、冠词和空格归一化；F1 为 token 重合度。它们不是官方语义准确率的别名。答案采用模型 final_response（可提取 `<answer>`），不使用裁判提取的答案；BCP 引用移除规则为方括号标记，因此原答案中的方括号内容也会被此补充口径移除。Hotpot 的答案从生成 JSON 的 answer 字段提取，sp 单独评分。

所有题保留在分母。缺失、失败、空输出计零。无效裁判输出保留错误及重试历史，`complete: false`，流程非零退出；不能作为完整 benchmark 结果。正式 score 入口强制 LLM judge，不提供跳过裁判的 shell 开关；底层 Python 和历史入口保留检索诊断等能力。

## 68.7 修改提示、兼容入口与检查

1. 修改训练策略：编辑具体方法、具体数据集的 10/11 文件中的 YAML；相同方法的两份文件相互独立，比较实验时主动同步公共超参数。
2. 修改推理预算：编辑具体 benchmark 的 40/42/44 或设置它支持的变量。新入口固定 benchmark/阶段，避免用尾随 CLI 参数覆盖到其他任务。
3. 修改裁判：在线修改私有 ENV_FILE，离线修改 JUDGE_MODEL；同时记录裁判版本、thinking、预算、温度和 seed。
4. 修改 tokenizer/长度：重新预处理并换 prepared 目录；训练 MODEL_PATH 必须与 manifest 登记身份一致。把 Linux 真实绝对路径写入 Linux 运行环境，不能使用 Windows 归档中的盘符路径。
5. 旧 Plain 20 / Beacon 30 保留为 OpenSeeker 10 的兼容跳转；Research 使用 11。旧 47/56/57 和 baseline 70–75 仍为历史兼容工具，可能支持 all 或其他提示方式；当前实验使用新独立测试和评分入口。Search-R1 的 22/32/34 shell 训练入口已停用并返回迁移提示；底层旧实现和配置留作历史兼容，当前矩阵只支持两种训练数据。
6. 已删除 BrowseComp SFT 适配器及 data/14、plain/25、beacon/39 训练入口。预处理 CLI 不接受 source=browsecomp；experiment 拒绝旧 BrowseComp SFT manifest。Benchmark 数据和工具未移除。

无模型回归检查可运行：

```bash
python -m unittest discover -s tests -p 'test_training_benchmark_matrix.py' -v
python -m unittest discover -s tests -p 'test_research_sft.py' -v
python -m unittest discover -s tests -p 'test_hotpot_judge.py' -v
python -m unittest discover -s tests -p 'test_experiment_dataset_name.py' -v
```

第一项覆盖 18 个测试/评分脚本的真实 CLI dry-run、6 个训练文件的 Bash 语法与固定配置、数据提示来源和指标分母。训练脚本只检查文本和语法，不调用 train/plan。其余检查使用内存样例或临时文件，不下载模型、不请求裁判。
