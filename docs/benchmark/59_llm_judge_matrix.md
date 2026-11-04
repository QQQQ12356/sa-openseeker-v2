# 59 Plain、Beacon、原始模型的三类基准与 LLM-Judge

三种方法现在均有 HotpotQA、BrowseComp、BrowseComp-Plus 的 LLM-Judge 路径。新增 HotpotQA 语义评分；BrowseComp 保留已有在线稳定裁判，BrowseComp-Plus 保留固定官方 prompt/parser 和本地裁判。不改变三种基准各自的工具环境或官方指标含义。

## 59.1 九种组合与入口

统一入口为 `scripts/sh/benchmark/59_run_benchmark.sh`。它只分发到既有脚本，不自动重复运行九次。

| METHOD | BENCHMARK=browsecomp | BENCHMARK=browsecomp-plus | BENCHMARK=hotpotqa |
|---|---|---|---|
| plain | plain/47_plain_browsecomp.sh | plain/56_plain_browsecomp_plus.sh | plain/57_plain_hotpotqa.sh |
| beacon | beacon/47_beacon_browsecomp.sh | beacon/56_beacon_browsecomp_plus.sh | beacon/57_beacon_hotpotqa.sh |
| base | baseline/72_qwen35_browsecomp.sh | baseline/74_qwen35_browsecomp_plus.sh | baseline/75_qwen35_hotpotqa.sh |

`METHOD` 默认 plain，`BENCHMARK` 默认 hotpotqa。其余环境变量和命令尾部 CLI 参数透传具体入口。Plain/Beacon 用 `RUN_DIR` 自动解析训练结果；也可 `MODEL_PATH` 显式指定模型。原始模型使用 `BASE_CHECKPOINT`，保持独立模型加载逻辑。Beacon 应走 native 推理以保留压缩行为。

BrowseComp SFT 训练后的 RUN_DIR 会定位其 prepared 训练提示，并将原历史工具 schema 适配为 benchmark 实际提供的 search/visit；保留原研究说明、声明历史工具不可用。原始模型要使用相同提示时，以 `PROMPT_DATA` 指向同一份 prepared/train.jsonl。此行为仅作用于带 browsecomp 来源和 training_only 标记的 prepared 数据；其他训练提示沿用原处理。训练与评测的能力差别及边界见 [14.5](../data/14_browsecomp_sft.md#145-产物追溯与评测边界)。

| 基准 | 裁判接入 | 主要分数 | 附加分数 |
|---|---|---|---|
| BrowseComp | 既有 SCORER OpenAI-compatible HTTP 服务 | 在线模型裁判准确率 | 原有覆盖/工具统计 |
| BrowseComp-Plus | 本地 JUDGE_MODEL，固定官方提示及 parser | accuracy / Accuracy (%) | 检索召回、引用、校准统计 |
| HotpotQA | 本地 JUDGE_MODEL，hotpot-answer-equivalence-v1 | llm_judge_accuracy，0..1 | 官方 em/f1、支持事实和 Joint 指标继续输出 |

HotpotQA 的语义正确性只比较回答与参考答案的事实含义，不对支持句作语义评分；SP/Joint 仍依赖模型自己生成的支持事实。三个基准的分数不能互相替代，也不能将新增语义分数称为 HotpotQA 官方指标。

## 59.2 一键运行和先检查参数

以下命令在已配置的 Linux 环境中运行。`DRY_RUN=1` 只解析输出参数，不加载待测模型和裁判，不请求 API，也不构建索引。

```bash
METHOD=plain BENCHMARK=hotpotqa ACTION=all DRY_RUN=1 \
  RUN_DIR=outputs/experiments/YOUR_RUN JUDGE_MODEL=/data/models/Qwen3-32B \
  bash scripts/sh/benchmark/59_run_benchmark.sh

# 已准备 Hotpot 官方数据/索引时，生成并评分；LIMIT=8 为小规模检查。
METHOD=plain BENCHMARK=hotpotqa ACTION=all LIMIT=8 \
  RUN_DIR=outputs/experiments/YOUR_RUN JUDGE_MODEL=/data/models/Qwen3-32B \
  bash scripts/sh/benchmark/59_run_benchmark.sh

METHOD=beacon BENCHMARK=browsecomp-plus ACTION=all LIMIT=8 \
  RUN_DIR=outputs/experiments/YOUR_BEACON_RUN JUDGE_MODEL=/data/models/Qwen3-32B \
  bash scripts/sh/benchmark/59_run_benchmark.sh

METHOD=base BENCHMARK=hotpotqa ACTION=all LIMIT=8 \
  BASE_CHECKPOINT=/data/models/Qwen3.5-4B JUDGE_MODEL=/data/models/Qwen3-32B \
  bash scripts/sh/benchmark/59_run_benchmark.sh

# 在线 BrowseComp：ENV_FILE 提供搜索、网页摘要和 SCORER 服务配置。
METHOD=plain BENCHMARK=browsecomp LIMIT=8 \
  RUN_DIR=outputs/experiments/YOUR_RUN ENV_FILE=/private/benchmark_env.sh \
  bash scripts/sh/benchmark/59_run_benchmark.sh
```

对所有组合使用统一题集、LIMIT、生成预算、检索索引、解码和裁判配置。正式全量使用 `LIMIT=-1`。原始模型离线脚本默认 `ACTION=run`，Plain/Beacon 离线默认 all，因此比较时显式指定 ACTION。BrowseComp 在线脚本不沿用离线 ACTION 定义：Plain/Beacon 通过 PREDICTIONS 切换仅评分，原始模型用 ACTION=run/score。

HotpotQA 的 shell 入口默认 `LLM_JUDGE=1`，因此 all/score 必须提供本地 `JUDGE_MODEL`；缺失时报错，不静默跳过。download/prepare/index/run 不需要加载裁判。旧实验仅需确定性官方指标时设置 `LLM_JUDGE=0`。直接调用 Python CLI 时保持兼容：默认只做官方评分；传 `--llm-judge --judge-model ...` 或直接指定 `--judge-model ...` 才增加 Hotpot 语义评分。

## 59.3 离线裁判变量：HotpotQA 与 BrowseComp-Plus

| 变量 | 默认 | 用途与修改提示 |
|---|---|---|
| JUDGE_MODEL | 空 | 本地裁判完整模型目录；all/score 的语义评分必填；模型不会自动下载 |
| LLM_JUDGE | Hotpot 为 1 | 1 加语义分数；0 仅保留官方指标；BCP 不使用此开关 |
| JUDGE_MAX_TOKENS | 4096 | 裁判每次输出 token 上限；不改变被测模型生成预算 |
| JUDGE_MAX_CONTEXT | 32768 | 裁判输入加输出的总预算；超限报错，不截断参考答案或候选回答 |
| JUDGE_TEMPERATURE | 0 | 默认贪心，尽量减少判分随机性 |
| JUDGE_TOP_P | 0.8 | 仅温度大于 0 时参与采样 |
| JUDGE_TOP_K | 20 | 仅采样时生效；正整数 |
| SEED | 42 | 本地裁判种子，独立于 GENERATION_SEED |
| JUDGE_ATTEMPTS | 3 | 单题最大总尝试次数；只重试传输、输出格式等错误；有效负判不重试 |
| JUDGE_ENABLE_THINKING | 0 | 显式关闭思考；1 开启时也只读取最终标签 |
| SKIP_JUDGE_PREFLIGHT | 0 | 默认先校准正反例；1 跳过，报告中需说明原因 |
| RETRIEVAL_ONLY | BCP 为 0 | BCP 设 1 仅检查检索，不产生答案 accuracy；不等于完成语义评分 |

Hotpot 参数在共享 `54_judge_options.sh` 中映射；Plain/Beacon 通过 `58_offline_common.sh` 调用，原始模型通过 73 调用。BCP 原有参数映射保持原位置，避免重复传入。

本地 HF 裁判按需加载：先完成并释放待测模型，再评分；空回答/生成失败不会触发裁判调用。`NUM_WORKERS` 控制待测模型按题多卡并行，本地裁判本身串行调用并用 `device_map=auto` 分配可见 GPU。显存不足时可先生成、再在合适的卡上单独 score，不需要重做生成。

## 59.4 在线 BrowseComp 裁判变量

沿用现有 `49_benchmark_env.example.sh` 和 [49 本地辅助模型](49_local_aux_models.md)：

| 变量 | 用途 |
|---|---|
| SCORER_URLS | chat API base URL，可用逗号分隔多个兼容端点 |
| SCORER_MODEL_NAME | 服务接受的裁判模型 ID |
| SCORER_API_KEY | 服务认证密钥；不要写入仓库 |
| JUDGE_WORKERS | 在线评分并发，默认 1 |
| JUDGE_MAX_TOKENS | 默认 1024，裁判单次输出预算 |
| JUDGE_TIMEOUT | 默认 120 秒，单次 HTTP 超时 |
| JUDGE_ATTEMPTS / JUDGE_ENABLE_THINKING / SKIP_JUDGE_PREFLIGHT | 同一稳定评分规则 |
| ENV_FILE | 在线入口读取的私有 shell 配置路径 |

本地离线 `JUDGE_MODEL` 与在线 `SCORER_MODEL_NAME` 用途不同，不能只设置其中一组就认为两条路径均已配置。BrowseComp 的搜索/访问和 SUMMARY 服务仍需按既有手册准备。

## 59.5 仅重评分已有预测

```bash
METHOD=beacon BENCHMARK=hotpotqa ACTION=score LIMIT=8 \
  PREDICTIONS=outputs/results/YOUR_GENERATION/predictions.jsonl \
  OUTPUT_DIR=outputs/results/YOUR_NEW_SCORE \
  JUDGE_MODEL=/data/models/Qwen3-32B \
  bash scripts/sh/benchmark/59_run_benchmark.sh
```

使用与原生成一致的 ROOT_DIR、LIMIT 和题集；默认 provenance 校验会拒绝题目 checksum、ID 选择不同的预测。输出目录须为新目录，避免覆盖旧评分。仅评分通常无需 RUN_DIR 或 MODEL_PATH。

## 59.6 HotpotQA 裁判行为与结果解释

新增提示 `hotpot-answer-equivalence-v1` 要求：接受事实等价表述和别名，拒绝缺失、矛盾和实质错误；将问题、金标准和候选回答作为 JSON 数据输入；最终只允许 A/B。模型输出 JSON 时提取其 answer 字段，不把 sp 当作答案正文；无 JSON 则使用其回答文本。

1. 只有 `status=completed` 且答案非空才调用裁判。缺题、空回答和生成失败固定计错，保留在总题数分母。
2. 评分前对 Paris/Tokyo 正负例做自检；失败终止，避免把配置错误当作真实得分。
3. 有效 B 直接记错，不靠重试争取 A；模糊或冲突标签经过有限重试仍失败时保留 `score=null`。
4. 未知标签令 `judge_errors>0`、`complete=false`，最终评分命令非零退出；仍保存结果和完整分母。
5. `llm_judge_accuracy=有效正确题数/全部选定题数`，范围 0..1。出现裁判错误时这是未完成结果，不能当作最终有效成绩。

| 产物 | 说明 |
|---|---|
| summary.metrics.json | 官方指标、新增 llm_judge_accuracy、judge_errors、complete、策略/提示版本 |
| judge.jsonl | 每题 ID、问题、参考答案、候选回答、来源、A/B 分数或 null、每次尝试原文和错误 |
| judge_preflight.json | 正反例校准结果；全为空/失败时无需调用裁判 |
| per_question.json | 既有官方逐题 EM/F1 结果，继续保留 |
| official_hotpot.log / hotpot_predictions.json / hotpot_gold.json | 既有官方评分文件 |
| manifest.json | 输入、模型来源、裁判路径和运行参数等实验信息 |

## 59.7 修改提示与验证范围

变更判定原则时修改 `offline/scoring.py::score_hotpot_judge` 并升级提示版本；不覆盖官方 evaluator。变更共享裁判参数时修改 54；变更模型或 benchmark 路由时修改 59，不复制九套评分实现。

为保持官方 SHA256，`.gitattributes` 固定 `search_comp/offline/reference/**` 使用 LF，同时固定 shell 脚本 LF。Windows 如果已有 CRLF 旧副本，必须恢复与固定指纹一致的原文件；不要修改 provenance 来通过检查。

本次标准库测试验证数据适配与裁判控制流，使用模拟标签；不宣称已经验证真实裁判质量、Linux GPU 模型加载、CUDA 兼容性或端到端 benchmark 成绩。没有安装环境，没有启动训练或真实 LLM-Judge 推理。验证命令和结果见 [99](../project/99_validation.md)。
