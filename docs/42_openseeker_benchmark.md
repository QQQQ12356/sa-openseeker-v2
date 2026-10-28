# 42 OpenSeeker 官方 Benchmark 对齐评测

本入口在现有训练、EM/F1、replay/corpus/http 和 Search-R1 流程之外增量添加。原有代码和编号入口继续可用。默认选择论文报告的英文 **BrowseComp**，它需要真实在线搜索和网页访问，适合评估训练后的复杂检索能力。它与从 Polar 训练轨迹切出的 test 集不同，不能互相代替。

官方来源：

- [OpenSeeker 官方仓库](https://github.com/PolarSeeker/OpenSeeker)，此次固定提交 `e3dec1f727d87c5eaa93506e4882c67237647098`。
- [固定版本生成与评分脚本](https://github.com/PolarSeeker/OpenSeeker/tree/e3dec1f727d87c5eaa93506e4882c67237647098/eval)。官方 README 的并发参数示例与实际生成脚本不一致，本项目按实际解析器使用 `--max_worker`，裁判则使用 `--max_workers`。
- [BrowseComp 官方数据加载与解密实现](https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py)。本项目采用相同的 canary → SHA256 重复密钥 → XOR → UTF-8 解码方式，分别解密 problem 与 answer。测试集不进入训练准备或长度筛选流程。

## 42.1 选择运行方式

| 模式 | 模型运行方式 | 工具、提示与生成 | 评分与适用范围 |
|---|---|---|---|
| `official-v2`（默认） | 已部署的 Plain 模型 completion 服务 | 原样执行官方 `generate_answer_v2.py`；search、visit、E2B；保留 boxed 格式指令 | 官方 `eval.py`；用于 v2 工具协议对比 |
| `official-v1` | 已部署的 Plain 模型 completion 服务 | 原样执行官方 `generate_answer.py`；search、visit | 同一官方裁判；适合只训练了 search/visit 的 checkpoint |
| `native` | 本项目直接加载 Plain / Beacon checkpoint | 官方 v1 system prompt、Search 和 Visit；本项目序列化、贪心生成、停止和预算策略 | 官方裁判；保留 Beacon 压缩，必须注明生成侧差异 |

模型训练中出现的工具决定适用协议；不要因为目录名含 v2 就把仅训练 search/visit 的模型默认解释为掌握 E2B 工具。v2 需要单独配置 E2B。若对比压缩方法，建议 Plain 和 Beacon 均使用 native、相同预算与工具供应商，再用官方模式的 Plain 单独作为参考。

`native` 通过已有 `beacon_generate` 保持压缩，并使用既有 observation 区域定义。普通 SGLang 不执行本项目的 Beacon 压缩；48 服务入口会拒绝 Beacon，不能把它加载成普通模型来宣称 Beacon 测试结果。native 不实现官方 v2 沙箱工具，因此不称为完整复现 v2 生成流程。

## 42.2 准备依赖和数据

```bash
# 仅下载固定版本官方代码、下载解密 BrowseComp；不会加载模型或请求付费 API。
bash scripts/sh/46_prepare_benchmark.sh

# 官方 API 评测建议在独立 Python 环境中安装；不修改训练依赖。
python -m pip install -r requirements.eval-openseeker.txt
# 或在独立环境安装固定官方仓库的完整 requirements.txt：
# python -m pip install -r outputs/vendor/OpenSeeker/requirements.txt
```

`46` 和 `47` 通过现有 `00_common.sh` 选择 Python：显式 `PYTHON` → `CONDA_ENV`（默认 search-comp-qwen3.5）→ 系统 Python。独立环境请设置 `PYTHON=/absolute/path/to/python`。native 需要本项目训练依赖和新增评测依赖；建议复制训练环境后安装评测依赖，避免官方工具依赖影响原训练环境。SGLang 另装到与目标模型、GPU/CUDA 匹配的服务环境，它不在新增依赖中强制安装。

默认产物：

- `outputs/vendor/OpenSeeker/`：官方代码的独立 checkout；版本不匹配或已跟踪文件被修改会拒绝运行，不重置用户目录。
- `outputs/data/benchmarks/browsecomp.jsonl`：只有 id、query、answer，按官方 CSV 顺序保留全量题目。
- `browsecomp.manifest.json`：下载来源、解密实现链接、原始 CSV / JSONL SHA256、题数、split 与顺序。

已有数据复用，首次运行 `47` 也可自动准备。已有 preparation manifest 的 JSONL 若指纹改变会拒绝评测。本次实际下载为 **1,266 题**。无网络环境可在联网机器先运行 46，再拷贝代码目录、JSONL 与 manifest；也可使用：

```bash
bash scripts/sh/46_prepare_benchmark.sh --csv /data/browse_comp_test_set.csv
# --official-dir 可指定另一个处于固定提交且未修改的官方 checkout
```

`--data` 支持已有明文 JSONL，字段为 `query`（或 `question`）和非空字符串 `answer`，可选 id。重复问题、空数据、缺参考答案会拒绝。适配器仅保留这些字段，不把 messages、参考轨迹或 evidence 加入推理。BrowseComp-ZH、xbench 等可由用户从官方来源准备成此格式，再填写 `BENCHMARK`；本次只实测 BrowseComp 数据导入，不声称已适配其他基准的专用指标。

## 42.3 配置 API

把 `scripts/sh/49_benchmark_env.example.sh` 复制到仓库外的私有路径，填写值，然后设置 `ENV_FILE=/private/path/env.sh`；也可直接 export 环境变量。脚本不会打印密钥。无需认证的兼容 API 仍需填 `EMPTY`。

| 变量 | 作用 |
|---|---|
| `OPENSEEKER_BASE_URL`、`OPENSEEKER_MODEL` | 官方生成使用的模型 completion URL 和模型标识；native 无需这两项 |
| `SEARCH_PROVIDER` | `serper`（默认）或 `tavily` |
| `SERPER_KEY_ID`（也接受官方的 `SERPER_API_KEY`）/ `TAVILY_API_KEY` | 所选搜索供应商密钥 |
| `JINA_API_KEYS` | 网页读取 |
| `SUMMARY_API_URL`、`SUMMARY_MODEL_NAME`、`SUMMARY_API_KEY` | visit 的目标相关摘要模型 |
| `SCORER_URLS`、`SCORER_MODEL_NAME`、`SCORER_API_KEY` | 官方 A/B 裁判；URL 支持逗号分隔 |
| `E2B_API_KEY` | 仅 official-v2 的沙箱工具所需 |

不自动安装依赖、不自动调用训练脚本，也不设置供应商账号。真实 47 运行会调用已配置的在线服务；46 与 `DRY_RUN=1` 不调用这些服务。仅评分已有预测时只要求裁判配置。

## 42.4 训练后的操作

Plain：训练完成后，在服务环境的终端 A 启动模型；RUN_DIR 与现有 20 训练产物相连，自动读取 `models/model/train_summary.json` 的 `final_model_path`。

```bash
RUN_DIR=outputs/experiments/<plain-run> bash scripts/sh/48_serve_checkpoint.sh
# 或 MODEL_PATH=/absolute/path/to/merged-model bash scripts/sh/48_serve_checkpoint.sh
```

48 使用参数数组启动 `python -m sglang.launch_server`，不执行 eval 字符串。默认 GPU=0、TP=1、context=131072、port=30018；用同名环境变量或编辑脚本顶部修改。先等服务就绪，再在终端 B 运行：

```bash
# 先审查配置，不创建目录、不下载数据、不调用 API：
DRY_RUN=1 bash scripts/sh/47_openseeker_benchmark.sh

# 用私有环境文件配置模型服务和工具；正式运行默认全量、官方 v2：
ENV_FILE=/private/path/env.sh bash scripts/sh/47_openseeker_benchmark.sh

# 仅 search/visit 训练模型可改为官方 v1；先做 2 题在线 smoke：
MODE=official-v1 LIMIT=2 ENV_FILE=/private/path/env.sh bash scripts/sh/47_openseeker_benchmark.sh
```

自动化也可用 `bash scripts/sh/02_workflow.sh benchmark [47 的命令行参数]`，它复用同一 Python 入口。

`OPENSEEKER_MODEL` 应与服务中的模型名称相符（48 会打印模型路径），并指向本次训练后的模型，而非原基座。官方模式从 API 推理，不接收 RUN_DIR/MODEL_PATH；这两个变量用于 48，或 47 的 native 模式。

Beacon：完成 30 的训练及必要的 LoRA 合并后，直接运行，不需要 SGLang 服务：

```bash
MODE=native RUN_DIR=outputs/experiments/<beacon-run> \
  ENV_FILE=/private/path/env.sh bash scripts/sh/47_openseeker_benchmark.sh
# 没有 RUN_DIR 时：MODE=native MODEL_PATH=/path/to/checkpoint KIND=beacon ...
```

目录仍是 adapter（含 `adapter_config.json`）会拒绝；按已有 `35_beacon_merge_adapter.sh` 合并。显式 MODEL_PATH 覆盖实验记录；KIND 默认读取实验配置或模型的 enable_beacon。训练本身的测试集路径不用于此独立 Benchmark。

若已用其他方式获得 Benchmark 预测，可只跑裁判：

```bash
MODE=native DATA=/data/benchmark.jsonl BENCHMARK=BrowseComp \
  PREDICTIONS=/data/predictions.jsonl ENV_FILE=/private/path/judge-env.sh \
  bash scripts/sh/47_openseeker_benchmark.sh
```

预测必须含 `query/answer/final_response`，问题和参考答案必须匹配 DATA；仅有现有 21/31 输出的 prediction 字段还不够，需根据原始 Benchmark 显式导出 query，不能直接把 Polar test 的预测冒充 BrowseComp。若输入来自官方 v2，选择 `MODE=official-v2`，保留官方添加的 boxed 问题后缀。manifest 会记录导入预测的指纹，但其生成来源仍由使用者负责说明。

## 42.5 参数、指标与产物

| 参数 | 默认值 | 语义 |
|---|---|---|
| MODE | official-v2 | 上表三种模式 |
| LIMIT | -1 | 全量；正数取原始顺序前 N 题，属于子集 smoke |
| WORKERS / JUDGE_WORKERS | 60 / 20 | 官方生成异步任务数 / 裁判线程数；native 逐题运行，WORKERS 不控制 GPU 副本 |
| MAX_TOKENS / TOOL_COUNT | 32768 / 200 | 官方生成脚本的 `max_tokens` / `tool_count_max` |
| MAX_CONTEXT | 131072 | native 完整未压缩序列预算；Beacon 压缩后也保留此逻辑预算 |
| MAX_TURNS / MAX_TOOL_TOKENS | 200 / 32768 | native 最大轮数 / 单条工具回复 token 上限 |
| OUTPUT_DIR | 自动生成新目录 | native 带 RUN_DIR 时落在实验 evaluation 下，否则 outputs/results 下 |

native 的 max_tokens 是每轮生成上限，max_turns 是轮数，不等于官方工具调用次数；TOOL_COUNT 不控制 native，MAX_CONTEXT/MAX_TURNS/MAX_TOOL_TOKENS 不控制官方脚本。官方模式保留上游停止、重试、解码和工具行为（不自行设置 temperature）；服务端默认值也影响结果，应随实验一起记录。48 服务默认上下文为 131072，官方部署示例为 256000，若要采用示例预算应显式设置 `CONTEXT_LENGTH=256000` 并确认模型支持。当前默认值不是论文全部实验设置的证明。

评分原样调用固定版本 `eval/eval.py`：英文 `JUDGE_PROMPT_BC_en`、system 为客观评判要求、temperature=0、`skip_special_tokens=False`，A 为正确、B 为错误。沿用上游对 `</think>` 等包装和 A/B 标签的解析，不用字符串 EM/F1 替代语义裁判。此口径是 **OpenSeeker 的 BrowseComp 裁判**，与 OpenAI simple-evals 自己的 grader prompt 不同。

新增运行目录包含：

| 文件 | 内容 |
|---|---|
| `questions.jsonl` | 本次选定的题目/参考答案，不含参考轨迹 |
| `manifest.json` | 模式、官方提交及脚本指纹、数据指纹、配置、裁判模型、搜索供应商、模型或导入预测来源；不保存密钥 |
| `result_tool200.jsonl`（随 TOOL_COUNT 改名） | 官方原始生成产物，保留轨迹、工具次数与耗时；native 保留 messages/events/status |
| `generation.log`、上游日志 | 官方生成 stdout/stderr 与上游统计；native 进度输出到终端，轨迹逐题落盘 |
| `result_for_judge.jsonl` | 统一评分输入，按基准顺序每题一条，重试同题使用最后一条；缺失生成填空回答并标记失败 |
| `coverage.json` | 原始尝试数、唯一预测数、漏题数、空回答数 |
| `result_for_judge_eval.jsonl` | 官方裁判结果与官方 summary |
| `judge.log`、`summary.metrics.json` | 裁判日志与增加覆盖审计后的汇总 |

`official_accuracy` 沿用官方已得到有效 0/1 标签的分母；`accuracy_all_questions` 以所有选定题目为分母，未知/错误裁判按 0 计入，作为覆盖审计补充指标。成功完整评分时两者一致。上游可能过滤裁判异常条目；因此本入口重新检查 item_index 覆盖率，缺任何裁判结果都会写出汇总并以非零状态结束，不能当成完整成绩。生成缺失题保留空回答交给同一官方裁判，不从分母删除。v2 的 boxed 后缀通过读取固定官方源码的常量处理，不修改其提示。

已有输出目录一律拒绝覆盖，避免上游按 item_index 恢复评分时串用旧文件。修复 API 后用新目录重跑，或用原始预测文件走仅评分模式。官方 checkout 的源码修改或版本变化也会拒绝；升级时应修改固定提交、对齐测试和说明，再建立新实验。

同一个 Benchmark、同一个模型也可能因搜索时间、供应商、摘要模型、裁判模型、解码默认值和上下文预算产生差异。报告分数时同时报告这些配置、样本数量、覆盖率、MODE 和固定版本。不能只因接上官方脚本，就把本地结果称为已复现论文成绩。
