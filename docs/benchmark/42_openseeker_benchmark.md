# 42 OpenSeeker 官方 Benchmark 对齐评测

当前保留训练原始 system、测试提示词来源选择、必要的离线适配及全部入口的保留范围见 [45 统一流程](../common/45_unified_agent_flow.md)。

`PROMPT_DATA` / `--prompt-data` 指定原训练 JSONL，仅取 system/schema；优先于 RUN_DIR 的 train_data_path，无两者时用内置原始 Polar profile。在线保留原文，离线仅适配必要能力和答案格式；多 system 必须显式选择单 profile，详情见 45。
本入口在现有训练、EM/F1、replay/corpus/http 和 Search-R1 流程之外增量添加。原有代码和编号入口继续可用。默认选择论文报告的英文 **BrowseComp**，它需要真实在线搜索和网页访问，适合评估训练后的复杂检索能力。它与从 Polar 训练轨迹切出的 test 集不同，不能互相代替。

官方来源：

- [OpenSeeker 官方仓库](https://github.com/PolarSeeker/OpenSeeker)，此次固定提交 `e3dec1f727d87c5eaa93506e4882c67237647098`。
- [固定版本生成与评分脚本](https://github.com/PolarSeeker/OpenSeeker/tree/e3dec1f727d87c5eaa93506e4882c67237647098/eval)。官方 README 的并发参数示例与实际生成脚本不一致，本项目按实际解析器使用 `--max_worker`，裁判则使用 `--max_workers`。
- [BrowseComp 官方数据加载与解密实现](https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py)。本项目采用相同的 canary → SHA256 重复密钥 → XOR → UTF-8 解码方式，分别解密 problem 与 answer。测试集不进入训练准备或长度筛选流程。

## 42.1 选择运行方式

| 模式 | 模型运行方式 | 工具、提示与生成 | 评分与适用范围 |
|---|---|---|---|
| `official-v2` | 已部署的 Plain 模型 completion 服务 | 原样执行官方 `generate_answer_v2.py`；search、visit、E2B；保留 boxed 格式指令 | 官方 `eval.py`；用于 v2 工具协议对比 |
| `official-v1` | 已部署的 Plain 模型 completion 服务 | 原样执行官方 `generate_answer.py`；search、visit | 同一官方裁判；适合只训练了 search/visit 的 checkpoint |
| `native`（默认） | 本项目直接加载 Plain / Beacon checkpoint | 与训练共享的 system prompt、官方 v1 Search 和 Visit；本项目序列化、默认贪心生成、停止和预算策略 | 官方裁判；保留 Beacon 压缩，必须注明生成侧差异 |

模型训练中出现的工具决定适用协议；不要因为目录名含 v2 就把仅训练 search/visit 的模型默认解释为掌握 E2B 工具。v2 需要单独配置 E2B。若对比压缩方法，建议 Plain 和 Beacon 均使用 native、相同预算与工具供应商，再用官方模式的 Plain 单独作为参考。

`native` 通过已有 `beacon_generate` 保持压缩，并使用既有 observation 区域定义。普通 SGLang 不执行本项目的 Beacon 压缩；48 服务入口会拒绝 Beacon，不能把它加载成普通模型来宣称 Beacon 测试结果。native 不实现官方 v2 沙箱工具，因此不称为完整复现 v2 生成流程。

## 42.2 准备依赖和数据

```bash
# 仅下载固定版本官方代码、下载解密 BrowseComp；不会加载模型或请求付费 API。
bash scripts/sh/benchmark/46_prepare_benchmark.sh

# 官方 API 评测建议在独立 Python 环境中安装；不修改训练依赖。
python -m pip install -r requirements.eval-openseeker.txt
# 或在独立环境安装固定官方仓库的完整 requirements.txt：
# python -m pip install -r outputs/vendor/OpenSeeker/requirements.txt
```

`46` 和 `47` 通过现有 `00_common.sh` 选择 Python：显式 `PYTHON` → `CONDA_ENV`（默认 search-comp-qwen35-cu126）→ 系统 Python。独立环境请设置 `PYTHON=/absolute/path/to/python`。native 需要本项目训练依赖和新增评测依赖；建议复制训练环境后安装评测依赖，避免官方工具依赖影响原训练环境。SGLang 另装到与目标模型、GPU/CUDA 匹配的服务环境，它不在新增依赖中强制安装。

默认产物：

- `outputs/vendor/OpenSeeker/`：官方代码的独立 checkout；版本不匹配或已跟踪文件被修改会拒绝运行，不重置用户目录。
- `outputs/data/benchmarks/browsecomp.jsonl`：只有 id、query、answer，按官方 CSV 顺序保留全量题目。
- `browsecomp.manifest.json`：下载来源、解密实现链接、原始 CSV / JSONL SHA256、题数、split 与顺序。

已有数据复用，首次运行 `47` 也可自动准备。已有 preparation manifest 的 JSONL 若指纹改变会拒绝评测。本次实际下载为 **1,266 题**。无网络环境可在联网机器先运行 46，再拷贝代码目录、JSONL 与 manifest；也可使用：

```bash
bash scripts/sh/benchmark/46_prepare_benchmark.sh --csv /data/browse_comp_test_set.csv
# --official-dir 可指定另一个处于固定提交且未修改的官方 checkout
```

`--data` 支持已有明文 JSONL，字段为 `query`（或 `question`）和非空字符串 `answer`，可选 id。重复问题、空数据、缺参考答案会拒绝。适配器仅保留这些字段，不把 messages、参考轨迹或 evidence 加入推理。BrowseComp-ZH、xbench 等可由用户从官方来源准备成此格式，再填写 `BENCHMARK`；本次只实测 BrowseComp 数据导入，不声称已适配其他基准的专用指标。

## 42.3 配置 API：从哪里获取、如何填写

这些变量配置四类独立服务：**待测模型**生成回答，**搜索与网页读取**提供外部信息，**摘要模型**整理 visit 的网页内容，**裁判模型**判断最终回答是否正确。环境变量名中的 MODEL 是 API 的模型标识，KEY 是对应服务的认证凭据，URL 是请求地址；它们不是从 Benchmark 数据集中获取的。

把 `scripts/sh/benchmark/49_benchmark_env.example.sh` 复制到仓库外的私有路径，填写后设置 `ENV_FILE`；也可先 source 私有文件，再运行 47。49 使用 `${VAR:-默认值}`：已有非空环境变量优先，只有未设置或空值才采用脚本默认值；不要以为改了默认值就一定覆盖终端里已有的配置。私有文件也可改用普通 `export VAR='实际值'` 直接赋值。

```bash
mkdir -p "$HOME/.config/openseeker"
cp scripts/sh/benchmark/49_benchmark_env.example.sh "$HOME/.config/openseeker/benchmark_env.sh"
chmod 600 "$HOME/.config/openseeker/benchmark_env.sh"
# 编辑这个副本，替换实际需要的 YOUR_... 占位符。
# 后文运行：ENV_FILE="$HOME/.config/openseeker/benchmark_env.sh" ...
```

### 42.3.1 每个变量的获取方式

| 变量 | 从哪里获取 | 填写案例与含义 |
|---|---|---|
| `OPENSEEKER_BASE_URL` | 启动 48 的待测 Plain 模型服务后，从启动输出获得 | `http://127.0.0.1:30018/v1/completions`；这里是 **completion 完整接口** |
| `OPENSEEKER_MODEL` | 48 打印的模型路径，或服务端配置的模型名 | `/data/models/plain-run/final`；必须匹配服务实际接受的 model，不是随意写的实验简称 |
| `SEARCH_PROVIDER` | 自己选择搜索供应商 | `serper` 或 `tavily`；两者选一个即可 |
| `SERPER_KEY_ID` | 登录 [Serper 官网](https://serper.dev/)，在账号控制台中查找并复制 API Key | 填完整密钥字符串；虽然名称有 ID，这里需要 **API Key 本身**，不是账号 ID |
| `TAVILY_API_KEY` | 登录 [Tavily 控制台](https://app.tavily.com/)，在 API Keys 中获取或创建密钥 | 仅 `SEARCH_PROVIDER=tavily` 时使用；[官方说明](https://www.tavily.com/pricing)提供获取 API Key 的入口 |
| `JINA_API_KEYS` | 在 [Jina Key Manager](https://jina.ai/api-dashboard/key-manager) 获取 API Key，见 [官方 API 说明](https://docs.jina.ai/) | 填一条完整密钥；当前固定版本把此变量作为单个 Bearer token，**不要填逗号分隔的多条 key** |
| `SUMMARY_API_URL` | 自部署摘要模型服务的地址，或所用模型供应商文档中的兼容 API base URL | `http://127.0.0.1:8890/v1`；SDK 会追加 `/chat/completions`，这里不再写该后缀 |
| `SUMMARY_MODEL_NAME` | 从摘要服务的 `/v1/models` 返回值或供应商模型列表获得 | 例如自部署服务注册的 `judge-summary`；不能用尚未部署的本地权重路径代替服务模型名 |
| `SUMMARY_API_KEY` | 摘要服务管理员设置的认证 key，或供应商控制台创建的 API Key | 服务确实无需认证时填 `EMPTY`；有认证时填真实 key |
| `SCORER_URLS` | 自部署裁判服务或供应商提供的兼容 API base URL | `http://127.0.0.1:8890/v1`；同样不加 `/chat/completions` |
| `SCORER_MODEL_NAME` | 裁判服务 `/v1/models` 返回值或供应商模型列表 | 必须与实际服务模型一致；可与 SUMMARY_MODEL_NAME 相同 |
| `SCORER_API_KEY` | 裁判服务的认证配置或供应商控制台 | 无认证时填 `EMPTY`；可与摘要 key 相同，但两类服务共用认证时才这样填写 |
| `E2B_API_KEY` | 注册 E2B 后，在 [E2B 控制台的 API Keys](https://console.e2b.dev/?tab=keys) 获取，见 [官方 Quickstart](https://docs.e2b.dev/quickstart) | 仅 official-v2 需要；它是云沙箱密钥，不是本机 SSH 密钥 |

`SERPER_API_KEY` 是官方也支持的另一个变量名，而且优先于 `SERPER_KEY_ID`。如果终端里已设置了前者，应保证两者一致，或只保留一种，避免意外使用旧 key。

各平台的模型权限、额度和价格以账号控制台为准。Serper、Tavily、Jina、E2B 的 key 不能相互替代；不能用 `EMPTY` 跳过真实第三方服务的认证。Jina Reader 本身支持无 key 的受限访问，但本项目当前预检查要求配置 JINA_API_KEYS，因此按此流程仍应获取 key。

### 42.3.2 不同模式需要填写哪些组

| 场景 | OPENSEEKER 两项 | 所选搜索 key + JINA | SUMMARY 三项 | SCORER 三项 | E2B |
|---|---|---|---|---|---|
| official-v1 在线生成并评分 | 需要 | 需要 | 需要 | 需要 | 不需要 |
| official-v2 在线生成并评分 | 需要 | 需要 | 需要 | 需要 | 需要 |
| native（Plain / Beacon）在线生成并评分 | 不需要 | 需要 | 需要 | 需要 | 不需要 |
| 设置 PREDICTIONS，仅评分已有回答 | 不需要 | 不需要 | 不需要 | 需要 | 不需要 |
| 46 准备数据或 DRY_RUN=1 | 不需要 | 不需要 | 不需要 | 不需要 | 不需要 |

未使用的组可以保留 YOUR_... 占位符。本项目环境加载不主动打印密钥；上游服务的异常日志可能含认证信息片段，分享日志前应检查。

### 42.3.3 案例 A：Beacon native，摘要与裁判共用本地服务

假设你已在**运行 47 的同一节点**部署一个兼容 chat completions 的辅助模型服务：端口 8890，服务模型名为 `judge-summary`，无需认证。下面的名称和端口只是配置案例，写入变量不会自动创建这个服务。可以向管理员索取已有服务，也可以自己部署一个支持此接口的模型。

编辑私有环境文件，主要配置如下（所有 key 都是占位符，需要替换）：

```bash
export SEARCH_PROVIDER='serper'
export SERPER_KEY_ID='YOUR_REAL_SERPER_KEY'
export JINA_API_KEYS='YOUR_REAL_JINA_KEY'

# 这里的 judge-summary 必须是服务实际接受的 model。
export SUMMARY_API_URL='http://127.0.0.1:8890/v1'
export SUMMARY_MODEL_NAME='judge-summary'
export SUMMARY_API_KEY='EMPTY'
export SCORER_URLS="$SUMMARY_API_URL"
export SCORER_MODEL_NAME="$SUMMARY_MODEL_NAME"
export SCORER_API_KEY="$SUMMARY_API_KEY"
# native 不需要 OPENSEEKER_* 和 E2B_API_KEY。
```

训练后的 Beacon 仍由本项目在当前进程加载；上述辅助模型只负责网页摘要和裁判，不是被测 Beacon。SUMMARY 和 SCORER 可共用服务以减少部署数量，也可分别使用两个模型。模型选择会影响结果；正式对比时记录摘要模型与裁判模型，不能据此宣称与论文完全相同。

```bash
ENV_FILE="$HOME/.config/openseeker/benchmark_env.sh" \
MODE=native RUN_DIR=outputs/experiments/<beacon-run> LIMIT=2 \
JUDGE_WORKERS=1 bash scripts/sh/benchmark/47_openseeker_benchmark.sh
```

这个案例仍然需要通过 Serper/Jina 访问外网，并不是完全离线评测。

### 42.3.4 案例 B：Plain 官方模式，待测模型与辅助模型分别部署

先用 48 部署训练后的 Plain 模型（例如端口 30018），再在案例 A 的私有文件中增加：

```bash
export OPENSEEKER_BASE_URL='http://127.0.0.1:30018/v1/completions'
# 将此值替换成 48 实际打印的 OPENSEEKER_MODEL。
export OPENSEEKER_MODEL='/data/models/plain-run/final'
# 选择 official-v2 时再填写 E2B；official-v1 不需要。
export E2B_API_KEY='YOUR_REAL_E2B_KEY'
```

```bash
ENV_FILE="$HOME/.config/openseeker/benchmark_env.sh" \
MODE=official-v1 LIMIT=2 WORKERS=1 JUDGE_WORKERS=1 \
  bash scripts/sh/benchmark/47_openseeker_benchmark.sh
# 已配置 E2B 且模型适合 v2 工具时，将 MODE 改成 official-v2。
```

如果改用 Tavily，在私有文件中设置 `SEARCH_PROVIDER=tavily` 并填 `TAVILY_API_KEY`；Serper key 就不再需要。切换供应商会改变搜索环境，比较实验时应固定选择。

### 42.3.5 案例 C：使用外部摘要/裁判服务

从供应商或管理员处获取 **base URL、模型 ID、API Key 三项**。下面的 `api.example.com` 仅用于展示格式，不是可用的供应商地址：

```bash
export SUMMARY_API_URL='https://api.example.com/v1'
export SUMMARY_MODEL_NAME='actual-model-id-from-provider'
export SUMMARY_API_KEY='YOUR_REAL_PROVIDER_KEY'
export SCORER_URLS="$SUMMARY_API_URL"
export SCORER_MODEL_NAME="$SUMMARY_MODEL_NAME"
export SCORER_API_KEY="$SUMMARY_API_KEY"
```

无需本地部署摘要/裁判模型，但要确认服务支持 OpenAI 兼容的 `/chat/completions` 请求。固定官方裁判还发送 `skip_special_tokens=false` 扩展参数；有些供应商虽兼容常规 chat API，却会拒绝此参数，应先用 1—2 题确认兼容性。此流程不会自动删除该参数，因为需要保留官方评分请求。

`SCORER_URLS` 可写成 `http://node-a:8890/v1,http://node-b:8890/v1`；官方会随机选端点。多个端点共用同一 SCORER_MODEL_NAME 和 SCORER_API_KEY，应确保都接受它们；不能把不同模型混进列表却当成固定裁判。

### 42.3.6 HPC 节点、连通检查与常见错误

在 HPC 环境中，`127.0.0.1` 指**执行进程所在节点**。若 47 在 gpu-node-02，摘要服务在 gpu-node-01，就应写 `http://gpu-node-01:8890/v1`，不能继续写 127.0.0.1。服务端也须监听可被其他节点访问的地址，并符合集群网络规则。48 默认监听 127.0.0.1；确需跨节点时可传 `--host 0.0.0.0`，客户端 URL 填实际服务节点名/IP，不能填 0.0.0.0。

以下从**实际运行评测的计算节点**执行；只列模型检查不会调用生成：

```bash
source "$HOME/.config/openseeker/benchmark_env.sh"
# 返回 data[].id，从中确认 SUMMARY_MODEL_NAME。
curl --fail --connect-timeout 10 --max-time 30 \
  -H "Authorization: Bearer $SUMMARY_API_KEY" "${SUMMARY_API_URL%/}/models"
# SCORER_URLS 可能有多个端点，这里只检查第一个。
SCORER_FIRST_URL="${SCORER_URLS%%,*}"
curl --fail --connect-timeout 10 --max-time 30 \
  -H "Authorization: Bearer $SCORER_API_KEY" "${SCORER_FIRST_URL%/}/models"
```

部分服务不提供 `/models`：可由服务管理员确认 model ID，再以 1—2 题 smoke 检查实际调用。`DRY_RUN=1` 只打印计划，不检查密钥有效性、模型服务可达性或真实权限。

| 现象 | 首先检查 |
|---|---|
| configure environment variables | 缺失或仍为 YOUR_... 的变量；是否给 47 设置 ENV_FILE |
| connection refused / timeout | 服务是否已启动、端口与监听地址、登录节点和计算节点是否不同、计算节点能否访问外网 |
| 401 / 403 | key 属于哪家服务、是否已过期/无权限；EMPTY 只适用于真实无认证服务 |
| 404 / model not found | SUMMARY/SCORER 是否误加了完整 chat/completions 后缀；model 是否与服务一致 |
| 429 | 账号额度/限流；先降低 WORKERS 和 JUDGE_WORKERS |
| unknown argument skip_special_tokens | 裁判服务不接受官方扩展参数，需使用兼容端点 |

不自动安装依赖、不自动启动模型服务或注册供应商账号。真实 47 运行会调用已配置的在线服务；46 与 DRY_RUN 不调用这些 API。

## 42.4 训练后的操作

Plain：训练完成后，在服务环境的终端 A 启动模型；RUN_DIR 与现有 20 训练产物相连，自动读取 `models/model/train_summary.json` 的 `final_model_path`。

```bash
RUN_DIR=outputs/experiments/<plain-run> bash scripts/sh/benchmark/48_serve_checkpoint.sh
# 或 MODEL_PATH=/absolute/path/to/merged-model bash scripts/sh/benchmark/48_serve_checkpoint.sh
```

48 使用参数数组启动 `python -m sglang.launch_server`，不执行 eval 字符串。默认 GPU=0、TP=1、context=131072、port=30018；用同名环境变量或编辑脚本顶部修改。先等服务就绪，再在终端 B 运行：

```bash
# 先审查配置，不创建目录、不下载数据、不调用 API：
DRY_RUN=1 bash scripts/sh/benchmark/47_openseeker_benchmark.sh

# 用私有环境文件配置模型服务和工具；正式运行默认全量、官方 v2：
ENV_FILE=/private/path/env.sh bash scripts/sh/benchmark/47_openseeker_benchmark.sh

# 仅 search/visit 训练模型可改为官方 v1；先做 2 题在线 smoke：
MODE=official-v1 LIMIT=2 ENV_FILE=/private/path/env.sh bash scripts/sh/benchmark/47_openseeker_benchmark.sh
```

自动化也可用 `bash scripts/sh/tools/02_workflow.sh benchmark [47 的命令行参数]`，它复用同一 Python 入口。

`OPENSEEKER_MODEL` 应与服务中的模型名称相符（48 会打印模型路径），并指向本次训练后的模型，而非原基座。官方模式从 API 推理，不接收 RUN_DIR/MODEL_PATH；这两个变量用于 48，或 47 的 native 模式。

Beacon：完成 30 的训练及必要的 LoRA 合并后，直接运行，不需要 SGLang 服务：

```bash
MODE=native RUN_DIR=outputs/experiments/<beacon-run> \
  ENV_FILE=/private/path/env.sh bash scripts/sh/benchmark/47_openseeker_benchmark.sh
# 没有 RUN_DIR 时：MODE=native MODEL_PATH=/path/to/checkpoint KIND=beacon ...
```

目录仍是 adapter（含 `adapter_config.json`）会拒绝；按已有 `35_beacon_merge_adapter.sh` 合并。显式 MODEL_PATH 覆盖实验记录；KIND 默认读取实验配置或模型的 enable_beacon。训练本身的测试集路径不用于此独立 Benchmark。

若已用其他方式获得 Benchmark 预测，可只跑裁判：

```bash
MODE=native DATA=/data/benchmark.jsonl BENCHMARK=BrowseComp \
  PREDICTIONS=/data/predictions.jsonl ENV_FILE=/private/path/judge-env.sh \
  bash scripts/sh/benchmark/47_openseeker_benchmark.sh
```

预测必须含 `query/answer/final_response`，问题和参考答案必须匹配 DATA；仅有现有 21/31 输出的 prediction 字段还不够，需根据原始 Benchmark 显式导出 query，不能直接把 Polar test 的预测冒充 BrowseComp。若输入来自官方 v2，选择 `MODE=official-v2`，保留官方添加的 boxed 问题后缀。manifest 会记录导入预测的指纹，但其生成来源仍由使用者负责说明。

## 42.5 参数、指标与产物

| 参数 | 默认值 | 语义 |
|---|---|---|
| MODE | native | 上表三种模式 |
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
| `generation.log`、上游日志 | 官方生成 stdout/stderr、上游统计与逐题完成计数；native 进度输出到终端，轨迹逐题落盘 |
| `result_for_judge.jsonl` | 统一评分输入，按基准顺序每题一条，重试同题使用最后一条；缺失生成填空回答并标记失败 |
| `coverage.json` | 原始尝试数、唯一预测数、漏题数、空回答数 |
| `result_for_judge_eval.jsonl` | 官方裁判结果与官方 summary |
| `judge.log`、`summary.metrics.json` | 裁判日志与增加覆盖审计后的汇总 |

生成和评分分别实时输出 `done=已完成/实际题数`。例如 `LIMIT=8` 时，官方生成显示 `[benchmark BrowseComp generation] done=1/8; elapsed=...s`，裁判显示 `[benchmark BrowseComp judge] done=1/8; elapsed=...s`；这些记录也写入对应的 `.log`。重试同题只计一次，计数表示已有结果，不代表回答正确或裁判标签有效。

`official_accuracy` 沿用官方已得到有效 0/1 标签的分母；`accuracy_all_questions` 以所有选定题目为分母，未知/错误裁判按 0 计入，作为覆盖审计补充指标。成功完整评分时两者一致。上游可能过滤裁判异常条目；因此本入口重新检查 item_index 覆盖率，缺任何裁判结果都会写出汇总并以非零状态结束，不能当成完整成绩。生成缺失题保留空回答交给同一官方裁判，不从分母删除。v2 的 boxed 后缀通过读取固定官方源码的常量处理，不修改其提示。

已有输出目录一律拒绝覆盖，避免上游按 item_index 恢复评分时串用旧文件。修复 API 后用新目录重跑，或用原始预测文件走仅评分模式。官方 checkout 的源码修改或版本变化也会拒绝；升级时应修改固定提交、对齐测试和说明，再建立新实验。

同一个 Benchmark、同一个模型也可能因搜索时间、供应商、摘要模型、裁判模型、解码默认值和上下文预算产生差异。Plain 的采样、Plain/Beacon 的重复惩罚范围见 [23](../common/23_decoding.md)，官方模式不接受这些非默认覆盖。报告分数时同时报告这些配置、样本数量、覆盖率、MODE 和固定版本。不能只因接上官方脚本，就把本地结果称为已复现论文成绩。

## 42.6 不依赖联网 API 的独立固定语料流程

56 BrowseComp-Plus 和 57 HotpotQA 已完整接入数据适配、持久化 BM25 与本地评分。它们不读取 49 的 API 配置，且不通过 47 的在线 search/visit 运行。BCP 完整评分使用本地裁判，Hotpot 使用无需 LLM 的官方 EM/F1。离线流程参数、准备与比较边界见 [43 离线手册](43_offline_benchmarks.md)。
