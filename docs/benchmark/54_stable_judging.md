# 54 模型评分稳定性：历史审计、裁判自检、重评与工具恢复

2026-11-03：离线 HotpotQA 新增本地答案语义裁判，复用本章自检、标签校验与有限重试；保留官方指标，新增 `llm_judge_accuracy`。全部变量和九种组合见 [59](59_llm_judge_matrix.md)。

## 54.1 本次远端结果说明与处理顺序

提供的 `tmp/benchmark_native_20261101T091822637884Z` 是导入预测后重新评分的 12 题子集。已对齐 12 条预测、12 个唯一 item_index，无缺失预测；11 条最终回答为空。原生成状态为 completed=1、invalid_call=10、turn_limit=1。原裁判把 item_index=8（第 9 题，参考答案 White Mischief）的空回答判成 A，导致 1/12=8.33%。唯一正常回答为 1985–1986，当前参考答案是 1988–96，按现有标签应判错。

此次先做离线审计，再补裁判自检，最后改进后续生成的格式恢复和 visit。旧目录不改、不重跑训练；审计只将空回答固定计错，修正后为 0/12=0%。这不代表重新核实了非空回答的事实，也不代表完整 BrowseComp 得分。原数据与裁判文件 SHA256 写入新审计结果，保留原 judge_raw 和 original_score。

本地审计生成目录：`outputs/results/judge_audit_20261101T093424349679Z`。该目录是本地输出、不入 Git；远端使用下面命令重新审计即可。

## 54.2 各方法、各 benchmark 的覆盖范围

| 路线 | 方法与入口 | 本次行为 |
|---|---|---|
| 在线 BrowseComp / 导入兼容 query-answer-final_response 数据 | Plain/Beacon 47、原始 Qwen3.5 72 | 共用 stable-judge-v1：空回答 0、正反例自检、严格标签、超时/截断校验、有限重试、完整分母 |
| official-v1/v2 在线 BrowseComp | Plain 47，显式 MODE | 官方生成脚本不变；评分改为固定官方提示 + 本项目稳定评分器，不能再称裁判请求与上游完全相同 |
| 离线 BrowseComp-Plus | Plain/Beacon 56、原始基座 73/74 | 本地 HF 裁判，空/失败预测 0；默认贪心、关闭思考；自检、有限重试、严格字段验证 |
| 离线 HotpotQA | Plain/Beacon 57、原始基座 73/75 | 保留官方 EM/F1 / supporting-fact 指标；默认增加本地语义裁判、自检与有界重试，见 [59](59_llm_judge_matrix.md) |
| 原有 OpenSeeker 21/31 与原始基座 70/71 的 EM/F1 | Plain/Beacon/基座 | 确定性评分保持不变，没有凭空引入模型裁判 |
| 后续 native 在线/离线生成 | 47/72、56/57/73/74/75 | 默认每题最多一次工具调用格式反馈；在线 native 另有可诊断的 bounded visit |

稳定性来自明确规则与失败检测，不能保证模型裁判对所有非空回答语义判断正确。新旧模型、thinking、sampling、工具恢复或 visit 策略不同，必须分别记录，不混称完全相同评测协议。

## 54.3 第一步：离线审计已有评分

仅需 Python 和本项目，不启动模型、不请求 API。将远端完整评分目录复制过来或直接在远端执行：

```bash
bash scripts/sh/benchmark/54_check_judge.sh audit \
  --result-dir tmp/benchmark_native_20261101T091822637884Z \
  --output-dir outputs/results/rescore_audit_001
```

输入要求目录包含 `result_for_judge.jsonl`、`result_for_judge_eval.jsonl`。审计检查 index 唯一性与范围，以及逐条 query、参考答案、final_response 一致性。数据不匹配会报错，不创建新目录；已有 OUTPUT_DIR 拒绝覆盖。

输出 `result_for_judge_eval.jsonl` 和 `summary.metrics.json`。空回答（含空白字符）统一为 0；非空评分保持原样。汇总包含 `audit_only=true`、`model_rejudged=false`、corrections、输入 hash 与分母。没有获得评分的题仍留在分母中，complete=false。不要把审计结果称为重新跑过裁判的结果，也不要用删除失败题的方法提高分数。

## 54.4 第二步：独立验证在线裁判

先按 [49](49_local_aux_models.md) 启动裁判服务，并确保当前环境文件与服务的 model、端口、认证一致。这里只需 SCORER 三项，不需要摘要、搜索、Jina 或待测 checkpoint。

```bash
ENV_FILE=/private/path/49_benchmark_env.sh \
  bash scripts/sh/benchmark/54_check_judge.sh preflight \
  --output-dir outputs/results/judge_preflight_001
```

每个 SCORER_URLS 端点分别检查两个真实请求：问题为法国首都，Paris 应判正确，Tokyo 应判错误。空回答由本地规则判错，不发请求。任一端点失败立即中止，并留下 `judge_preflight_*.json`；其中保留期望值、标签、每次原始文本和安全错误码，不打印 API key。自检不使用 benchmark 参考答案，不会产生基准成绩。

可选 CLI：`--judge-attempts 3`、`--judge-max-tokens 1024`、`--judge-timeout 120`、`--judge-enable-thinking`、`--official-dir /path/to/OpenSeeker`。默认使用固定 revision 的官方提示；若上游目录不存在，ensure_official 会联网获取源码，受限节点先按 46 准备代码。audit 不需要上游源码。

自检识别不了所有语义错误。正式实验仍应抽查非空预测（正确、错误、含数值、多部分答案、拒答等），并固定同一裁判权重和配置对比各方法。此自检不是 BrowseComp-Plus 的置信度校准误差指标；二者用途不同。

## 54.5 第三步：重新评分已有在线预测

在有评测依赖的环境中执行；不需要在评测环境安装 SGLang。服务保持运行，使用全新结果目录：

```bash
unset RUN_DIR MODEL_PATH PROMPT_DATA
ENV_FILE=/private/path/49_benchmark_env.sh \
MODE=native \
DATA=tmp/benchmark_native_20261101T091822637884Z/questions.jsonl \
PREDICTIONS=tmp/benchmark_native_20261101T091822637884Z/result_for_judge.jsonl \
OUTPUT_DIR=outputs/results/rescore_stable_001 \
LIMIT=-1 JUDGE_WORKERS=1 \
  bash scripts/sh/plain/47_plain_browsecomp.sh
```

该命令适用于 Plain/Beacon/基座产生的兼容预测，不会重新加载待测模型。official-v2 来源须选 MODE=official-v2，保持其问题后缀一致。使用原输出的 questions.jsonl 可避免将 12 题预测错误地放进全量分母。PREDICTIONS 内含 DATA 之外的题会报错，不能直接在全量预测上设置 LIMIT=2 而不裁剪文件。

评分步骤：对齐全部题目 → 非空预测存在时做端点自检 → 空回答本地固定 0 → 对非空回答调用裁判 → 写逐题及汇总。全部预测为空时不调用裁判、不做在线自检，但 47 仍要求 SCORER 配置字段非空。

默认请求为 temperature=0、max_tokens=1024、非流式、skip_special_tokens=false、chat_template_kwargs.enable_thinking=false。响应必须 finish_reason=stop、有非空最终 content；length、空 content、只有思考内容均视为失败。SGLang/Qwen 的 thinking 参数参考 [Qwen 官方部署说明](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/sglang.md) 与 [SGLang API 说明](https://sgl-project.github.io/basic_usage/openai_api_completions.html)。49 服务启动器也通过 `AUX_ENABLE_THINKING=0` 设置服务默认关闭思考，以覆盖未显式传 thinking 配置的官方摘要请求；请求级配置优先。更新脚本后需重启服务才能生效。目标服务不支持这些扩展时会显式报错，不自动更换协议或偷偷去掉参数。

只接受完整的 A/B 标签，或 `A. [CORRECT]` / `B. [INCORRECT]` 等一致标签；剥离完整的 think 段后再解析。`A or B`、`A. [INCORRECT]`、截断输出及自由长解释不计有效标签。错误格式、网络超时、429/5xx 等最多总尝试 3 次，退避 0.5、1 秒；400/401/403/404 立即停止该题尝试。有效的 B 不重试，不“重试到答对”，不靠多数投票抬分。

所有未知/异常写入逐题文件，score=null、score_source=judge_error。汇总仍保留全部题，complete=false 时最后以异常退出。`official_accuracy` 是有效 0/1 标签子集上的比例，排障参考；应报告 `accuracy_all_questions` 并确认 complete=true，不能只挑有效题作成绩。

主要输出：

| 文件/字段 | 用途 |
|---|---|
| `manifest.json` | 输入/预测指纹、固定上游 revision、所有 CLI 配置、judge_policy 和实现 hash |
| `coverage.json` | 预测覆盖、空回答、缺失条数 |
| `judge_preflight*.json` | 正/负例自检证据；全空或显式跳过时不产生 |
| `result_for_judge_eval.jsonl` | 每题 query/answer/response、score_source、judge_raw、judge_attempts、generation_status |
| `summary.metrics.json` | 全部分母、unknown 数量、complete、生成状态分布、评分策略 |
| `judge.log` | 完成数量及评分来源，不输出认证 key |

## 54.6 离线 BrowseComp-Plus 的评分与复现

仍使用固定官方 GRADER_TEMPLATE / parser，并增加字段有效性检查，不修改 vendored 官方文件。不把 BrowseComp 的 A/B 提示套到 BrowseComp-Plus。空/失败生成保留在完整分母并计 0，且不加载本地裁判；非空 completed 预测才触发 HF 模型加载和自检。

```bash
ACTION=score \
ROOT_DIR=/data/benchmarks/browsecomp-plus \
PREDICTIONS=/data/original-run/predictions.jsonl \
OUTPUT_DIR=outputs/results/bcp_rescore_stable_001 \
JUDGE_MODEL=/data/models/Qwen3-32B \
JUDGE_TEMPERATURE=0 JUDGE_ATTEMPTS=3 \
  bash scripts/sh/plain/56_plain_browsecomp_plus.sh
```

Beacon 换成 beacon/56；原始基座使用 baseline/74。保持 ROOT_DIR、LIMIT 与原生成选择一致，不能省略原子集 LIMIT 后改成全量。此路线不使用 SCORER_URLS，也不请求本地 SGLang；模型在本机 HF 加载。完全离线网络保护继续有效。

默认 `JUDGE_TEMPERATURE` 从 0.7 调整为 0，`enable_thinking=false`。可显式恢复温度与思考，但需记录为另一协议；固定 seed 不能保证跨硬件/版本逐位一致。裁判输入超过 max_context 会报错，不静默截断；输出耗尽预算会失败。失败最多总尝试 3 次，原始输出/错误保留。

裁判必须提供 extracted_final_answer、reasoning、唯一 correct: yes/no、唯一有效 confidence（0—100）。冲突标签、缺字段或越界置信度视为未知，不能借助宽松上游 parser 自动钳制到 100 后当成功。`judge.jsonl` 中保留全部尝试；summary 的 judge_errors>0 / complete=false 时不得报告为完成的成绩。引用、检索 recall 与 HotpotQA 指标继续使用原有口径。

## 54.7 参数与脚本变量总表

公共映射文件 `scripts/sh/benchmark/54_judge_options.sh` 由 47、56/57、72/73 路线自动加载。可用环境变量设置，也可在命令末尾传同名 CLI 覆盖。不存在全局自动安装、自动启动服务或自动训练。

| 环境变量 | CLI | 默认 | 用途与修改提示 |
|---|---|---|---|
| `JUDGE_WORKERS` | `--judge-workers` | 在线 1 | 在线裁判并发；确认模型容量后提高，默认由 20 降为 1 |
| `JUDGE_ATTEMPTS` | `--judge-attempts` | 3 | 每题总尝试数，正整数；不是“额外重试数” |
| `JUDGE_MAX_TOKENS` | `--judge-max-tokens` | 在线 1024，BCP 4096 | 仅裁判输出预算，不是待测模型生成预算 |
| `JUDGE_TIMEOUT` | `--judge-timeout` | 在线 120 秒 | 单次 API 请求超时，正有限数；BCP HF 无 HTTP timeout |
| `JUDGE_ENABLE_THINKING` | `--judge-enable-thinking` | 0 | 1 开启裁判思考；需增加预算并单独记录 |
| `SKIP_JUDGE_PREFLIGHT` | `--skip-judge-preflight` | 0 | 1 显式跳过正反例自检；排障或特殊模型使用，记录原因 |
| `JUDGE_TEMPERATURE` | `--judge-temperature` | BCP 0 | 仅本地 HF；在线固定 0 |
| `JUDGE_MAX_CONTEXT` | `--judge-max-context` | BCP 32768 | 输入+生成总预算，超限报错 |
| `JUDGE_TOP_P` / `JUDGE_TOP_K` | 对应同名参数 | BCP 0.8 / 20 | 温度>0 才参与采样 |
| `TOOL_CALL_RETRIES` | `--tool-call-retries` | 1 | native 每题工具调用格式反馈总额度，非负；0 恢复严格失败 |
| `TOOL_POLICY` | `--tool-policy` | 在线 native stable | stable 使用新的 bounded visit；official 使用固定上游 Visit；不改变官方生成脚本 |

54 独立工具只接受表中适用的 CLI，不读取 JUDGE_* shell 映射；例如 `54_check_judge.sh preflight --judge-timeout 180`。47/56 等的 DRY_RUN 只解析配置，不做自检、不访问 API/模型，不能代替真实自检。数据/模型导入与详细角色变量见 [49](49_local_aux_models.md)。

## 54.8 工具格式恢复与网页摘要诊断

原失败样例包括不合法 JSON 引号、末尾 `】`、arguments 非对象。严格 parse_calls 保持不变；不会自动修引号、执行部分解析或把错误工具调用当成最终答案。native 入口默认最多一次格式反馈：

1. 保存原始 assistant 文本和解析错误事件；此次没有工具执行。
2. 在 tool_response 中要求模型按原协议重新输出合法 JSON，不添加参考答案。
3. 重试消耗原 max_turns、context/token 预算；额度耗尽仍 invalid_call，最终空回答仍计 0。

encode 仅在显式启用恢复的推理中允许携带历史非法调用用于反馈，序列化 token 与正文不变；训练数据 validate、默认 encode 和 parse_calls 仍严格拒绝。记录 `events[].kind=invalid_call`、retry_requested 和 manifest 的 tool_call_retries；重试机会会改变生成协议，比较实验必须一致。原有 21/31 调用默认不额外开启恢复。

在线 native 的 stable visit 分离两个阶段：Jina 网页获取、摘要模型 JSON 提取。Jina 单次默认 50 秒，每阶段最多 3 次；取消上游 8×3 嵌套读取重试。401/403/404 等配置/访问错误不在同一次阶段里反复请求。失败输出区分 `page_fetch_failed (http_401)`、`summary_failed (...)`；不会伪装成成功网页证据，也不把 key 或 HTTP 响应体塞进错误日志。不同 URL 仍独立处理。

摘要默认 temperature=0、关闭思考、输出 2048 tokens，使用 SUMMARY_* 服务配置并复用稳定 HTTP 客户端。必须生成 rational/evidence/summary 三个字符串字段且 summary 非空。没有改变离线固定语料的检索/visit，也没有修改官方-v1/v2 的生成工具。

| stable visit 变量 | 默认 | 修改提示 |
|---|---|---|
| `SUMMARY_ATTEMPTS` | 3 | 网页获取/摘要每阶段总尝试次数，1—10 |
| `SUMMARY_TIMEOUT` | 120 | 摘要单次请求超时秒数，正有限数 |
| `SUMMARY_MAX_TOKENS` | 2048 | 摘要输出预算，正整数 |
| `SUMMARY_MAX_INPUT_CHARS` | 40000 | 仅网页正文字符上限；超出显式标记 input_clipped=true，并记录原字符数 |
| `JINA_TIMEOUT` | 50 | 网页单次请求超时秒数，正有限数 |

字符上限不是 tokenizer 精确 token 上限；若摘要模型上下文较小或 goal 很长，需减小 SUMMARY_MAX_INPUT_CHARS，或提高服务 AUX_CONTEXT_LENGTH。不会在收到 HTTP 400 时无提示地不断截短网页重试；错误应修复预算配置。服务总上下文必须容纳摘要提示、正文和生成预算。

如需尽量恢复原 native 生成行为：`TOOL_POLICY=official TOOL_CALL_RETRIES=0`。新稳定评分仍会拦截空回答、校验标签并记录策略；本次没有保留“让空回答交给裁判”作为可选开关。

## 54.9 验证边界

本次用提供的历史文件做真实离线审计，确认修正项恰为 item_index=8，输入指纹与 manifest 一致。自动化覆盖空回答无 API 调用、截断的 A 被拒绝、A/B 冲突、非法输出重试、有效 B 不重试、401 不重试、unknown 留在分母、自检识别总判 A、审计不改旧文件、工具格式恢复和网页认证诊断；相关现有基准测试继续检查 Plain/Beacon/基座与离线保护路径。

没有在远端创建环境、运行真实 Qwen/SGLang、自检真实裁判、访问搜索/Jina 或重新生成真实题目，也没有启动训练。真实模型的语义可靠性与网页可访问性需要目标服务器按 54.4、54.5 和小批量重新生成步骤验证。

## 54.10 preflight 报 ast.Call / malformed node 的修复

远端日志 `tmp/log.txt` 显示异常发生在读取 `eval/prompt.py`，尚未构造裁判客户端或发出模型请求。固定上游的真实赋值是 `JUDGE_PROMPT_BC_en = """...""".strip()`；赋值右侧是 AST Call，而不是单纯字符串 Constant。旧代码直接对右侧调用 ast.literal_eval，因其不支持执行 `.strip()` 而抛出 `ValueError: malformed node or string ... <ast.Call ...>`。这是本项目读取代码的缺陷，与 API key、401、CUDA 或模型能力无关。

修复统一使用 `search_comp.evaluation.judging.load_judge_prompt`，只接受字符串字面量及无参 `.strip()`，不执行上游 Python 文件，不使用 eval。preflight 与 47 正式评分共用同一读取器，避免修好自检后正式评分仍在同一位置失败。原上游文件和提示内容不修改；不需要升级 Python、安装依赖或重启 SGLang 来解决这个异常。

将以下三个文件一起同步到目标服务器，不能只更新调用处而遗漏 helper：

- `search_comp/evaluation/judging.py`
- `search_comp/workflows/check_judge.py`
- `search_comp/workflows/benchmark_openseeker.py`

之后重跑：

```bash
ENV_FILE=/private/path/49_benchmark_env.sh \
  bash scripts/sh/benchmark/54_check_judge.sh preflight
```

默认会生成新的输出目录。若指定 OUTPUT_DIR / --output-dir，使用尚不存在的目录；正式评分旧失败目录也不原地覆盖。修复通过只表示提示可读取；后续仍要看自检正/反例是否通过。

回归覆盖真实上游的 `.strip()` 赋值结构、字面量兼容、非法表达式拒绝、禁止执行源码，以及 preflight CLI 与正式评分链路。先前只含字符串字面量的测试替身没有覆盖真实 `.strip()` 形式，此次已纠正，并直接检查仓库内固定上游提示能正确读取。
