# 40 答案提取与 EM/F1

当前保留训练原始 system、测试提示词来源选择、必要的离线适配及全部入口的保留范围见 [45 统一流程](../common/45_unified_agent_flow.md)。

参考 [OpenSeeker 官方实现](https://github.com/PolarSeeker/OpenSeeker/blob/e3dec1f727d87c5eaa93506e4882c67237647098/src/llm_tool_openseeker.py)：模型生成出现 `</answer>` 时结束作答；`_get_last_assistant_answer_from_messages()` 取最后一条 assistant 消息、移除工具内容，并截取 `</think>` 后的回复。`eval/generate_answer.py` 将其保存为 `final_response`；`eval/eval.py` 把 final_response、query、参考 answer 交给 LLM 裁判判断语义正确性。因此原项目的裁判 accuracy 与本项目的 EM/F1 是不同指标，不能直接互换。

本项目 `search_comp/evaluation/tool_agent.py` 在结束工具循环后，只从当前生成的最后一轮 assistant 回复提取 prediction：去掉 think/thinking 与工具包装，若有完整 `<answer>...</answer>` 则取最后一个块的正文；结构化数据的普通 assistant 回复可直接使用清理后的正文。不从问题、参考答案、历史工具返回或早期推理中提取答案。未闭合思考/答案标签、空正文、生成预算耗尽或工具异常不会作为成功作答计分。生成停止边界为工具调用结束、答案闭合或消息结束。

预测 JSONL 同时记录 `final_response`（用于评分的生成回答）、`prediction`、参考 `answer`、完整 messages、status、工具事件、耗时和生成 token 数。失败样本仍计入总数，EM/F1 为 0，避免只对成功样本计分。

为保持原项目 EM 行为，OpenSeeker 路径的 EM 使用大小写折叠与空白归一化后的完整字符串相等，保留标点和冠词差异；F1 使用已有 SQuAD 风格的归一化与词频交集，去英文标点/冠词、小写、按空白分词。长解释可能使 EM=0、F1 降低，这是对完整生成回答评分的结果，不用参考答案反向构造精简预测。Search-R1 原评测路径继续使用其既有标准化 EM。中英文混合文本的 F1 仍为按空白分词的原实现，不声称是字符级中文 F1。

`21_plain_eval.sh` 和 `31_beacon_eval.sh` 是分别独立的 Plain / Beacon shell 入口，各自直接调用通用 Python 评测器，不互相调用。参数在各自脚本顶部修改，或通过同名环境变量覆盖：RUN_DIR 或 MODEL_PATH、DATA/SPLIT、LIMIT、MAX_CONTEXT、MAX_NEW_TOKENS、MAX_TOOL_TOKENS、MAX_TURNS、GPU 与 NUM_WORKERS。自动读取 RUN_DIR 模型时，入口会检查实验模型类型；如显式指定 MODEL_PATH，RUN_DIR 仍可用来选择同一数据。预测与 `.metrics.json` 写到同一结果目录，汇总包含 exact_match、f1、失败状态和配置。NUM_WORKERS 是模型副本数，不是训练分布式进程数。

21 的 MODEL_PATH 可直接指向 Plain LoRA adapter，BASE_MODEL_PATH 可指定训练基座、TOKENIZER_PATH 可指定训练 tokenizer；具体回退规则和原始模型对照见 [22](../baseline/22_qwen35_baseline_and_adapter.md)。此能力不扩展到 47/56/57 或 Beacon adapter。Plain native 可使用 [23](../common/23_decoding.md) 调整贪心、采样和重复惩罚。

| BACKEND | 行为与解释 |
|---|---|
| replay | 严格核对生成动作和记录动作，匹配才返回对应观测；用于离线动作复现诊断 |
| corpus | 对给定 JSONL 本地 BM25 search/visit，可执行新查询；结果取决于语料来源 |
| http | 向 ENDPOINT POST `{name, arguments}`，要求返回 `{content: str}`；用于真实工具服务 |

不把 replay 结果解释为在线搜索能力。历史 visit 语料库要注明来源划分，不能称为独立检索评测。当前保留 EM/F1，不自动调用付费裁判 API。

## 40.1 独立 Benchmark 与官方裁判

新增 [42 OpenSeeker Benchmark](42_openseeker_benchmark.md) 流程以 BrowseComp 为默认独立测试集，固定官方代码版本并直接调用官方生成/裁判脚本；Beacon 通过 native 模式保留压缩并采用同一裁判。新结果写入独立目录，不替换上述 EM/F1 汇总。native 的序列化、解码和预算与官方不同，详见对齐表；官方 accuracy 与 EM/F1 必须分别报告。原有入口仍不自动调用裁判 API。

## 40.2 固定语料离线评分

新增 56/57 的离线 search/visit 与本地推理共享现有 Plain/Beacon Agent。BCP 使用固定官方 prompt/parser、本地裁判和 evidence/gold 累积检索召回；Hotpot 执行官方 EM/F1、SP/Joint（仅模型生成的支持事实），保留 yes/no/noanswer 特例，不替换历史 EM/F1 的行为。漏题与失败保留分母，裁判未知会使评分非零退出。完整参数和指标定义见 [43](43_offline_benchmarks.md)。

## 40.3 当前各 Benchmark 指标总表

以下说明对应当前评分代码。Plain、Beacon、原始 Qwen3.5 在同一 benchmark 上共用指标实现；模型方法不会改变指标定义。不同工具后端、数据子集、语料范围和裁判配置仍须分别记录。

| 测试路线 | 主要指标（JSON 字段） | 辅助指标 | 汇总文件 |
|---|---|---|---|
| OpenSeeker 数据集诊断（21 / 31 / 基座 71） | 严格 EM：`exact_match`；token F1：`f1` | `statuses`、`count`、`seconds` | 预测文件同名 `.metrics.json` |
| 在线 BrowseComp（47 / 基座 72，含 official-v1、official-v2、native） | 语义裁判准确率：`accuracy_all_questions` | `official_accuracy`、`judged`、`judge_unknown_or_error`、生成覆盖率与状态分布 | `summary.metrics.json` |
| 离线 BrowseComp-Plus（56 / 基座 74） | 语义裁判准确率：`accuracy` / `Accuracy (%)`；检索召回：`evidence_recall` / `Recall (%)` | `gold_recall`、引用指标、校准误差、`avg_tool_stats` | `summary.metrics.json` |
| 离线 HotpotQA（57 / 基座 75） | 官方答案 `em`、`f1` | 答案 `prec` / `recall`，支持事实 `sp_*`，联合 `joint_*` | `summary.metrics.json` |

除明确带 `(%)` 的字段外，上表质量分数均为 0—1 比例，展示百分比时乘 100。耗时单位为秒，数量和平均工具调用数不属于比例指标。`generation.metrics.json` 仅包含生成状态、数量与耗时等信息，不能代替答案评分结果；只执行 `run` 不会得到离线基准的最终质量指标。

### OpenSeeker：严格 EM 与 token F1

- `exact_match`：对预测和参考答案分别执行 `casefold()`、按空白切分后重新连接，再判断完整字符串相等；不删除标点或冠词。
- `f1`：小写、去英文标点和英文冠词 `a/an/the`、压缩空白后按空白分词。按 token 词频交集计算 precision、recall，再取调和平均；无公共 token 或空 token 列表得 0。
- 两者均逐题计算后取平均。`status != completed` 或参考答案为空的题目计 0，保留在实际评测题目总数 `count` 中。
- `replay` 是动作复现诊断，`corpus` / `http` 是不同工具环境；即使指标同名，也需标明后端。此处严格 EM 不等于 HotpotQA 官方 EM，也不等于语义裁判准确率。

### 在线 BrowseComp：裁判准确率

固定 OpenSeeker 的 `JUDGE_PROMPT_BC_en` 对问题、参考答案和生成回答进行语义判定，合法判决转换为 0/1。

- `accuracy_all_questions = 正确题数 / 全部选定题数`，作为完整分母口径。
- `official_accuracy = 正确题数 / 获得合法判决的题数`。没有合法判决时为 null；仅在 `complete=true` 时与前者相等。
- 缺失预测经对齐生成空回答，空回答直接判 0。非空回答进入裁判；当前在线评分器不额外按生成 `status` 拒绝非空回答，状态另行记录。
- 裁判未知、解析失败或重试耗尽时保留在完整分母中，增加 `judge_unknown_or_error`，并设置 `complete=false`、非零退出。此时不能将已判决子集的准确率报告为完整成绩。

裁判模型、提示词协议、生成模式和是否通过裁判自检应随结果记录；详细规则见 [54](54_stable_judging.md)。

### BrowseComp-Plus：答案、检索、引用与校准

- `accuracy`：本地语义裁判判对题数除以全部选定题数；`Accuracy (%)` 为其百分比，保留两位小数。预测缺失、失败状态或空回答计 0。裁判出错增加 `judge_errors` 并使 `complete=false`，该结果不能作为完整成绩。
- `evidence_recall`：对每题取全部 search 返回的 docid 并集 R，以官方 evidence 标注文档集合 E 计算 `|R ∩ E| / |E|`，再按全部题目取平均。`Recall (%)` 是其百分比。它不是单次搜索的 Recall@k。
- `gold_recall`：采用相同公式，但分母集合改为 gold 标注文档。未完成的预测仍可贡献已经检索到的文档；缺失预测的检索集合为空，召回为 0。
- `citation_coverage`：completed 且非空、并提取到至少一个引用的回答数，除以全部题目数。`citation_precision`、`citation_recall` 使用官方引用函数，对这些有引用的回答取平均，参照集合为 evidence 文档；无此类回答时为 0。它们不是所有题目上的检索召回。
- `Calibration Error (%)`：使用固定官方校准函数，输入为有效裁判结果的 confidence 与正确性。有效置信度少于 100 条时输出占位值 0.0，并标记 `calibration_available=false`；此时该值不代表完美校准，需同时查看 `calibration_sample_count`。
- `avg_tool_stats`：各工具调用总数除以全部选定题数，包括失败题；用于衡量调用开销。

`--retrieval-only` 模式输出检索、引用和工具统计，不输出答案 accuracy 或校准误差，不能当作完整答案评测。本流程未实现 nDCG@10 等排序指标。字段细节见 [43](43_offline_benchmarks.md#4382-browsecomp-plus)。

### HotpotQA：答案、支持事实与 Joint

使用项目固定版本的官方 `hotpot_evaluate_v1.py`，全部指标逐题计算后按全部选定问题取平均。

- 答案 `em`：经官方归一化（小写、去英文标点及冠词、压缩空白）后完全相等。`prec`、`recall`、`f1` 按归一化后的 token 词频交集计算；`yes/no/noanswer` 不一致时，官方答案 F1/precision/recall 直接为 0。
- 支持事实 `sp_em`、`sp_prec`、`sp_recall`、`sp_f1`：将模型输出的 `[文档标题, 从 0 开始的句子编号]` 对视为集合，与官方支持事实集合比较。EM 要求集合完全相等，其余为集合交集的 precision、recall 和 F1。
- 联合 `joint_em = em × sp_em`；`joint_prec = prec × sp_prec`、`joint_recall = recall × sp_recall`；`joint_f1` 是 joint precision 和 joint recall 的调和平均，二者和为 0 时得 0。它不是答案 F1 与 SP F1 的直接乘积。
- 模型应输出 `{"answer":"短答案","sp":[["标题",0]]}`。普通文本按整段答案评分；缺失或不合法的 `sp` 不导出支持事实，该题 SP/Joint 计 0，但合法答案仍可独立获得答案分数。检索文档和参考支持事实不会自动成为预测 `sp`。
- 缺失、失败或空回答不导出有效答案，该题计 0 并保留分母。`supporting_fact_predictions` 是导出的有效 `sp` 条数，包含合法空列表，不是答对支持事实的题数。

### 结果记录与实现位置

报告成绩时至少记录 benchmark、数据版本/子集、题数、模型方法、主要指标、`complete`（适用时）、生成失败/漏题数；使用裁判时另记裁判模型及配置。不能只对成功生成的样本重新计算平均值。当前生成预测会直接保存参考 `answer`，供 HTML 展示；参考答案在生成结束后写入结果，不进入待测模型提示词。

实现核对入口：

- [OpenSeeker 评分与汇总](../../search_comp/workflows/evaluate_openseeker.py)：`score_answer`、`write_summary`。
- [在线完整分母与合法判决统计](../../search_comp/workflows/benchmark_openseeker.py)：`audit_scores`；[在线裁判](../../search_comp/evaluation/judging.py)：`score_online`。
- [离线评分](../../search_comp/offline/scoring.py)：`score_hotpot`、`retrieval_metrics`、`score_browsecomp`。
