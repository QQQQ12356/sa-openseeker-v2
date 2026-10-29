# 40 答案提取与 EM/F1

当前训练/评测共享提示词、v4 数据迁移、算法一致性检查及全部入口的保留范围见 [45 统一流程](45_unified_agent_flow.md)。

参考 [OpenSeeker 官方实现](https://github.com/PolarSeeker/OpenSeeker/blob/e3dec1f727d87c5eaa93506e4882c67237647098/src/llm_tool_openseeker.py)：模型生成出现 `</answer>` 时结束作答；`_get_last_assistant_answer_from_messages()` 取最后一条 assistant 消息、移除工具内容，并截取 `</think>` 后的回复。`eval/generate_answer.py` 将其保存为 `final_response`；`eval/eval.py` 把 final_response、query、参考 answer 交给 LLM 裁判判断语义正确性。因此原项目的裁判 accuracy 与本项目的 EM/F1 是不同指标，不能直接互换。

本项目 `search_comp/evaluation/tool_agent.py` 在结束工具循环后，只从当前生成的最后一轮 assistant 回复提取 prediction：去掉 think/thinking 与工具包装，若有完整 `<answer>...</answer>` 则取最后一个块的正文；结构化数据的普通 assistant 回复可直接使用清理后的正文。不从问题、参考答案、历史工具返回或早期推理中提取答案。未闭合思考/答案标签、空正文、生成预算耗尽或工具异常不会作为成功作答计分。生成停止边界为工具调用结束、答案闭合或消息结束。

预测 JSONL 同时记录 `final_response`（用于评分的生成回答）、`prediction`、参考 `answer`、完整 messages、status、工具事件、耗时和生成 token 数。失败样本仍计入总数，EM/F1 为 0，避免只对成功样本计分。

为保持原项目 EM 行为，OpenSeeker 路径的 EM 使用大小写折叠与空白归一化后的完整字符串相等，保留标点和冠词差异；F1 使用已有 SQuAD 风格的归一化与词频交集，去英文标点/冠词、小写、按空白分词。长解释可能使 EM=0、F1 降低，这是对完整生成回答评分的结果，不用参考答案反向构造精简预测。Search-R1 原评测路径继续使用其既有标准化 EM。中英文混合文本的 F1 仍为按空白分词的原实现，不声称是字符级中文 F1。

主入口 `21_plain_eval.sh` 是 Plain 薄包装，转入共用的 `31_beacon_eval.sh`；参数在 31 顶部修改，或通过同名环境变量覆盖：RUN_DIR 或 MODEL_PATH、DATA/SPLIT、LIMIT、MAX_CONTEXT、MAX_NEW_TOKENS、MAX_TOOL_TOKENS、MAX_TURNS、GPU 与 NUM_WORKERS。预测与 `.metrics.json` 写到同一结果目录，汇总包含 exact_match、f1、失败状态和配置。NUM_WORKERS 是模型副本数，不是训练分布式进程数。

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
