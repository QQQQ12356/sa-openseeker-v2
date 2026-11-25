# 37 BrowseComp-Plus 一次运行生成与评分

Baseline、Plain 和 Beacon 各有一个 `46_run_browsecomp_plus.sh`。每次执行评测一个 Qwen3.5-4B 模型或检查点、一个本地检索器和一个观测窗口。三个入口共用原始题目顺序、逐题种子生成规则、结果归档和判分代码。Baseline 默认使用 Qwen 原生浏览器工具循环，Plain 和 Beacon 使用训练协议。 [参考仓库](https://github.com/i-DeepSearch/observation-masking) · [论文实验设置](https://arxiv.org/html/2606.00408#S3)

## 1. Linux 上准备环境与资产

1. 激活 [Linux 运行环境](10_environment.md)，把原始 Qwen3.5-4B 权重与 tokenizer 准备到本地。Baseline 直接加载该目录。Plain、Beacon 使用本项目训练实验的最终模型或周期检查点，原训练基座路径和提示词数据须能在 Linux 上访问。 [检查点格式](33_checkpoint_evaluation.md)
2. 按 [检索依赖说明](35_browsecomp_plus_dense.md#1-准备题目语料和文档库) 在 Linux 安装 Pyserini／FAISS。官方 BM25 和 Baseline 原生浏览器均需 Java 21。按 [数据准备流程](20_data.md#8-准备-browsecomp-plus) 运行 31 的 `download → prepare → index`，三个方法默认共用 `v2`。已有 `v2` 时用同一版本重跑 31 的 `download` 补齐高亮 JAR，无需重建 prepared 和索引。推理只接收问题，标准答案和 qrels 用于评分。
3. 默认裁判为 `gpt-5-mini`。运行前从 Linux 环境或作业密钥管理器导出 `OPENAI_API_KEY`。选择 `openai` 会向指定裁判端点发送题目、模型回答和标准答案。需要全离线评分时，将 `SA_JUDGE__JUDGE_BACKEND=local`，并把 `SA_JUDGE__JUDGE_MODEL` 改为本地裁判目录。仅生成预测时设 `SA_SCORE_AFTER_GENERATION=false`。

## 2. 编辑一次测试的配置

打开对应方法的 Bash 入口并在文件顶部编辑参数。相对路径以项目根目录为准。

| 方法 | Bash 入口 | 模型配置 |
|---|---|---|
| Baseline | `scripts/baseline/sh/46_run_browsecomp_plus.sh` | `SA_MODEL_PATH` 填原始 Qwen3.5-4B 本地目录。 |
| Plain | `scripts/plain/sh/46_run_browsecomp_plus.sh` | `SA_TRAINING_RUN` 填 Plain 的 Qwen3.5-4B 实验根目录。 |
| Beacon | `scripts/beacon/sh/46_run_browsecomp_plus.sh` | `SA_TRAINING_RUN` 填 Beacon 的 Qwen3.5-4B 实验根目录。 |

Plain 和 Beacon 的 `SA_CHECKPOINT` 留空时使用最终模型，也可填如 `checkpoint-200`。若做训练重合筛选，使用 `SA_OVERLAP=filter` 并记录筛选范围。报告全量分数时保留默认的 `SA_GENERATION__LIMIT=-1`，Plain 和 Beacon 同时设 `SA_OVERLAP=off`。首次可先将 LIMIT 设为 5 验证模型加载、搜索与裁判。

### 系统提示词与答案提取

`SA_GENERATION__PROMPT_PROFILE=qwen-native` 是 Baseline 46 的默认值。它用原始 Qwen tokenizer 的 `apply_chat_template(..., tools=..., add_generation_prompt=True)`，把推理、函数调用和工具结果分别作为原生消息交给模型。固定的深度检索指令含日期、行号引用和同轮多工具调用提示。 `SA_GENERATION__PROMPT_DATE` 留空时，全程使用 Linux 启动日期；需要指定时填 `YYYY-MM-DD`。自动续测沿用源运行的日期。

Plain 和 Beacon 46 默认使用 `benchmark`，配合训练兼容的 OpenSeeker `search/visit` 工具格式。需要让 Baseline、Plain、Beacon 使用相同提示词和工具协议做消融时，可将 Baseline 也切到 `benchmark`。要沿用训练系统提示词，设为 `training`。Baseline 此时用内置 Polar 提示词，也可通过 `SA_PROMPT_DATA` 指定规范化 JSONL。Plain 和 Beacon 从训练记录定位提示词来源，混合训练实验使用 `training` 时要指定一套 `SA_PROMPT_DATA`。

最终回复要求含 `Explanation`、`Exact Answer` 和 `Confidence`。解释可写 `[docid]` 引用。程序从最终 assistant 回复移除推理和工具包装，完整可见文本写入 `final_response`，用于语义裁判与引用评分。单独提取的答案写入 `exact_answer` 和 `prediction`，用于 EM/F1。原生循环识别带标签的答案或闭合的 `<answer>`。OpenSeeker 入口也兼容旧检查点的纯文本回答。

### Baseline 原生浏览器

- `browser.search` 返回带原始 URL 和链接 ID 的结果，默认每条查询取 15 篇。使用 Lucene 查询相关摘要，限制为 180 字符。该高亮处理同样用于 FAISS 检索结果。
- `browser.open` 可用 URL 或 cursor 上的链接 ID 打开固定语料内文档，也可按 `loc`、`num_lines` 翻页。默认文档视口为 50 行。 `browser.find` 在已打开的页面定位文本并返回可继续打开的匹配位置。
- 同一轮的工具调用使用执行前的 cursor 状态，各工具结果按调用顺序回填。检索器内部串行使用，页面读取和查找可并行。工具错误作为反馈回填，旧成功观察才参与 CM。原生模型上下文中的单条成功观察超过 12,000 字符时截取，完整观察仍保存到轨迹。

### 选择检索器

| `SA_GENERATION__RETRIEVER` | 准备与参数 |
|---|---|
| `bm25` | 默认。使用 31 下载的官方 Lucene 索引，由 Pyserini 排序。`TITLE_WEIGHT` 只用于旧版或自定义数据的 SQLite FTS5。 |
| `qwen3-embedding-8b` | 用 [33 脚本](35_browsecomp_plus_dense.md) 导入同一索引 commit 的官方分片，运行时用 FAISS `IndexFlatIP`。设置 `VECTOR_INDEX_DIR`、`EMBEDDING_MODEL_PATH` 和查询编码设备 `EMBEDDING_DEVICE`。 |

两个检索器都在评测进程内读取本地资产，准备和运行命令均从 `sa-reconstruction` 执行。默认 BM25 在 31 完成后可直接运行 46。测试 Qwen3-Embedding-8B 时，用相同的 `SA_VERSION` 执行 33 的 `download → prepare`，再在 46 脚本填入 33 打印的路径。例如：

```bash
export SA_GENERATION__RETRIEVER=qwen3-embedding-8b
export SA_GENERATION__VECTOR_INDEX_DIR=artifacts/benchmarks/browsecomp-plus/v2/vector_indexes/qwen3-embedding-8b-official
export SA_GENERATION__EMBEDDING_MODEL_PATH=/path/to/Qwen3-Embedding-8B/commit
export SA_GENERATION__EMBEDDING_DEVICE=cuda:0
```

向量模型和生成模型在同一个 worker 加载。GPU 显存不足时可将 `EMBEDDING_DEVICE` 改为 `cpu`。两种检索器都通过同一份本地语料生成观察内容，召回按官方 docid 计算。 [向量资产准备与校验](35_browsecomp_plus_dense.md)

### 设置 CM 与解码

| 变量（生成项均以 `SA_GENERATION__` 开头） | 默认 | 含义 |
|---|---:|---|
| `OBSERVATION_WINDOW` | 0 | 0 为 No CM。设 5 复现论文报告的保留窗口。按成功的工具调用轮计数，一轮有多条返回时整组保留或归档。 |
| `MAX_TURNS` / `MAX_NEW_TOKENS` | 500 / 8192 | 每题最多生成轮数与每轮输出 token 上限。 |
| `MAX_CONTEXT` / `MAX_TOOL_TOKENS` | 131072 / 32768 | 单题完整输入预算与单条工具返回上限，需结合 GPU 显存。 |
| `TOOL_COUNT` / `TOOL_CALL_RETRIES` | 4096 / 1 | 每题最多工具调用对象数与非法格式修正次数。若本地工具执行失败，该题仍按现有流程记录为失败。 |
| `DECODING` / `TEMPERATURE` / `TOP_P` | sample / 1.0 / 1.0 | 解码策略。支持 `greedy`，其温度和 top-p 不参与取样。 |
| `REPETITION_PENALTY` / `GENERATION_SEED` | 1.0 / 42 | 重复惩罚和逐题采样种子。 |
| `TOPK` / `SNIPPET_TOKENS` / `READ_TOKENS` | 15 / 512 / 4096 | 每条查询取回篇数，搜索摘录和本地访问的 token 预算。 |
| `ALLOW_VISIT` / `VERIFY_INDEX` | true / true | 允许本地文档访问，启动时校验索引内容。 |
| `NUM_WORKERS` / `CUDA_VISIBLE_DEVICES` | 1 / 0 | 每个 worker 使用一张 GPU 并加载完整模型副本。使用多个 worker 时，填写足够的可见 GPU。 |

对同一方法做 No CM / CM 对照时，只切换窗口及用于识别结果的 `SA_RUN_NAME`。旧的工具返回会在送入模型的新一轮上下文中变成占位符。原始调用及完整返回仍存于轨迹。Plain 在窗口滚动时重建被改写的缓存前缀。Beacon 可同时使用自己的观测压缩与外部窗口，比较 Beacon 压缩本身时将外部窗口设为 0。

Baseline 原生路径使用本项目的 Transformers 生成器和本地浏览页面渲染，参考仓库使用 vLLM 和自己的浏览器渲染。对照参考实验时同时核对资产 commit、tokenizer、提示日期和实际观察。表中的摘录／读取 token 预算、`ALLOW_VISIT` 和格式修正次数用于 OpenSeeker 路径。三种方法的 46 入口继续按原顺序选题，并以 `SA_GENERATION__GENERATION_SEED=42` 为默认种子。

### 设置评分

| 变量（均以 `SA_JUDGE__` 开头） | 默认 | 含义 |
|---|---|---|
| `JUDGE_BACKEND` / `JUDGE_MODEL` | openai / gpt-5-mini | 在线 API 模型名，或选择 `local` 后填本地模型目录。 |
| `JUDGE_URL` | `https://api.openai.com/v1` | OpenAI API base URL，代码追加 `/chat/completions`。仅 `openai` 使用。 |
| `JUDGE_MAX_TOKENS` / `JUDGE_ATTEMPTS` | 4096 / 3 | 每次生成预算与单次判定的最大尝试次数，在线时预算含推理 token。 |
| `JUDGE_TIMEOUT` | 120 | 在线裁判单次 HTTP 请求超时秒数。 |

## 3. 运行、核验与重新评分

启动前确认 `SA_BENCHMARK_DIR` 中同时有 `index.sqlite` 和 `index.manifest.json`。如果只有官方 Lucene 文件或已准备的 Parquet／JSONL，在 `scripts/data/sh/31_prepare_browsecomp_plus.sh` 设同一 `SA_VERSION` 和 `SA_PHASE=index` 后运行 31。这一步生成本地文档库，BM25 和 FAISS 都用它读取正文。若只看到 `index.building.sqlite`，先检查对应的 index 任务是否还在运行。任务已终止时，将残留文件移到另一个名字后重跑 index。若两个成品都存在却仍打不开，请从评测节点检查文件读权限和各级目录的访问权限。

```bash
bash scripts/baseline/sh/46_run_browsecomp_plus.sh
# 选择其他方法时执行对应的 46
# bash scripts/plain/sh/46_run_browsecomp_plus.sh
# bash scripts/beacon/sh/46_run_browsecomp_plus.sh
```

默认每次执行创建独立评测目录，并把已完成的评测传给同一方法的评分流程。按需在同一个脚本切换阶段。

| 运行方式 | `SA_EVALUATION_DIR` | `SA_SCORE_AFTER_GENERATION` |
|---|---|---|
| 生成并评分（默认） | 空 | `true` |
| 仅生成 | 空 | `false` |
| 对已有预测评分 | 已完成的评测根目录 | 任意值 |

可在终端找到对应的产物路径：

| 文件 | 内容 |
|---|---|
| `results/predictions.jsonl` | 完整 `messages`、逐轮 `turn_stats`、检索 docid、工具调用计数、`final_response` 和 `exact_answer`。开启 CM 时另有 `final_messages` 和 `masked_observations`。 |
| `results/manifest.json` / `evaluation.json` | 题目范围、检索索引、模型、运行参数和预测指纹。 |
| `scores/<评分记录>/results/judge.jsonl` | 逐题语义裁判、失败原因、召回与引用判定。 |
| `scores/<评分记录>/results/summary.metrics.json` | `accuracy` 使用全部已选题作分母，`judged_accuracy` 只统计有有效裁判结果的题，另有 EM/F1、证据与金标准文档召回、每题工具调用统计。全量时应核对 `count=830`、`subset=false`、`judge_errors=0`。 |

语义裁判沿用官方 BrowseComp-Plus 评分提示词，输入题目、完整可见回复和标准答案。默认 GPT-5-mini，判定可重试并保留原始返回和失败记录。

生成成功后，如果裁判服务中断或需要更换裁判，将该次**评测根目录**填入同一个方法脚本的 `SA_EVALUATION_DIR`，再次运行会读取原预测并创建新评分目录。要扩展一个已完成的测试，清空 `SA_EVALUATION_DIR`，将原评测填入 `SA_RESUME_FROM`，提高 `LIMIT`，并沿用原生成参数和提示词档案。 [续测与中断恢复](36_browsecomp_plus_resume.md) · [逐题报告](33_checkpoint_evaluation.md#3-核对结果并评分)
