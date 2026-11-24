# 37 BrowseComp-Plus 一次运行生成与评分

`scripts/tools/sh/46_run_browsecomp_plus.sh` 每次运行一个方法、一个本地检索器和一个观测窗口设置。它借鉴参考实验的结构，将逐题生成、工具轨迹保存和判分串起来，三个方法都使用 Qwen3.5-4B 基座。检索、生成和结果归档均由本仓库实现。 [流程参考](https://github.com/i-DeepSearch/observation-masking) · [论文实验设置](https://arxiv.org/html/2606.00408#S3)

## 1. Linux 上准备环境与资产

1. 激活 [Linux 运行环境](10_environment.md)，把原始 Qwen3.5-4B 权重与 tokenizer 准备到本地。Baseline 直接加载该目录。Plain、Beacon 使用本项目训练实验的最终模型或周期检查点，原训练基座路径和提示词数据须能在 Linux 上访问。 [检查点格式](33_checkpoint_evaluation.md)
2. 按 [数据准备流程](20_data.md#8-准备-browsecomp-plus) 用 31 脚本依次完成 `download → prepare → index`。三种方法共用一个 `SA_BENCHMARK_DIR`。有已建好的同版本资产时直接复用。正式成绩使用全量题和独立全量语料。
3. 默认裁判为 `gpt-5-mini`。运行前从 Linux 环境或作业密钥管理器导出 `OPENAI_API_KEY`。选择 `openai` 会向指定裁判端点发送题目、模型回答和标准答案。需要全离线评分时，将 `SA_JUDGE__JUDGE_BACKEND=local`，并把 `SA_JUDGE__JUDGE_MODEL` 改为本地裁判目录。仅生成预测时设 `SA_SCORE_AFTER_GENERATION=false`。

## 2. 编辑一次测试的配置

打开 `scripts/tools/sh/46_run_browsecomp_plus.sh`。相对路径以项目根目录为准。

| `SA_METHOD` | `SA_MODEL_PATH` | `SA_TRAINING_RUN` | `SA_CHECKPOINT` |
|---|---|---|---|
| `baseline` | 原始 Qwen3.5-4B 本地目录 | 空 | 空 |
| `plain` | 空 | Plain 的 Qwen3.5-4B 训练实验根目录 | 空表示最终模型，或填如 `checkpoint-200` |
| `beacon` | 空 | Beacon 的 Qwen3.5-4B 训练实验根目录 | 空表示最终模型，或填如 `checkpoint-200` |

`SA_PROMPT_DATA` 留空时，Baseline 用内置 Polar 提示词，Plain 和 Beacon 沿用所属训练实验的提示词。要对齐提示词时，Baseline 可以指向同一套规范化训练 JSONL。混合训练实验须显式指定一套 system/tools 来源。若做训练重合筛选，使用 `SA_OVERLAP=filter` 并记录筛选范围。报告全量分数使用 `SA_OVERLAP=off` 和 `SA_GENERATION__LIMIT=-1`。首次可先将 LIMIT 设为 5 验证模型加载、搜索与裁判。

### 选择检索器

| `SA_GENERATION__RETRIEVER` | 准备与参数 |
|---|---|
| `bm25` | 默认。本项目的 SQLite FTS5 索引，由 31 构建。设置与建索引一致的 `TITLE_WEIGHT`。 |
| `qwen3-embedding-8b` | 用 [33 脚本](35_browsecomp_plus_dense.md) 准备向量索引。设置 `VECTOR_INDEX_DIR`、`EMBEDDING_MODEL_PATH` 和查询编码设备 `EMBEDDING_DEVICE`。 |

两个检索器都在评测进程内读取本地资产，准备和运行命令均从 `sa-reconstruction` 执行。默认 BM25 在 31 完成后可直接运行 46。测试 Qwen3-Embedding-8B 时，用相同的 `SA_VERSION` 执行 33 的 `download → prepare`，再在 46 脚本填入 33 打印的路径。例如：

```bash
export SA_GENERATION__RETRIEVER=qwen3-embedding-8b
export SA_GENERATION__VECTOR_INDEX_DIR=artifacts/benchmarks/browsecomp-plus/v1/vector_indexes/qwen3-embedding-8b-official
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

本实现沿用项目的 OpenSeeker `search/visit` 工具格式、Transformers 和 Beacon 推理入口。参考仓库使用原生浏览器工具与 vLLM。两者的观察渲染、执行并行度和默认上下文预算有差异，公开论文分数可用于对照。

### 设置评分

| 变量（均以 `SA_JUDGE__` 开头） | 默认 | 含义 |
|---|---|---|
| `JUDGE_BACKEND` / `JUDGE_MODEL` | openai / gpt-5-mini | 在线 API 模型名，或选择 `local` 后填本地模型目录。 |
| `JUDGE_URL` | `https://api.openai.com/v1` | OpenAI API base URL，代码追加 `/chat/completions`。仅 `openai` 使用。 |
| `JUDGE_MAX_TOKENS` / `JUDGE_ATTEMPTS` | 4096 / 3 | 每次生成预算与单次判定的最大尝试次数，在线时预算含推理 token。 |
| `JUDGE_TIMEOUT` | 120 | 在线裁判单次 HTTP 请求超时秒数。 |

## 3. 运行、核验与重新评分

```bash
bash scripts/tools/sh/46_run_browsecomp_plus.sh
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
| `results/predictions.jsonl` | 完整 `messages`、逐轮 `turn_stats`、检索 docid、工具调用计数和最终回答。开启 CM 时另有 `final_messages` 和 `masked_observations`。 |
| `results/manifest.json` / `evaluation.json` | 题目范围、检索索引、模型、运行参数和预测指纹。 |
| `scores/<评分记录>/results/judge.jsonl` | 逐题语义裁判、失败原因、召回与引用判定。 |
| `scores/<评分记录>/results/summary.metrics.json` | 全部已选题作为分母的语义准确率、证据与金标准文档召回、每题工具调用统计。全量时应核对 `count=830`、`subset=false`、`judge_errors=0`。 |

生成成功后，如果裁判服务中断或需要更换裁判，将该次**评测根目录**填入 `SA_EVALUATION_DIR` 并保留相同的 `SA_METHOD`。同一脚本会读取原预测并创建新评分目录。要扩展一个已完成的测试，清空 `SA_EVALUATION_DIR`，将原评测填入 `SA_RESUME_FROM`，提高 `LIMIT`，并沿用原生成参数。 [续测与中断恢复](36_browsecomp_plus_resume.md) · [逐题报告](33_checkpoint_evaluation.md#3-核对结果并评分)
