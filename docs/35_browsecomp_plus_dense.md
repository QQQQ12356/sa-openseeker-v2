# 35 BrowseComp-Plus 使用官方 BM25 和 Qwen3-Embedding-8B 检索

使用默认 `v2` 数据时，`bm25` 用 Pyserini 读取 Tevatron 官方预建 Lucene 索引，`qwen3-embedding-8b` 用本地模型编码查询并使用 CPU FAISS `IndexFlatIP` 排序。两种检索器共用 SQLite 文档库，通过原始 docid 读取正文并计算召回。旧版或自定义语料仍可用 SQLite FTS5 排序。

官方向量导入复用四个已编码的分片。分片约 1.64 GB，另需模型权重和转换产物的空间。自定义语料可使用 `encode`。评测直接在本地进程中加载 Lucene 或 FAISS。 [官方 Lucene 索引](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes/tree/main/bm25) · [官方向量分片](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes/tree/main/qwen3-embedding-8b) · [参考检索实现](https://github.com/i-DeepSearch/observation-masking/blob/main/tools/backend.py)

## 1. 准备题目、语料和文档库

在已激活的 Linux 评测环境中安装检索依赖。官方 BM25 需要 Java 21，确保 `java -version` 显示 21 且 `JAVA_HOME` 指向同一安装。

```bash
python -m pip install -r requirements.browsecomp-plus.txt
java -version
```

用 [BrowseComp-Plus 数据准备流程](20_data.md#8-准备-browsecomp-plus)完成 `31_prepare_browsecomp_plus.sh` 的 `download → prepare → index` 三个阶段。如果对应版本已完成，可直接复用。三种评测入口的 `SA_BENCHMARK_DIR` 和下文 33 脚本的 `SA_VERSION` 应指向同一数据版本。

向量准备默认读取该版本 `prepared/` 下的 `manifest.json` 和 `corpus.jsonl`。可用 `SA_PREPARED_DIR` 复用其他位置的已准备语料。评测仍需要与该语料匹配的 `index.sqlite` 和 `index.manifest.json`，其中保存按原 `docid` 读取的正文。官方导入要求完整的官方语料及完全匹配的文档 ID。指定自定义或裁剪语料时，走自行编码路径。

## 2. 准备向量索引

在 Linux 上编辑并执行：

```bash
bash scripts/data/sh/33_prepare_browsecomp_plus_dense.sh
```

1. 设置 `SA_PHASE=download`、`SA_SOURCE=official` 和 `HF_HUB_OFFLINE=0` 后运行。记录打印的模型 commit 目录和 `SA_OFFICIAL_INDEX_DIR`。模型保存在 `artifacts/models/qwen3-embedding-8b/<commit>/`，官方文档向量快照保存在目标 benchmark 版本的 `raw/qwen3-embedding-8b/<commit>/`。
2. 改为 `SA_PHASE=prepare`。填写 `SA_EMBEDDING_MODEL_PATH` 和 `SA_OFFICIAL_INDEX_DIR`，可将 `HF_HUB_OFFLINE` 设为 `1`，再运行同一入口。
3. 使用打印的 `SA_GENERATION__VECTOR_INDEX_DIR`。其目录结构为 `artifacts/benchmarks/browsecomp-plus/<VERSION>/vector_indexes/<INDEX_PROFILE>/`，其中 `vectors.npy` 是向量矩阵，`docids.json` 是行号到原文档 ID 的映射，`manifest.json` 记录来源、模型、语料摘要、编码配置和文件 SHA256。

自行编码时，从第一步开始设 `SA_SOURCE=encode`。下载阶段只下载 embedding 模型。准备阶段从本地 prepared 语料逐批编码每篇文档的正文，并按 `SA_DOC_MAX_LENGTH` 截取编码输入。每篇文档对应一个向量，原始正文和访问接口继续由 SQLite 提供。换用新 `SA_INDEX_PROFILE` 以便保留已有产物。失败后检查该次 `jobs/` 日志。若输出目录已留下部分文件，请换新 profile 重跑。

### 准备脚本参数

| 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `PYTHON_BIN` | 当前环境的 `python` | 使用装有仓库训练依赖的 Linux 环境。也可在启动前设置解释器绝对路径。 |
| `HF_ENDPOINT` | `https://huggingface.co` | 下载使用的 Hub 地址，可改成 Linux 可访问的镜像。 |
| `HF_HUB_OFFLINE` | `0` | 下载时使用 0。准备阶段仅访问本地文件，可设 1。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 自行编码使用的物理 GPU 编号。导入官方向量可在 CPU 任务上运行。 |
| `SA_VERSION` | `v2` | 题目和语料版本，与数据准备 31 及评测的 benchmark 目录一致。 |
| `SA_INDEX_PROFILE` | `qwen3-embedding-8b-official` | 向量产物版本名。只允许字母、数字、下划线和连字符；改变输入或配置时换名。 |
| `SA_PHASE` | `download` | `download` 下载权重和所需官方向量；`prepare` 转换官方向量或自行编码。 |
| `SA_SOURCE` | `official` | `official` 导入四个官方分片；`encode` 对本地 prepared 语料自行编码。 |
| `SA_MODEL_REVISION` | `main` | 下载阶段的 Qwen3-Embedding-8B revision。正式实验可填固定 commit；程序对 `main` 解析并记录实际 commit。 |
| `SA_INDEX_REVISION` | 空 | 官方向量下载沿用同版本 31 记录的索引 commit。显式填写时须与 31 一致。 |
| `SA_EMBEDDING_MODEL_PATH` | 空 | 准备阶段必填。填模型下载输出的 commit 目录，包含模型文件和 `download_manifest.json`。同目录用于评测的查询编码。 |
| `SA_OFFICIAL_INDEX_DIR` | 空 | 官方导入必填。填向量下载输出的快照根目录，目录内有 `download_manifest.json`。自行编码可留空。 |
| `SA_PREPARED_DIR` | 空 | 仅 `prepare` 使用，适用于 `official` 和 `encode`。填含 `manifest.json` 和 `corpus.jsonl` 的目录。相对路径以项目根目录为准。留空使用 `artifacts/benchmarks/browsecomp-plus/<VERSION>/prepared/`。向量索引的输出位置仍由 `SA_VERSION` 和 `SA_INDEX_PROFILE` 决定。 |
| `SA_DEVICE` | `cuda:0` | 仅用于自行编码。`cuda:0` 是第一张可见 GPU，使用 FP16；`cpu` 用 FP32，编码通常更慢。 |
| `SA_BATCH_SIZE` | `4` | 自行编码的每批文档数。遇到显存不足先减小该值。 |
| `SA_DOC_MAX_LENGTH` | `4096` | 自行编码的单篇输入 token 上限，范围 1–32768。影响向量覆盖的文本范围。 |

## 3. 运行评测和评分

一次完成生成与评分时，把下述向量参数填入 [对应方法的 46 脚本](37_browsecomp_plus_run.md#选择检索器)。也可分别使用 42 和 43。

编辑对应方法的 `scripts/plain/sh/42_test_browsecomp_plus.sh`、`scripts/beacon/sh/42_test_browsecomp_plus.sh` 或 `scripts/baseline/sh/42_test_browsecomp_plus.sh`。训练检查点和提示词设置继续按 [检查点测试说明](33_checkpoint_evaluation.md) 填写。向量模式增加以下四项：

```bash
export SA_GENERATION__RETRIEVER=qwen3-embedding-8b
export SA_GENERATION__VECTOR_INDEX_DIR=artifacts/benchmarks/browsecomp-plus/v2/vector_indexes/qwen3-embedding-8b-official
export SA_GENERATION__EMBEDDING_MODEL_PATH=/path/to/embedding-model-commit
export SA_GENERATION__EMBEDDING_DEVICE=cuda:0
```

| 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `SA_GENERATION__RETRIEVER` | `bm25` | 可改为 `qwen3-embedding-8b` 启用向量检索。 |
| `SA_GENERATION__VECTOR_INDEX_DIR` | 空 | 向量模式必填，填写 33 输出的向量目录。 |
| `SA_GENERATION__EMBEDDING_MODEL_PATH` | 空 | 向量模式必填，填写与向量清单绑定的本地模型 commit 目录。 |
| `SA_GENERATION__EMBEDDING_DEVICE` | `cuda:0` | 查询编码设备。每个生成进程各加载一份 embedding 模型，且每个 worker 只见到其分配 GPU。多 worker 用 `cuda:0` 表示各自的卡。单 worker 显存紧张可改为 `cpu`，需要为 FP32 权重及推理留足内存。 |
| `SA_GENERATION__VERIFY_INDEX` | `true` | 校验 SQLite 及所选的 Lucene 或向量资产 SHA256。关闭时仍核对来源和文件清单。 |
| `SA_GENERATION__TITLE_WEIGHT` | `1.0` | 用于旧版或自定义语料的 SQLite FTS5 排序。 |
| `SA_GENERATION__TOPK` | `15`（46）；`5`（42） | 每条查询取回的不同文档数。 |
| `SA_GENERATION__SNIPPET_TOKENS` / `SA_GENERATION__READ_TOKENS` | `512` / `4096` | 分别控制每篇搜索摘录和 visit 正文的 token 预算。 |

首次以 `SA_GENERATION__NUM_WORKERS=1` 和 `SA_GENERATION__LIMIT=5` 跑小样本，检查 `results/manifest.json` 的 `index.retriever`、`index.ranking` 和向量模式下的 `index.vector_index`。每个 worker 都会加载一份 CPU FAISS 索引，全量向量约 1.53 GiB／worker；各 GPU 还需容纳 Agent 和查询编码器。确认小样本能生成和检索后，再增大题数或 worker 数。

评分继续使用同方法的 `43_score_browsecomp_plus.sh`，将 `SA_EVALUATION_DIR` 指向刚生成的测试记录目录。它复用已生成的逐题预测和题目范围。正式对照时，固定 checkpoint、题目范围、重合筛选、提示词、TopK、工具观测预算与解码参数，仅切换检索器。比较答案准确率、`evidence_recall`、`gold_recall` 和推理耗时。

## 4. 索引核验与排错

- 向量准备检查所有分片的 SHA256、向量维度和归一化，以及向量与 prepared 文档 ID 的完整对应关系。评测核对 SQLite 和向量所绑定的同一份语料摘要，以及查询模型快照。官方向量导入的文档分词与截断以官方分片为准。
- 遇到语料摘要或文档 ID 不匹配，检查 31 和 33 的 `SA_VERSION`、评测 `SA_BENCHMARK_DIR` 以及向量目录。自定义语料使用 `encode` 生成自身的向量索引。
- 模型快照不匹配时，恢复构建向量用的模型目录或使用新 profile 重新准备。评测中的 `SA_MODEL_PATH` 或 `SA_TRAINING_RUN` 仍负责选择作答模型。
- GPU 显存不足时，先降准备阶段的 `SA_BATCH_SIZE`。评测阶段可让查询编码使用 CPU，或使用能同时容纳 Agent 与 embedding 模型的 GPU。
