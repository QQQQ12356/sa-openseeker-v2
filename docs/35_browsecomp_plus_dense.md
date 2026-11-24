# 35 BrowseComp-Plus 使用 Qwen3-Embedding-8B 检索

Plain、Beacon 和 Baseline 的 BrowseComp-Plus 测试支持两种检索器。默认 `bm25` 继续查询 SQLite FTS5；`qwen3-embedding-8b` 用本地 embedding 模型编码每条查询，在完整语料的向量中按内积排序。两种模式均用同一份 SQLite 文档库交付正文和处理 `visit(local://<docid>)`。生成答案用的是原先选定的 Agent 模型，embedding 模型只用于检索。

优先走**官方向量导入**，可省去重新编码十万篇文档的时间。Tevatron 的官方 Qwen3-Embedding-8B 向量由四个分片组成，下载约 1.64 GB，另需模型权重的存储空间。需要测试自定义语料或自行控制文档编码时，再使用 `encode`。向量检索运行时使用 NumPy 内存映射和精确内积，不需要额外的 FAISS 安装或单独的网络服务。 [官方向量分片](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes/tree/main/qwen3-embedding-8b) · [官方检索实现](https://github.com/texttron/BrowseComp-Plus/blob/main/searcher/searchers/faiss_searcher.py) · [模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

## 1. 准备题目、语料和文档库

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
| `SA_VERSION` | `v1` | 题目和语料版本，与数据准备 31 及评测的 benchmark 目录一致。 |
| `SA_INDEX_PROFILE` | `qwen3-embedding-8b-official` | 向量产物版本名。只允许字母、数字、下划线和连字符；改变输入或配置时换名。 |
| `SA_PHASE` | `download` | `download` 下载权重和所需官方向量；`prepare` 转换官方向量或自行编码。 |
| `SA_SOURCE` | `official` | `official` 导入四个官方分片；`encode` 对本地 prepared 语料自行编码。 |
| `SA_MODEL_REVISION` | `main` | 下载阶段的 Qwen3-Embedding-8B revision。正式实验可填固定 commit；程序对 `main` 解析并记录实际 commit。 |
| `SA_INDEX_REVISION` | `980cfd0c8250482e8be900326fddca806511d05c` | 官方向量快照 revision，只有 `official` 下载阶段使用。切换时检查分片、模型和语料是否匹配。 |
| `SA_EMBEDDING_MODEL_PATH` | 空 | 准备阶段必填。填模型下载输出的 commit 目录，包含模型文件和 `download_manifest.json`。同目录用于评测的查询编码。 |
| `SA_OFFICIAL_INDEX_DIR` | 空 | 官方导入必填。填向量下载输出的快照根目录，目录内有 `download_manifest.json`。自行编码可留空。 |
| `SA_PREPARED_DIR` | 空 | 仅 `prepare` 使用，适用于 `official` 和 `encode`。填含 `manifest.json` 和 `corpus.jsonl` 的目录。相对路径以项目根目录为准。留空使用 `artifacts/benchmarks/browsecomp-plus/<VERSION>/prepared/`。向量索引的输出位置仍由 `SA_VERSION` 和 `SA_INDEX_PROFILE` 决定。 |
| `SA_DEVICE` | `cuda:0` | 仅用于自行编码。`cuda:0` 是第一张可见 GPU，使用 FP16；`cpu` 用 FP32，编码通常更慢。 |
| `SA_BATCH_SIZE` | `4` | 自行编码的每批文档数。遇到显存不足先减小该值。 |
| `SA_DOC_MAX_LENGTH` | `4096` | 自行编码的单篇输入 token 上限，范围 1–32768。影响向量覆盖的文本范围。 |

## 3. 运行评测和评分

一次完成生成与评分时，把下述向量参数填入 [46 共用入口](37_browsecomp_plus_run.md#选择检索器)。以下说明适用于分别运行 42、43 的情况。

编辑对应方法的 `scripts/plain/sh/42_test_browsecomp_plus.sh`、`scripts/beacon/sh/42_test_browsecomp_plus.sh` 或 `scripts/baseline/sh/42_test_browsecomp_plus.sh`。训练检查点和提示词设置继续按 [检查点测试说明](33_checkpoint_evaluation.md) 填写。向量模式增加以下四项：

```bash
export SA_GENERATION__RETRIEVER=qwen3-embedding-8b
export SA_GENERATION__VECTOR_INDEX_DIR=artifacts/benchmarks/browsecomp-plus/v1/vector_indexes/qwen3-embedding-8b-official
export SA_GENERATION__EMBEDDING_MODEL_PATH=/path/to/embedding-model-commit
export SA_GENERATION__EMBEDDING_DEVICE=cuda:0
```

| 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `SA_GENERATION__RETRIEVER` | `bm25` | 可改为 `qwen3-embedding-8b` 启用向量检索。 |
| `SA_GENERATION__VECTOR_INDEX_DIR` | 空 | 向量模式必填，填写 33 输出的向量目录。 |
| `SA_GENERATION__EMBEDDING_MODEL_PATH` | 空 | 向量模式必填，填写与向量清单绑定的本地模型 commit 目录。 |
| `SA_GENERATION__EMBEDDING_DEVICE` | `cuda:0` | 查询编码设备。每个生成进程各加载一份 embedding 模型，且每个 worker 只见到其分配 GPU。多 worker 用 `cuda:0` 表示各自的卡。单 worker 显存紧张可改为 `cpu`，需要为 FP32 权重及推理留足内存。 |
| `SA_GENERATION__VERIFY_INDEX` | `true` | 校验 SQLite 文件和向量产物的 SHA256。关闭时仍核对来源、维度和模型清单。 |
| `SA_GENERATION__TITLE_WEIGHT` | `1.0` | BM25 排序使用。向量模式仍读取 SQLite 文档，向量排名不使用此权重。 |
| `SA_GENERATION__TOPK` | `5` | 每条查询取回的不同文档数。 |
| `SA_GENERATION__SNIPPET_TOKENS` / `SA_GENERATION__READ_TOKENS` | `512` / `4096` | 分别控制每篇搜索摘录和 visit 正文的 token 预算。 |

首次以 `SA_GENERATION__NUM_WORKERS=1` 和 `SA_GENERATION__LIMIT=5` 跑小样本，检查日志和 `results/manifest.json` 的 `index.retriever`、`index.vector_index`。CPU 对文档向量做精确内积，多个 worker 可共享底层内存映射的页缓存；每个 GPU 仍需容纳 Agent、embedding 模型和各自的运行时内存。确认小样本能生成和检索后，再增大题数或 GPU worker 数量。

评分继续使用同方法的 `43_score_browsecomp_plus.sh`，将 `SA_EVALUATION_DIR` 指向刚生成的测试记录目录。它复用已生成的逐题预测和题目范围。正式对照时，固定 checkpoint、题目范围、重合筛选、提示词、TopK、工具观测预算与解码参数，仅切换检索器。比较答案准确率、`evidence_recall`、`gold_recall` 和推理耗时。

## 4. 索引核验与排错

- 向量准备检查所有分片的 SHA256、向量维度和归一化，以及向量与 prepared 文档 ID 的完整对应关系。评测核对 SQLite 和向量所绑定的同一份语料摘要，以及查询模型快照。官方向量导入的文档分词与截断以官方分片为准。
- 遇到语料摘要或文档 ID 不匹配，检查 31 和 33 的 `SA_VERSION`、评测 `SA_BENCHMARK_DIR` 以及向量目录。自定义语料使用 `encode` 生成自身的向量索引。
- 模型快照不匹配时，恢复构建向量用的模型目录或使用新 profile 重新准备。评测中的 `SA_MODEL_PATH` 或 `SA_TRAINING_RUN` 仍负责选择作答模型。
- GPU 显存不足时，先降准备阶段的 `SA_BATCH_SIZE`。评测阶段可让查询编码使用 CPU，或使用能同时容纳 Agent 与 embedding 模型的 GPU。
