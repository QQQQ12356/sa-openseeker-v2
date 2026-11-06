# 20 模型、训练数据与 benchmark 准备

本文字段名称表示逻辑参数：SH 中普通变量加 `SA_` 前缀，例如 `MODEL_PATH` 对应 `SA_MODEL_PATH`；嵌套键用双下划线，例如 `TRAINING.max_length` 对应 `SA_TRAINING__MAX_LENGTH`。环境变量如 `HF_ENDPOINT` 不加前缀。实际可编辑名称、类型与默认值见 [61 脚本参数清单](61_script_inventory.md)。

## 1. 准备模型

编辑 `scripts/data/sh/01_download_model.sh` 的 `REPO_ID`、`REVISION`、`MODEL_NAME` 和 `HF_ENDPOINT` 后运行：

```bash
bash scripts/data/sh/01_download_model.sh
```

程序将 revision 解析为实际 commit，保存至 `artifacts/models/<MODEL_NAME>/<commit>/`，保留文件摘要。将终端输出的本地目录填入所有预处理脚本的 `SA_MODELS` 及训练脚本的 `SA_MODEL_PATH`。二者使用完全相同的路径字符串。裁判模型可用同一下载脚本另取一个 `MODEL_NAME` 准备。

如果模型已在 Linux 本地，不必下载；直接填写真实目录。

## 2. 下载训练数据

| 数据身份 | 下载入口 | `SA_SOURCE` | 上游数据 |
|---|---|---|---|
| OpenSeeker | `scripts/data/sh/10_download_openseeker.sh` | `polar`，可改 `zephyr` | `PolarSeeker/OpenSeeker-v1-Data` / `Zephyr271828/openseeker_v1_sft` |
| Research | `scripts/data/sh/11_download_research.sh` | 固定 `research` | `AmanPriyanshu/tool-reasoning-sft-RESEARCH-OpenSeeker-v1-Data` |

编辑 `REVISION` 和镜像设置，再运行选定文件：

```bash
bash scripts/data/sh/10_download_openseeker.sh
bash scripts/data/sh/11_download_research.sh
```

每个下载清单记录实际 commit、原文件大小与摘要、导出的全部 `jsonl_inputs`。Parquet 保留原件并导出 JSONL；后续通过清单读取所有 shard。已有相同 commit 的原始快照可续用，操作日志仍独立保存。

## 3. 预处理与划分

| 数据身份 | 入口 | 默认筛选 | 默认数据版本 |
|---|---|---|---|
| OpenSeeker | `scripts/data/sh/20_prepare_openseeker.sh` | `Correct`，观测不限 | `qwen35_L16384_s42` |
| Research | `scripts/data/sh/21_prepare_research.sh` | 正确性不限，有工具观测 | `qwen35_L16384_s42` |

依次修改：

1. `SA_SNAPSHOT_DIR`：填写下载输出的 commit 目录。也可留空并用 `SA_INPUTS` 列出本地 JSONL/Parquet；二者互斥。
2. `SA_MODELS`：填写后续训练使用的 Linux 本地 tokenizer 路径；不会加载模型权重。
3. `SA_PROFILE`：选择新的数据版本名；改变来源、tokenizer、长度、筛选或划分时使用新版本。
4. `SA_MIN_LENGTH`、`SA_MAX_LENGTH`：限制完整轨迹 token 长度，超长整条过滤，禁止截断。
5. `SA_CORRECTNESS`、`SA_OBSERVATIONS`：按标签和工具观测筛选。Research 只能使用 `CORRECTNESS='any'`。
6. `SA_VAL_FRACTION`、`SA_TEST_FRACTION`、`SA_SEED`：默认约 96% / 2% / 2%，按问题哈希分组，避免同题跨集合。

```bash
bash scripts/data/sh/20_prepare_openseeker.sh
bash scripts/data/sh/21_prepare_research.sh
```

仅运行所需数据入口。结果位于 `artifacts/datasets/<dataset>/prepared/<PROFILE>/`，包含 `normalized.jsonl`、三个划分、拒绝记录、计长记录和 `manifest.json`。公开的 validation/test 问题优先保留原划分，实际比例为近似值。Research 保留轨迹不是独立 benchmark 标准答案。

## 4. 可选：重新筛选

使用 `scripts/data/sh/22_select_openseeker.sh` 或 `23_select_research.py`。设置父版本 `SA_PREPARED`、新 `SA_PROFILE`、`SA_TOKENIZER`、长度和筛选条件后直接运行。程序读取父版本实测 `normalized.jsonl`，不重新分词，恢复公开划分并记录父版本。父版本只按预处理时的 tokenizer 计过长度；新增 tokenizer 必须重新预处理。

筛选结果不复制全量 normalized 文件；如要继续扩大/改变范围，应仍从拥有 normalized 文件的原始预处理版本筛选。保留父目录及其原始输入，保证公开划分可恢复。

## 5. 可选：检查数据

编辑 `scripts/tools/sh/80_dataset_report.sh` 的 `SA_DATASET_DIR`、`SPLIT`、`LIMIT` 后运行：

```bash
bash scripts/tools/sh/80_dataset_report.sh
```

HTML 保存在该数据版本的 `reports/` 下。全量计长分布和筛选计数位于原 `manifest.json`；HTML 展示数量由 `LIMIT` 决定。

## 6. 准备 BrowseComp

编辑 `scripts/data/sh/30_prepare_browsecomp.sh` 的 `SA_VERSION`、镜像和可选 `SA_CSV`，保持 `SA_PHASE='download'` 后运行：

```bash
bash scripts/data/sh/30_prepare_browsecomp.sh
```

题目从官方加密 CSV 解密，生成 `questions.jsonl` 及来源清单；OpenSeeker 工具固定至代码声明的 commit，保存在该 benchmark 版本的 `vendor/OpenSeeker/`。BrowseComp 使用在线搜索，不建立本地语料索引。

## 7. 准备 BrowseComp-Plus

使用 `scripts/data/sh/31_prepare_browsecomp_plus.sh`，保持同一个 `SA_VERSION`：

| 执行顺序 | 修改 `SA_PHASE` | 行为 |
|---|---|---|
| 1 | `download` | 下载固定问题版本、独立全量语料和官方评测资产 |
| 2 | `prepare` | 解密问题，规范化语料，保留官方 qrels 和来源摘要 |
| 3 | `index` | 建立持久化 SQLite FTS5 BM25 索引 |

每次编辑一个阶段后执行同一文件：

```bash
bash scripts/data/sh/31_prepare_browsecomp_plus.sh
```

已有本地资产时填写 `SA_DATA` / `SA_CORPUS`；已解密题目设 `SA_DECRYPTED=true`。修改来源、语料 revision 或 `SA_TITLE_WEIGHT` 时新建 `SA_VERSION`；测试脚本同步该版本和标题权重。正式测试使用完整独立语料。

## 8. 准备 HotpotQA

使用 `scripts/data/sh/32_prepare_hotpotqa.sh`，同样按 `download → prepare → index` 分三次编辑并运行：

```bash
bash scripts/data/sh/32_prepare_hotpotqa.sh
```

使用 fullwiki dev 题目与独立维基摘要语料，校验官方语料 MD5。程序不会将题目提供的候选 context 作为检索语料，也不会将答案或 supporting facts 注入模型提示词。本地小语料结果会标记为非正式全量语料。

## 9. 复用规则

三种方法共享同一 benchmark 版本，保证题目、语料和索引一致。数据目录禁止覆盖；操作失败后保留日志，原始下载可重试，部分 prepared/index 建议保留后改用新版本重建。进入训练与测试前，把生成的准确目录填入方法脚本。
