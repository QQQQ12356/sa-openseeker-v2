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
| BrowseComp SFT | `scripts/data/sh/12_download_browsecomp_sft.sh` | 固定 `browsecomp-sft` | `AmanPriyanshu/tool-reasoning-sft-RESEARCH-grill-lab-browsecomp-plus-runs-data-cleaned-rectified` |

编辑 `REVISION` 和镜像设置，再运行选定文件：

```bash
bash scripts/data/sh/10_download_openseeker.sh
bash scripts/data/sh/11_download_research.sh
```

每个下载清单记录实际 commit、原文件大小与摘要、导出的全部 `jsonl_inputs`。Parquet 保留原件并导出 JSONL；后续通过清单读取所有 shard。已有相同 commit 的原始快照可续用，操作日志仍独立保存。

BrowseComp SFT 下载入口保留两个 Parquet 原件，清单通过 `parquet_inputs` 指向原件。预处理直接批量读取 Parquet，节省额外的 JSONL 导出空间。接入步骤与字段说明见 [BrowseComp SFT](21_browsecomp_sft.md)。

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
5. `SA_CORRECTNESS`、`SA_OBSERVATIONS`：按标签和工具观测筛选。OpenSeeker 与 Research 均支持 `Correct`、`Incorrect`、`any`；Research 默认仍为 `any`。
6. `SA_VAL_FRACTION`、`SA_TEST_FRACTION`、`SA_SEED`：默认约 96% / 2% / 2%，按问题哈希分组，避免同题跨集合。

```bash
bash scripts/data/sh/20_prepare_openseeker.sh
bash scripts/data/sh/21_prepare_research.sh
```

仅运行所需数据入口。结果位于 `artifacts/datasets/<dataset>/prepared/<PROFILE>/`，包含 `normalized.jsonl`、三个划分、拒绝记录、计长记录和 `manifest.json`。公开的 validation/test 问题优先保留原划分，实际比例为近似值。Research 保留轨迹不是独立 benchmark 标准答案。

Research 优先读取原始字段 `trajectory_correctness`，该字段缺失时兼容旧字段 `trajectory correctness`。OpenSeeker 读取 `trajectory correctness`。规范化统一保存为 `correctness`；筛选不重新判断答案对错。

| `SA_CORRECTNESS` | 保留条件 | 修改提示 |
|---|---|---|
| `Correct` | 标签精确等于 `Correct` | 区分大小写；缺失、空值或未知标签会被过滤 |
| `Incorrect` | 标签精确等于 `Incorrect` | 仅保留明确标记的错误轨迹；缺失标签不视为错误 |
| `any` | 不按正确性过滤 | 包含缺失/未知标签；仍应用长度、工具观测和去重规则 |

例如，仅保留 Research 的正确轨迹时，在 `21_prepare_research.sh` 中设置 `export SA_CORRECTNESS=Correct`，并填写新的 `SA_PROFILE`。若原始数据没有对应标签，筛选结果可能为空；可查看原预处理版本的 manifest.correctness、筛选计数及 rejected.jsonl。不要根据 demonstration_final_response 或 answer_source 推断正确性。

## 4. 可选：重新筛选

使用 `scripts/data/sh/22_select_openseeker.sh` 或 `scripts/data/sh/23_select_research.sh`。设置父版本 `SA_PREPARED`、新 `SA_PROFILE`、`SA_TOKENIZER`、长度和筛选条件后直接运行。Research 二次筛选同样支持 `SA_CORRECTNESS=Correct/Incorrect/any`，使用父版本 normalized.jsonl 中已有的 correctness 标签。程序不重新分词，恢复公开划分并记录父版本。父版本只按预处理时的 tokenizer 计过长度；新增 tokenizer 必须重新预处理。

筛选结果不复制全量 normalized 文件；如要继续扩大/改变范围，应仍从拥有 normalized 文件的原始预处理版本筛选。保留父目录及其原始输入，保证公开划分可恢复。

如果旧版 Research 预处理未识别 `trajectory_correctness`，请使用相同原始数据和新的 `SA_PROFILE` 重新运行 `21_prepare_research.sh`，以重新读取标签。

## 5. 可选：混合 OpenSeeker 与 Research

先分别完成两类数据的预处理。若两边需要不同的正确性或工具观测条件，先各自筛选，再把选好的两个目录填入 [24_mix_datasets.sh](../scripts/data/sh/24_mix_datasets.sh)。可使用原始 prepared 版本或 22/23 二次筛选产物。每个父目录需要 `manifest.json`、`train.jsonl`、`validation.jsonl` 和 `test.jsonl`。

| SH 变量 | 作用及修改提示 |
|---|---|
| `PYTHON_BIN` | Linux Python 解释器，默认使用当前环境中的 `python`。也可在启动前设为环境的 `bin/python` 绝对路径。 |
| `SA_OPENSEEKER_PREPARED` | OpenSeeker 父版本路径，来源可为 polar、zephyr 或 combined。 |
| `SA_RESEARCH_PREPARED` | Research 父版本路径。两来源模式可填普通 Research 或 BrowseComp SFT。三来源模式填写普通 Research。 |
| `SA_BROWSECOMP_SFT_PREPARED` | 可选第三个父版本路径，填写 `25_prepare_browsecomp_sft.sh` 产物。留空使用两来源。 |
| `SA_PROFILE` | 输出版本名。调整父版本、比例、长度或 seed 后使用新名称。 |
| `SA_TOKENIZER` | 各父版本清单中共同的 tokenizer 键，须与训练的 `SA_MODEL_PATH` 一致。混合时比较计长身份与 tokenizer 指纹。 |
| `SA_MAX_LENGTH` | 完整轨迹 token 上限，正整数。超长整条排除。调高上限只能使用父划分里已有的轨迹。 |
| `SA_RATIO` | 按训练**轨迹数**指定各来源的正整数比例。两来源顺序 OpenSeeker:Research，默认 `1:1`。三来源顺序 OpenSeeker:Research:BrowseComp SFT，例如 `2:1:1`。 |
| `SA_SEED` | 整数随机种子，控制无放回抽样和训练文件顺序。固定父文件、参数和 Python 环境时可复现。 |

```bash
bash scripts/data/sh/24_mix_datasets.sh
```

程序按问题文本归一化后的哈希处理跨源重合，优先级为 `test > validation > train`。同一问题只保留最高优先级划分内的轨迹。即使高优先级轨迹超长，同题的低优先级轨迹也会被排除。保留下来的验证和测试轨迹全量写出。训练集按约分后的比例取可组成的最大整数批次，任一来源不足以形成一批时会报错。例如可用量为 101 / 40，比例为 `2:1` 时输出 80 / 40 条。比例约束样本数，两边 token 占比会随长度分布变化。

输出为 `artifacts/datasets/mixed/prepared/<PROFILE>/`，包含三个划分和 `manifest.json`。清单记录父目录及文件摘要、目标比例、实际训练样本数之比、各划分的来源样本数及 token 数，以及跨划分冲突和超长排除数。混合沿用父版本中已保存的轨迹和计长，保留每条轨迹的 system 与工具定义。

训练时编辑对应方法的 `scripts/plain/sh/11_train_research.sh` 或 `scripts/beacon/sh/11_train_research.sh`，设置 `SA_DATASET_NAME=mixed`、`SA_DATASET_DIR` 为输出目录、`SA_MODEL_PATH` 为混合时的 tokenizer 键，并把 `SA_TRAINING__MAX_LENGTH` 设为至少等于混合上限。实验存放在 `artifacts/experiments/<方法>/mixed/`。评测时填写该实验的 `SA_TRAINING_RUN`，并通过 `SA_PROMPT_DATA` 选择一套相同 system 和工具定义的规范化 JSONL，例如确认提示词一致的一个父版本训练文件。详见 [checkpoint 评测](33_checkpoint_evaluation.md)。

## 6. 可选：检查数据

编辑 `scripts/tools/sh/80_dataset_report.sh` 的 `SA_DATASET_DIR`、`SPLIT`、`LIMIT` 后运行：

```bash
bash scripts/tools/sh/80_dataset_report.sh
```

HTML 保存在该数据版本的 `reports/` 下。全量计长分布和筛选计数位于原 `manifest.json`；HTML 展示数量由 `LIMIT` 决定。

## 7. 准备 BrowseComp

编辑 `scripts/data/sh/30_prepare_browsecomp.sh` 的 `SA_VERSION`、镜像和可选 `SA_CSV`，保持 `SA_PHASE='download'` 后运行：

```bash
bash scripts/data/sh/30_prepare_browsecomp.sh
```

题目从官方加密 CSV 解密，生成 `questions.jsonl` 及来源清单；OpenSeeker 工具固定至代码声明的 commit，保存在该 benchmark 版本的 `vendor/OpenSeeker/`。BrowseComp 使用在线搜索，不建立本地语料索引。

## 8. 准备 BrowseComp-Plus

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

## 9. 准备 HotpotQA

使用 `scripts/data/sh/32_prepare_hotpotqa.sh`，同样按 `download → prepare → index` 分三次编辑并运行：

```bash
bash scripts/data/sh/32_prepare_hotpotqa.sh
```

使用 fullwiki dev 题目与独立维基摘要语料，校验官方语料 MD5。程序不会将题目提供的候选 context 作为检索语料，也不会将答案或 supporting facts 注入模型提示词。本地小语料结果会标记为非正式全量语料。

## 10. 复用规则

三种方法共享同一 benchmark 版本，保证题目、语料和索引一致。数据目录禁止覆盖；操作失败后保留日志，原始下载可重试，部分 prepared/index 建议保留后改用新版本重建。进入训练与测试前，把生成的准确目录填入方法脚本。
