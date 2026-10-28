# 43 固定语料离线评测：BrowseComp-Plus 与 HotpotQA

本文是新增离线流程的操作手册。所有修改均为增量：原来的 21/31 EM/F1、47 在线 OpenSeeker、48 SGLang、Search-R1 和训练器继续保留。新增实现位于 `search_comp/offline/`，统一 Python 入口为 `scripts/py/52_offline_benchmark.py`，一键入口是 **56 BrowseComp-Plus** 和 **57 HotpotQA**。

## 43.1 两套基准的用途与边界

| 项目 | BrowseComp-Plus | HotpotQA |
|---|---|---|
| 使用划分 | 官方 test 问题集 | 有答案与支持事实的 dev / HF fullwiki validation |
| 官方规模参考 | 830 题、100,195 篇固定网页文档 | 7,405 道 dev 问题；索引完整官方 2017 Wikipedia 摘要语料 |
| 核心能力 | 多轮复杂搜索与证据整合 | 多跳检索、短答案及支持事实定位 |
| 本地检索 | SQLite FTS5 持久化 BM25 | 同左 |
| 本地阅读 | 默认 search-only，可选择 local visit 扩展 | 默认 search + local visit |
| 答案评分 | 官方裁判 prompt/parser + 本地 HF 模型，建议 Qwen3-32B | 执行固定版本官方 EM/F1、SP 和 Joint 指标代码 |
| 是否需要辅助模型 | 完整答案评分需要本地裁判；检索诊断不需要 | 不需要摘要模型或裁判模型 |
| 在线 API | 无 | 无 |

BrowseComp-Plus 的固定语料结果不是在线 BrowseComp 成绩。本实现使用 SQLite FTS5 BM25，而官方 BrowseComp-Plus 检索示例使用 Lucene/Pyserini；分词、IDF 与排序并非逐项相同，因此应报告 **BrowseComp-Plus + SQLite FTS5 BM25 + native agent**，不能称为原论文检索器的逐位复现。Plain 与 Beacon 在本项目内对比时保持同一索引、提示、预算和裁判配置。

HotpotQA 的 **fullwiki 语料**与 `hotpot_dev_fullwiki_v1.json` 的 **context** 不同。后者只是问题附带的候选段落，不能当成全库。本流程忽略 dev context，独立索引官方发布的 2017 Wikipedia introductory paragraphs。它是每篇 Wikipedia 页面的摘要段落库，不是原始完整页面全文库，更不是当前实时 Wikipedia。

官方来源、固定提交、参考文件和许可证见 [44 来源说明](44_offline_sources.md)。本文中的规模来自官方说明；本次只运行离线合成回归与小样本流程，没有实际下载并构建两套全量语料，也没有运行真实 checkpoint 成绩。

## 43.2 先理解三个目录与两种模型

```text
ROOT_DIR/                           # 一个基准一套目录，不与另一基准混用
├── assets.json                     # 官方版本、文件指纹；下载时另记数据修订与来源
├── official/                       # 随代码附带的官方参考文件在此物化，按 SHA256 校验
├── raw/                            # 一次性下载或拷贝到 HPC 的原始数据
├── prepared/
│   ├── questions.jsonl             # 规范化问题、参考答案、仅评分可用的标注
│   ├── corpus.jsonl                # 从独立 corpus 导入的文档，不来自题目的正样本列表
│   └── manifest.json               # 来源、指纹、规模、是否为正式全量语料
├── index.sqlite                    # 可复用的磁盘索引 + 文档正文
└── index.manifest.json              # BM25 配置、SQLite 版本、索引指纹

outputs/experiments/<run>/           # 原有训练产物，RUN_DIR 指向这里
└── models/model/train_summary.json  # 用 final_model_path 定位已训练模型

outputs/results/offline_<...>/       # 本次测试结果；也可显式 OUTPUT_DIR
```

- **待测模型**：Plain 或 Beacon，必须是本地目录且 LoRA 已合并；通过 RUN_DIR 自动定位，也可设置 MODEL_PATH。
- **辅助裁判**：仅 BrowseComp-Plus 完整答案评分使用，与待测模型分开设置 JUDGE_MODEL。被测 Beacon 不负责给自己打分。

不需要 Serper、Tavily、Jina、E2B、SUMMARY_*、SCORER_* 或 OPENSEEKER_*。本地正文按固定 token 预算读取，没有 LLM 摘要。49 的在线 API 配置文件不参与这条流程。

## 43.3 环境与资源准备

使用原有训练环境即可：模型运行依赖 torch、Transformers、accelerate；Parquet 适配依赖 pyarrow；下载依赖 huggingface_hub。这些已在项目 requirements 中。额外的 Java、Pyserini、FAISS、vLLM、OpenAI SDK 和 ujson 不是本地流程的必需依赖。

SQLite 必须启用 FTS5，可先检查：

```bash
python - <<'PY'
import sqlite3
connection = sqlite3.connect(':memory:')
connection.execute('CREATE VIRTUAL TABLE check_fts USING fts5(text)')
print('SQLite', sqlite3.sqlite_version, 'FTS5 ready')
PY
```

56/57 使用现有 `00_common.sh`：设置 PYTHON 的可执行绝对路径优先，其次 CONDA_ENV，默认 `search-comp-qwen3.5`，否则当前 Python。注意脚本不会自动安装依赖。

本地模型目录需要完整的 config、tokenizer、权重分片与索引文件，不能只有一个指向在线模型的名称。BrowseComp-Plus 官方裁判 Qwen3-32B 的 BF16 权重本身约为 64 GB 量级，还需考虑 KV cache 和运行开销；本实现通过 Transformers `device_map=auto` 分配可见 GPU，不能把训练时的 torchrun 配置当成裁判的张量并行参数。

准备下载用的机器需要网络，**prepare / index / run / score / all 均不下载**。这些动作会设置 HF/Transformers offline 环境，并通过 Python audit hook 拒绝 IP socket connect 和 DNS 解析；本流程不使用本地 HTTP 服务，因此也不依赖 localhost API。先下载，再拷贝目录和模型到断网计算节点即可。

完整构建应为 raw、规范化 corpus、SQLite 正文、FTS 倒排索引和临时构建文件预留磁盘空间。不要只按压缩包大小分配配额，也不要把文档数简单当作内存需求；数据适配和索引逐条写入磁盘，不将 fullwiki 全库加载到 Python 列表。索引过程固定使用约 64 MiB 的 SQLite page cache，但操作系统页缓存、FTS 构建缓存和大单篇文档仍会额外占内存。

## 43.4 最快开始：HotpotQA

### 43.4.1 有网络的准备机：一次性下载

```bash
ACTION=download bash scripts/sh/57_hotpotqa_offline.sh
```

这会在默认 ROOT_DIR 下下载：

1. `raw/hotpot_dev_fullwiki_v1.json`。优先使用官方历史链接；历史 CMU 主机不可达时，自动尝试 `hotpotqa/hotpot_qa` 的 fullwiki **validation** Parquet，并生成等价的 id/question/answer/supporting_facts JSON。HF revision 会记录到 assets.json，原 Parquet 留在 raw/dev-hf。不会切换到不带答案的 test。
2. `raw/wiki-abstracts.tar.bz2`。官方摘要包约 1.55 GB，MD5 必须为 `01edf64cd120ecc03a2745352779514c`。不会下载约 7.4 GB 的完整 processed 页面包来替代摘要包。
3. 官方评分源文件和 LICENSE.txt 已随项目附带，首次使用会复制到 ROOT_DIR/official，无需从网络再次获取。

下载失败可重试。普通文件先写 `.part`，完成后改名；不实现 HTTP Range 断点续传。校验失败的已有文件需要先移到别处再重下；下载过程不会把校验失败当作成功。

镜像环境可以使用 Python 参数覆盖下载地址；Wikipedia 镜像仍强制核验相同官方 MD5：

```bash
ACTION=download bash scripts/sh/57_hotpotqa_offline.sh \
  --dev-url https://your-mirror.example/hotpot_dev_fullwiki_v1.json \
  --wiki-url https://your-mirror.example/wiki-abstracts.tar.bz2
```

上述 example 域名不可直接使用。也可在其他机器手动下载，将文件放到默认 raw 路径。若希望在这种手动下载方式下记录官方完整语料身份，再执行 download 会复用已有文件并校验/登记，不会重复下载。

### 43.4.2 断网机器：规范化与构建一次，后续复用

```bash
ACTION=prepare bash scripts/sh/57_hotpotqa_offline.sh
ACTION=index bash scripts/sh/57_hotpotqa_offline.sh
```

支持直接读取 `.tar.bz2` 内部的 `.bz2` JSONL，不解包到任意文件路径。也接受已经解压好的含 bz2 分片目录。对原始 text 的 sentence arrays 去除 HTML 标签，保留正文句子编号；若存在独立标题段则去掉标题段，使正文从 0 编号。

### 43.4.3 训练后测试并评分

```bash
# 首先审查实际参数；不会创建目录、加载模型、校验网络或执行生成。
DRY_RUN=1 RUN_DIR=outputs/experiments/<your-run> \
  bash scripts/sh/57_hotpotqa_offline.sh

# 先做前 2 题；已有 prepared 和 index 时直接复用。
RUN_DIR=outputs/experiments/<your-run> LIMIT=2 \
  bash scripts/sh/57_hotpotqa_offline.sh

# 全量 dev，使用全部已准备的语料。
RUN_DIR=outputs/experiments/<your-run> LIMIT=-1 \
  bash scripts/sh/57_hotpotqa_offline.sh
```

默认 ACTION=all，流程为 prepare（不存在才做）→ index（不存在才做）→ run → score。它不会替你执行 download。无 RUN_DIR 时：

```bash
MODEL_PATH=/absolute/path/to/merged-checkpoint KIND=beacon LIMIT=2 \
  bash scripts/sh/57_hotpotqa_offline.sh
```

KIND 支持 plain/beacon，留空时先读实验配置，再读模型配置 enable_beacon。run/all 会先检查模型路径；只有 adapter_config.json 的目录会拒绝。合并方法沿用 35 脚本。

## 43.5 BrowseComp-Plus：下载、解密、建库和评分

### 43.5.1 一次性下载到本地

```bash
ACTION=download bash scripts/sh/56_browsecomp_plus.sh
```

它使用 Hugging Face snapshot 下载：

- `Tevatron/browsecomp-plus`：固定 query revision `144cff8e35b5eaef7e526346aa60774a9deb941f`，下载到 raw/queries；含加密问题、答案及较大的嵌套文档数据，不能仅按“830 题”估计下载体积。
- `Tevatron/browsecomp-plus-corpus`：首次显式下载时解析 revision 并写入 assets.json，下载到 raw/corpus；后续默认复用同一 revision。可通过 `--corpus-revision <SHA>` 指定，已有目录不得直接切换 revision，换 ROOT_DIR。
- 官方 evidence/gold qrels、解密函数、裁判参考脚本和 MIT LICENSE 已随代码附带，不依赖下载后的网页是否仍可访问。

官方 corpus 数据卡列出的下载大小约为 1.76 GB，解压文本约 3.26 GB；query 数据因含嵌套文档也达到 GB 量级。实际大小与版本以下载结果为准。不要把题目自带的 gold_docs、evidence_docs 或 negative_docs 拼成全库；本流程只从独立 corpus 数据集建库。

```bash
ACTION=prepare bash scripts/sh/56_browsecomp_plus.sh
ACTION=index bash scripts/sh/56_browsecomp_plus.sh
```

默认 DECRYPTED=0，使用随代码附带的官方 canary、derive_key、decrypt_string 解密问题和答案；参考文档正文不进入问题 JSONL，只保留评分用的 qrels 文档 ID。源码实际文件名是 **qrel_golds.txt**（带 s），不是部分 README 示例中的 qrel_gold.txt。

### 43.5.2 用本地裁判完成一次测试

```bash
RUN_DIR=outputs/experiments/<your-run> \
JUDGE_MODEL=/models/Qwen3-32B LIMIT=2 \
  bash scripts/sh/56_browsecomp_plus.sh
```

all 在生成完成后释放待测模型的 Python 引用，回收并清理 CUDA cache，再按需加载裁判。它是顺序执行，不是同时部署两个 GPU 服务。PyTorch allocator 或其他进程占用仍需按实际服务器检查。

如果显存、GPU 分配或运行时间需要拆开：

```bash
# 第一阶段只生成，不需要 JUDGE_MODEL；OUTPUT_DIR 必须尚不存在。
ACTION=run RUN_DIR=outputs/experiments/<your-run> LIMIT=2 \
OUTPUT_DIR=outputs/results/bcp_generation_001 \
  bash scripts/sh/56_browsecomp_plus.sh

# 第二阶段使用原始预测，在另一进程/另一组 GPU 上评分。
ACTION=score LIMIT=2 CUDA_VISIBLE_DEVICES=0,1 \
PREDICTIONS=outputs/results/bcp_generation_001/predictions.jsonl \
JUDGE_MODEL=/models/Qwen3-32B \
OUTPUT_DIR=outputs/results/bcp_score_001 \
  bash scripts/sh/56_browsecomp_plus.sh
```

score 的 LIMIT 必须与生成阶段一致。已有 run manifest 的预测会校验问题集指纹和 selected_ids，防止拿“前 2 题”当作全量结果，或串用旧参考答案。导入没有 manifest 的外部预测会逐项校验 id/query/status，漏题仍计入分母；其生成来源需要使用者另行说明。

### 43.5.3 没有本地裁判时：只能做检索诊断

```bash
ACTION=score RETRIEVAL_ONLY=1 LIMIT=2 \
PREDICTIONS=outputs/results/bcp_generation_001/predictions.jsonl \
  bash scripts/sh/56_browsecomp_plus.sh
```

此时输出 evidence/gold recall 和工具统计，不输出答案 accuracy。它不是完整问答评分，不能把检索召回率当作答案正确率。

## 43.6 已经拥有数据时怎样修改路径

ROOT_DIR 是本流程拥有的缓存/索引目录，DATA 和 CORPUS 是输入资源，不要求源文件放在仓库中。源路径可包含空格。

```bash
# BrowseComp-Plus 已解密官方 JSONL + 独立 Parquet corpus：
ACTION=prepare DECRYPTED=1 ROOT_DIR=/scratch/bcp_v1 \
DATA=/datasets/browsecomp_plus_decrypted.jsonl \
CORPUS=/datasets/browsecomp-plus-corpus \
  bash scripts/sh/56_browsecomp_plus.sh
ACTION=index ROOT_DIR=/scratch/bcp_v1 bash scripts/sh/56_browsecomp_plus.sh

# Hotpot 官方 dev JSON + 解压后的独立 wiki abstracts bz2 目录：
ACTION=prepare ROOT_DIR=/scratch/hotpot_v1 \
DATA=/datasets/hotpot_dev_fullwiki_v1.json \
CORPUS=/datasets/enwiki-20171001-abstracts \
  bash scripts/sh/57_hotpotqa_offline.sh
ACTION=index ROOT_DIR=/scratch/hotpot_v1 bash scripts/sh/57_hotpotqa_offline.sh
```

自备路径会明确标注 custom/subset，保留来源指纹，不凭文件名宣称正式全量。默认官方下载布局中，BCP 的 qcount=830、doccount=100195、记录的快照来源以及无 corpus limit 才标为 official fixed corpus；Hotpot 默认摘要包路径需与 assets 中校验过的 SHA256 一致，且 qcount=7405。自备数据即使内容完整，也可保持这种保守标注；报告时附上自己的来源与指纹，不要为了去掉标签修改 manifest。

支持的输入：问题为 JSON array / JSONL / Parquet；语料支持相应文件、bz2/gz JSONL、目录和官方 tar.bz2。目录按路径排序、文件内按原始顺序处理。目录内仅放对应数据，不混放别的 JSON 配置；.cache 子目录会忽略。JSON array 文件会一次加载该文件，大型自备语料应优先使用 JSONL、Parquet 或压缩 JSONL；官方全量语料路径按流式处理。

原始字段映射：

| 来源 | 必需字段 | 输出 |
|---|---|---|
| BCP 问题 | query_id、query、answer；默认加密，可 DECRYPTED=1 | id、query、answer、官方 qrels 对应 evidence_docs/gold_docs ID |
| BCP corpus | docid、text，可选 url/title | id、title、text、source_url、空 sentences |
| Hotpot dev JSON | _id、question、answer、supporting_facts | id、query、answer、supporting_facts；丢弃 context/type/level |
| Hotpot HF fullwiki validation | id、question、answer、supporting_facts dict/list | 与上一行一致 |
| Hotpot Wiki abstracts | id、title、text sentence arrays，可选 url | id、title、text、sentences、source_url |

空问题、空答案、重复 question ID、空语料、重复 corpus ID、缺失 BCP qrels、被修改的官方参考文件会拒绝。prepared/、已发布 index.sqlite 和结果目录都拒绝原地覆盖；all 仅复用已经完成且配置相符的产物。prepare 或 index 中断会留下可审查的部分文件；先把部分目录移到备份路径，再重新执行，或使用新的 ROOT_DIR。

## 43.7 参数表与修改提示

参数既可修改 56/57 顶部，也可通过同名环境变量覆盖。58 是共享参数转换层，正常使用无需编辑。额外 Python 参数可放在 shell 命令末尾，例如 `--verify-index`。

### 43.7.1 通用环境和阶段

| shell 变量 | 默认 | Python 参数 | 修改提示 |
|---|---|---|---|
| ACTION | all | 首个位置参数 | download 是唯一允许联网阶段；prepare/index 不加载模型；run 不评分；score 不加载待测模型 |
| ROOT_DIR | outputs/data/benchmarks/<基准> | --root | 每个数据/索引版本一个目录；两套基准不要共用 |
| PYTHON / CONDA_ENV | 空 / search-comp-qwen3.5 | shell 解释器选择 | 使用目标训练环境，路径支持空格 |
| CUDA_VISIBLE_DEVICES | 0 | CUDA 环境 | 单进程 device_map=auto；不是每卡一个样本 worker，不使用 torchrun |
| RUN_DIR | 空 | --run-dir | 训练实验目录，读取最终 checkpoint，不把训练集 test 当作本基准 |
| MODEL_PATH / KIND | 空 / 自动 | --model / --kind | 模型路径可覆盖 RUN_DIR；KIND 为 plain/beacon |
| DATA / CORPUS | 空，按 root 自动定位 | --data / --corpus | 新数据应换 ROOT_DIR；prepare 不能把 dev context 当 corpus |
| DECRYPTED（仅 BCP） | 0 | --decrypted | 明文 BCP 问题必须设 1；Hotpot 不使用此参数 |
| OFFICIAL_ONLY | 0 | --official-only | 仅 ACTION=download 有效；只安装随代码附带的参考文件，不下载数据 |
| CORPUS_LIMIT | 空 | --corpus-limit | 测索引流程时取前 N 文档，强制标注子集；不要据此报告正式 benchmark |
| OUTPUT_DIR | 新时间戳目录 | --output-dir | 已存在就拒绝；RUN_DIR 非空时默认落到实验 evaluation 下 |
| PREDICTIONS | 空 | --predictions | score 必填；规范格式见 43.9 |
| LIMIT | -1 | --limit | -1 全量，正数前 N 题；与 corpus limit 是两个独立维度 |
| DRY_RUN | 0 | --dry-run | 只打印解析后的参数，不检查模型/数据可用性，不创建文件 |

准备、建索引后，如果重复执行 all，默认会复用已有 prepared/index。更改 DECRYPTED、CORPUS_LIMIT、显式 DATA/CORPUS 来源或 TITLE_WEIGHT 时会拒绝复用，应该换 ROOT_DIR。修改已有 raw 文件不会自动触发重建，也不应修改它；准备时会保存源文件指纹，正式复现实验应从原 manifest 指向的不可变数据出发。

### 43.7.2 索引、检索、阅读和生成预算

| 变量 | 默认 | Python 参数 | 含义和修改影响 |
|---|---|---|---|
| TITLE_WEIGHT | 1.0 | --title-weight | 标题列的 BM25 权重，正文固定 1；正有限数，更改后需重建 |
| VERIFY_INDEX | 0 | --verify-index | 1 时对整个索引做 SHA256 核验，数 GB 索引会增加启动 I/O |
| TOPK | 5 | --topk | 每个 query 返回文档数；影响召回与上下文，正式对比应固定 |
| SNIPPET_TOKENS | 512 | --snippet-tokens | 每篇搜索结果正文的 token 上限，使用待测模型 tokenizer；不是所有结果之和 |
| ALLOW_VISIT | BCP=0，Hotpot=1 | --allow-visit | 允许通过 local://docid 阅读；BCP 开启即为阅读扩展协议 |
| READ_TOKENS | 4096 | --read-tokens | 每篇 visit 正文上限；不根据 goal 调摘要模型，固定读取正文前缀 |
| TOOL_COUNT | 200 | --tool-count | 单题全部工具调用数上限；一批中多个 call 分别计数，一个 search 内多 query 算一次 call |
| MAX_TURNS | 200 | --max-turns | Agent 最大生成轮数，与工具次数上限分别约束 |
| MAX_CONTEXT | 131072 | --max-context | 完整序列化历史的逻辑上限；Beacon 也按未压缩历史执行此预算 |
| MAX_NEW_TOKENS | 8192 | --max-new-tokens | 每轮新增生成 token 上限，当前 native 使用贪心解码 |
| MAX_TOOL_TOKENS | 32768 | --max-tool-tokens | 单条完整 tool_response 的 token 上限，包含 JSON/文档元信息 |

SQLite FTS5 的 BM25 使用固定 k1=1.2、b=0.75，分词为 `unicode61 remove_diacritics 2`；不能用 TITLE_WEIGHT 改变 k1/b。查询词按 OR 检索，每个 query 最多 128 个去重词，字面化处理 FTS 运算符；同分按文档插入 rowid 稳定排序。不是 Elasticsearch，也没有后台 daemon；index.sqlite 同时存储原文与倒排索引，不用 pickle 加载整个 corpus。

一个 search 最多接受 16 个 query；一次 visit 最多 16 篇文档。搜索结果包含 docid、title、local URL、source URL 和正文片段；visit 只允许 local://，不会沿 source_url 访问网站。Hotpot 正文展示 `[0] sentence` 等零基编号。截断发生在明确的正文预算处，不会补参考答案，也不反向挑选 gold 证据；若拼接后的整条工具回复仍超过 MAX_TOOL_TOKENS，现有 Agent 会标为 tool_budget，不会继续偷偷裁剪。

### 43.7.3 BrowseComp-Plus 本地裁判

| 变量 | 默认 | 含义 |
|---|---|---|
| JUDGE_MODEL | 空，完整评分必填 | 本地 Qwen3-32B 或明确记录的替代模型目录；HF tokenizer 和权重只能从本地读取 |
| RETRIEVAL_ONLY | 0 | 1 时仅检索诊断，不加载裁判、不输出答案 accuracy |
| JUDGE_MAX_TOKENS | 4096 | 裁判最大输出；预算耗尽会记录错误，不把未完成判决当正常标签 |
| JUDGE_MAX_CONTEXT | 32768 | 裁判输入 + 预留输出的上限；超限报错，不静默截断答案或参考 |
| JUDGE_TEMPERATURE | 0.7 | 官方参考脚本默认；设 0 切换贪心，属于不同评分配置 |
| JUDGE_TOP_P | 0.8 | 官方默认 nucleus sampling |
| JUDGE_TOP_K | 20 | 官方默认 sampling top-k；不是文档检索 TOPK |
| SEED | 42 | 本地裁判随机种子；记录配置便于复现 |

对应 Python 参数为 `--judge-model`、`--retrieval-only`、`--judge-max-tokens`、`--judge-max-context`、`--judge-temperature`、`--judge-top-p`、`--judge-top-k`、`--seed`。官方使用 vLLM，本实现使用 Transformers，本地模型的默认 chat template / thinking 行为被保留；prompt、parser 与默认 temperature/top_p/top_k/max_output_tokens 对齐，但不同引擎或随机数实现不保证逐 token 一致。替换成更小裁判可用于资源受限测试，必须在报告中注明，不能视为官方 Qwen3-32B 裁判成绩。

Hotpot 不需要这些参数；裁判大模型只影响 BCP。完整 Hotpot EM/F1 计算不使用任何 API 或辅助生成模型。

## 43.8 答案输出与评分规则

### 43.8.1 HotpotQA

本地 system prompt 要求最终输出：

```text
<answer>{"answer":"a short exact answer","sp":[["Document title",0]]}</answer>
```

已有 Agent 会提取 answer 标签正文。评分适配器解析 JSON，独立导出 `answer` 与 `sp` 给官方 evaluator；若模型给出普通文本，则整个文本作为答案进行官方归一化，不用 gold 反向提取“最像答案”的词。这样普通长解释可能使 EM=0、F1 较低，属于真实模型输出。

- EM/F1 的大小写、英文标点、冠词、空白归一化与 yes/no/noanswer 特例全部执行官方实现。
- 默认 EM/F1 是 0—1 比例；乘 100 才是百分数。
- SP/Joint 只有模型自己输出有效支持事实才有意义。不把检索到的文档、阅读过的所有句子或 gold supporting facts 自动当作预测。
- 不输出 sp 时，官方 SP/Joint 按缺失处理，仍保留全部题目为分母；不能用这些 0 分断言模型具备或不具备已专门训练过的证据标注能力。
- sp 格式不合法会记录 supporting_parse_error 并不导出 sp，**不影响合法短答案的独立 EM/F1**。
- 漏题、tool_error、context_limit、生成预算耗尽等未完成作答不给正确分，即使部分文本出现标准答案。

Hotpot 官方代码唯一兼容处理是执行 AST 时将 `import ujson as json` 替换为标准库 json，源文件和所有评分函数保持原样。输出的 hotpot_predictions.json、hotpot_gold.json 可再交给安装了 ujson 的原脚本单独复核。

### 43.8.2 BrowseComp-Plus

本地裁判收到问题、完整生成回答、标准答案；这些参考内容只在评分阶段出现，不进入待测模型推理或本地工具。使用固定官方 GRADER_TEMPLATE、create_judge_prompt、parse_judge_response，按语义等价得到 yes/no，并保留原始判决。

主要指标：

- accuracy / Accuracy (%)：所有选定问题为分母；生成缺失/失败按 0；不是只对成功生成的题目计算。
- evidence_recall / Recall (%)：每道问题 **所有 search 调用返回的 docid 并集** 对官方 evidence qrels 的召回率，再按题平均。
- gold_recall：相同方式对 gold qrels 计算；gold 表示包含最终答案信息的标注文档，不等于 evidence 的全部集合。
- avg_tool_stats：工具调用数按所有选定问题平均，不忽略失败题。
- citation_coverage、citation_precision、citation_recall：沿用官方 citation 提取与计算函数，precision/recall 在有引用的回答上平均；它们与检索召回不同。
- Calibration Error (%)：沿用官方函数。少于 100 个有效 confidence 时，官方返回 0.0 占位，本项目同时写 `calibration_available=false` 和样本数，不能把这个占位值当作已获得完美校准。

语义裁判出现未知标签、异常或上下文/生成超限，会写 judge.jsonl、保留分母、将 complete=false，并以非零退出码结束。修复模型配置后，可用同一 predictions 文件和新的 OUTPUT_DIR 重评分，不需要重新生成。

本次没有实现额外的静态 TREC run / nDCG@10 导出；这里的 retrieval recall 是官方 Agent 评测的累积检索口径，不是单次检索排序的 nDCG，也不以 recall 冒充答案 accuracy。

## 43.9 产物、复用与导入预测

| 文件 | 内容 |
|---|---|
| manifest.json | 本次模式、模型、参数、selected_ids、问题/语料/索引/本地代码指纹、状态 |
| predictions.jsonl | 每题当前模型的最终回答、完整 messages/events、状态、token 数、工具次数、retrieval_trace、检索/阅读 docid |
| generation.metrics.json | 生成状态分布、数量、耗时 |
| summary.metrics.json | 基准主要指标、漏题数、是否为子集、是否完整、离线标记 |
| per_question.json | 逐题评分或裁判/召回详情 |
| hotpot_predictions.json / hotpot_gold.json | 可交官方 Hotpot 脚本复核的格式 |
| official_hotpot.log | 原始官方 eval 的输出及缺失项提示 |
| judge.jsonl | BCP 逐题判决、异常和召回；逐题 flush |
| official_runs/*.json | BCP 官方 evaluate_run 可接受的 query_id/status/retrieved_docids/result 格式 |

导入的预测 JSONL 最少应包含：

```json
{"id":"question-id","status":"completed","final_response":"model answer","retrieved_docids":["docid"],"tool_call_counts":{"search":1}}
```

id 必须属于当前选定问题，且只出现一次；final_response 也接受 prediction 别名。query 若提供必须与当前基准一致。status 必须显式提供；BCP 的 retrieved_docids 应是所有 search 返回文档的并集，不能塞入参考 gold 文档 ID。没有此字段时召回自然为 0。外部预测的来源、工具真实性仍由生成者负责；评分器不能凭 JSON 证明外部运行确实检索过这些文档。

只评 Hotpot 示例：

```bash
ACTION=score LIMIT=2 ROOT_DIR=/scratch/hotpot_v1 \
PREDICTIONS=/scratch/results/hotpot_run/predictions.jsonl \
  bash scripts/sh/57_hotpotqa_offline.sh
```

score 不需要再次加载索引，也不加载待测模型，读取 prepared 问题和官方评分参考即可。结果目录禁止覆盖，避免旧判决混入新配置。all 复用 prepared/index，但总是创建全新结果目录。

## 43.10 HPC 操作、问题排查和修改建议

1. 在登录节点或允许联网的准备机下载原始数据、依赖、完整本地模型；prepare/index 可以放到内存和磁盘配额足够的计算节点。
2. 把同一基准的 ROOT_DIR 整体拷贝到目标目录，保持 prepared/index/manifest 配套；不要只拷一个 SQLite 文件后丢失问题集与来源。
3. 在任务脚本内激活环境，设置 PYTHON、CUDA_VISIBLE_DEVICES、RUN_DIR/MODEL_PATH 和所需 JUDGE_MODEL，然后执行 56 或 57。无需打开任何 HTTP 端口。
4. 先 LIMIT=2 检查模型输出和工具调用，再做 LIMIT=-1；小模型不会因接入测试集就自动学会新 prompt 的结构化 sp 输出。
5. Plain 和 Beacon 使用相同 TOPK、SNIPPET_TOKENS、ALLOW_VISIT、READ_TOKENS、MAX_CONTEXT、工具轮数与裁判，另外记录 Beacon 压缩比和模型训练配置。

| 错误或现象 | 原因与修改提示 |
|---|---|
| local source not found | all 不联网；先 ACTION=download，或设置 DATA/CORPUS 并完成 prepare/index |
| no such module: fts5 | 当前 Python 的 SQLite 未启用 FTS5；换支持的 Python/SQLite 环境 |
| prepared directory exists | prepare 不能覆盖；后续用 index/run/all 复用，或换 ROOT_DIR |
| index.building.sqlite already exists | 上次构建中断；先备份部分文件再重建，不能直接当作完整索引 |
| TITLE_WEIGHT differs | 当前参数和已构建索引不一致；恢复原参数，或新 ROOT 重建 |
| checksum/source changed | 数据、索引或官方代码与记录不一致；恢复原文件，不要手改 manifest 绕过 |
| only adapters / missing model | 先完成训练和 LoRA 合并，指向包含完整 config/tokenizer/权重的目录 |
| tool_error / invalid_call | 检查逐题 messages/events、调用包裹格式、local:// URL、是否启用了 visit |
| tool_budget | 多 query/topk 的总回复太大；检查 SNIPPET_TOKENS、TOPK、MAX_TOOL_TOKENS，并保持对比实验一致 |
| context_limit | 完整历史超过逻辑预算；增加模型可支持的 MAX_CONTEXT，或减少轮次/阅读预算 |
| judge context budget exceeded | 减少最终回答长度或增加裁判支持的 JUDGE_MAX_CONTEXT；不通过裁剪 gold 绕过 |
| GPU OOM | 控制可见卡、待测模型上下文和裁判上下文；可将 run 与 score 分开进程运行 |
| score selection differs | 独立 score 的 LIMIT 与生成不同；使用同样的 ROOT_DIR/LIMIT |
| offline forbids network sockets | 缺本地资源或库试图访问网络；先准备完整资源，不要通过打开在线 API 绕过 |

程序化调用也可使用：

```bash
python scripts/py/52_offline_benchmark.py index --benchmark hotpotqa --root /scratch/hotpot_v1
bash scripts/sh/02_workflow.sh offline-benchmark score --benchmark hotpotqa \
  --root /scratch/hotpot_v1 --predictions /scratch/predictions.jsonl --limit 2
```

本次验证范围及日志见 [99 验证记录](99_validation.md)。这里没有宣称已获得真实 checkpoint 的完整基准成绩；部署后的真实生成质量、GPU 显存和全量索引时间需要在目标机器测量。
