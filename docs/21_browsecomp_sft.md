# BrowseComp SFT 数据接入

下载和预处理在 Linux 上运行，使用仓库的训练环境和目标模型的本地 tokenizer。预处理使用 CPU。来源固定为 [BrowseComp SFT 数据集](https://huggingface.co/datasets/AmanPriyanshu/tool-reasoning-sft-RESEARCH-grill-lab-browsecomp-plus-runs-data-cleaned-rectified/tree/1a67da5e15590e15fc962035a75cd427a47070b2)。

| 原始分片 | 内容 | 训练使用方式 |
|---|---|---|
| `syn_planner.parquet` | 19,505 条网页搜索与抓取轨迹，包含 HotpotQA 等六个来源。 | 默认选用。保留实际评测题文件用于去重。 |
| `data.parquet` | 30,355 条 BrowseComp 运行记录，`user` 消息可能含题目的长改写或原题。已确认与 BrowseComp、BrowseComp-Plus 评测同题。 | 显式选择，并提供原题文件用于归并。只要实验仍用这两套题评测，就把它们保留在受保护列表中。若实验专门研究这些轨迹，可使用独立评测集，按实验范围调整保护列表。 |

## 下载与准备评测题

1. 执行 `bash scripts/data/sh/12_download_browsecomp_sft.sh`。下载产物位于 `artifacts/datasets/research/raw/browsecomp-sft/<commit>/`。保存终端输出路径。两个 Parquet 合计约 2.13 GB，主分片只有一个大 row group。处理主分片时建议先降低 `SA_WORKERS` 并检查 Linux 主机可用内存。
2. 准备所有计划评测的题目，参照 [模型、训练数据与 benchmark 准备](20_data.md)。默认保护列表包含 BrowseComp、BrowseComp-Plus、HotpotQA。BrowseComp 题目由 30 的 `download` 生成，另两套题目由 31/32 的 `prepare` 生成。已有题目 JSONL 时直接用其真实路径。这个步骤只需题目文件，重合检查无需建立索引。
3. 修改 `scripts/data/sh/25_prepare_browsecomp_sft.sh`，运行 `bash scripts/data/sh/25_prepare_browsecomp_sft.sh`。

### 下载参数

| SH 变量 | 默认值或修改提示 |
|---|---|
| `PYTHON_BIN` | 当前 Linux 环境的 `python`。也可在启动脚本前指定环境的 `bin/python`。 |
| `HF_ENDPOINT` | 默认 `https://huggingface.co`。网络需走镜像时改为可访问的 Hub 地址。 |
| `HF_HUB_OFFLINE` | 下载使用 `0`。 |
| `CUDA_VISIBLE_DEVICES` | 下载和计长无需 GPU。 |
| `SA_SOURCE` | 固定 `browsecomp-sft`。 |
| `SA_REVISION` | 默认固定到已核验的 commit。切换 revision 后检查文件名、列和工具协议。 |

### 预处理参数

| SH 变量 | 作用和修改提示 |
|---|---|
| `PYTHON_BIN`、`HF_ENDPOINT`、`HF_HUB_OFFLINE`、`CUDA_VISIBLE_DEVICES` | 同下载脚本。预处理默认 `HF_HUB_OFFLINE=1`，使用本地模型文件。 |
| `SA_SOURCE` | 固定 `browsecomp-sft`。输出归入 Research 训练入口。 |
| `SA_SNAPSHOT_DIR` | 下载输出的 commit 目录。用快照时将 `SA_INPUTS` 留为 `[]`。 |
| `SA_INPUTS` | 已有本地文件时填写 JSON 路径数组，如 `["/data/syn_planner.parquet"]`，并清空 `SA_SNAPSHOT_DIR`。按数组顺序处理。 |
| `SA_SHARDS` | 仅在使用快照时生效。默认 `["syn_planner.parquet"]`。选主分片填 `["data.parquet"]`，同时处理时按快照清单顺序读取。 |
| `SA_CANONICAL_QUESTIONS` | 主分片必填，建议用 BrowseComp 的 `questions.jsonl`。原题从最佳匹配读取，用于归一化训练问题和生成分组哈希。辅助分片可留空。 |
| `SA_PROTECTED_QUESTIONS` | 评测题 JSONL 的路径数组，每行提供 `query` 或 `question`、`answer` 和可选 `id`。默认三套评测，路径随版本调整。缺少文件会报错。只有明确允许这些题进入训练时才移除对应项。 |
| `SA_OVERLAP_THRESHOLD` | 默认 `0.30`，范围 `(0, 1]`。影响改写匹配，调低可提高检出率，也会增加误报。精确和嵌入匹配优先。 |
| `SA_PROFILE` | 输出版本名。修改输入、计长、保护列表、匹配或划分策略后换新名称。 |
| `SA_MODELS` | 本地 tokenizer 路径的 JSON 数组。至少一个，与训练 `SA_MODEL_PATH` 的路径字符串完全一致。 |
| `SA_MIN_LENGTH`、`SA_MAX_LENGTH` | 完整轨迹 token 范围，默认 1 至 16,384。按当前模板实际编码，超长整条剔除。训练上限至少为预处理上限。 |
| `SA_CORRECTNESS` | 使用 `any`。当前源没有独立正确性标签，`Correct` 和 `Incorrect` 会筛空这些数据。 |
| `SA_OBSERVATIONS` | 默认 `with`，保留带观测的轨迹。可取 `any`、`with`、`without`。 |
| `SA_WORKERS` | CPU 计长进程数，默认 4。内存紧张时降低。 |
| `SA_SEED` | 默认 42。控制按问题哈希分组后的确定性划分。 |
| `SA_VAL_FRACTION`、`SA_TEST_FRACTION` | 默认各 0.02，允许 0，二者和必须小于 1。实际样本数由哈希和筛选结果决定。 |
| `SA_OFFLINE` | 默认 `true`，只从本地加载 tokenizer。 |

主分片要求原题匹配唯一，最佳与次佳覆盖率差小于 0.10 的轨迹记入 `unresolved_benchmark_question`。辅助分片使用原问题，并保留上游来源及 `qid` 供追溯。两种分片都按问题文本生成 `question_id`，同一题仅保留第一条符合筛选条件的轨迹。保护列表在计长前检查，命中时写入 `rejected.jsonl`。原题映射后也检查保护列表。归并后的同题不会分配到多个本地划分。

工具调用统一为训练协议。辅助分片的 `web_search(query)` 映射为 `search(query=[...])`，`crawl_webpage(url)` 映射为 `visit(url, goal)`，其中 `goal` 用当前问题补齐。主分片的 `search(parameters.user_query)` 统一为 `search(arguments.query)`，使用只读检索轨迹的 search-only prompt。未支持的工具调用进入拒绝清单。工具观测保留历史文本，运行时检索质量需要单独评估。最终回复保留为示范回答，内部保留划分适合训练诊断。

## 训练与混合

- 单独训练时使用 Plain 或 Beacon 的 `11_train_research.sh`，将 `SA_DATASET_NAME` 设为 `research`，`SA_DATASET_DIR` 指向 `artifacts/datasets/research/prepared/<PROFILE>/`。
- 混合时使用 `24_mix_datasets.sh`。与 OpenSeeker 两路混合时将 SFT 目录填入 `SA_RESEARCH_PREPARED`。同时加入普通 Research 时，把 SFT 填入 `SA_BROWSECOMP_SFT_PREPARED`，按 OpenSeeker:Research:BrowseComp SFT 顺序设置 `SA_RATIO`，例如 `2:1:1`。混合仍优先保留高优先级的同题划分，按 `test > validation > train` 排除低优先级副本。
- 使用混合训练产物时，将训练脚本的 `SA_DATASET_NAME` 设为 `mixed`。测试脚本中的 `SA_PROMPT_DATA` 选一个与评测工具能力匹配、仅含一种 system/tools 组合的规范化 JSONL。单独训练但同时纳入两个 SFT 分片时，同样可用此变量显式选择一种提示词。
- 预处理结束先检查 `manifest.json` 的输入文件指纹、题目参考文件指纹、计长分布与划分数量。查看 `rejected.jsonl` 的 `protected_question`、`unresolved_benchmark_question` 和工具格式错误，确认训练集非空后再启动训练。

## BrowseComp 与 BrowseComp-Plus 测试前筛选

Plain 和 Beacon 的 `40_test_browsecomp.sh`、`42_test_browsecomp_plus.sh` 默认启用筛选。测试从训练实验读取实际 `train.jsonl`，验证其训练时记录的 SHA256，在生成前排除重合题。训练版本和官方题文件需要保留在实验记录所指路径。

| SH 变量 | 作用和修改提示 |
|---|---|
| `SA_OVERLAP` | `filter` 生成 `overlap.json` 并排除匹配题。需要复跑完整历史题集时显式设为 `off`，并记录实验口径。 |
| `SA_OVERLAP_THRESHOLD` | 改写匹配阈值，默认 0.30。较低值更容易检出改写。 |
| `SA_GENERATION__LIMIT` | `-1` 使用筛选后全部题目。正数取筛选后按官方顺序的前 N 条。 |

匹配先比较大小写和符号归一后的词序列，再检查长问题是否被完整嵌入。改写匹配要求至少 5 个不同的四词片段重合，并达到参考题的覆盖率阈值。大幅改写或翻译可能漏检，共享长片段的不同题可能误匹配。报告记录排除 ID、匹配方式、覆盖率、示例训练行 ID 和输入指纹。评分复用该报告及题数限制。全被排除时测试停止并保留报告。由此产生的分数应与相同题目子集比较。
