# 10 Polar 数据处理与细粒度筛选

原始数据位于 `outputs/data/raw/polar/`，包括原始快照目录和 `openseeker_v1_data.jsonl`。只从 `../sa-qwen-openseeker/outputs/data/raw/polar/` 复制这一份数据；不依赖源项目的派生数据。重新下载入口保留为 16，默认只下载 Polar，当前迁移无需运行下载。

首次准备运行 `bash scripts/sh/10_prepare.sh`。已生成同名输出时拒绝覆盖，修改 `OUTPUT_DIR` 可创建另一版本。主要变量：

| 变量 | 含义 |
|---|---|
| INPUTS / SOURCE | 输入 JSONL/Parquet 路径与来源；当前为 Polar |
| OUTPUT_DIR | 默认 `outputs/data/openseeker/polar_correct_qwen35_16k` |
| MODELS | 用于实测的 tokenizer 标识，可列多个 |
| MIN_LENGTH / MAX_LENGTH | 完整训练序列 token 长度闭区间，默认 1—16384 |
| CORRECTNESS | `Correct` / `Incorrect` / `any`，默认严格 Correct，缺失标签也被排除 |
| OBSERVATIONS | `any` / `with` / `without`；默认 any |
| WORKERS / OFFLINE | CPU 分词并发与仅本地缓存 |
| VAL_FRACTION / TEST_FRACTION / SEED | 默认约 96%/2%/2%，按问题哈希分组 |

规范化保留 Polar system、问题、assistant 推理/调用/回答正文。结构化 OpenAI calls 转换成一致的工具协议，tool role 转为 user observation。只对 assistant 内容计算主损失，仅 observation 正文允许压缩；不截断样本。`answer` 仅是独立评分参考，不注入轨迹或推理提示词。当前协议标识为 `openseeker-chatml-v3-original-answer`；旧派生数据必须重新生成。

输出包含：`normalized.jsonl`（所有通过结构校验的样本及全部长度，包括超长和 Incorrect）、`train/validation/test.jsonl`（筛选后划分）、`rejected.jsonl`（拒绝原因）、`measurements.jsonl`（逐条长度与工具统计）、`manifest.json`（来源 SHA256、tokenizer 指纹、筛选参数和分布）。输入官方 validation/test 按问题优先保留划分；普通数据按问题哈希划分并去重，避免相同问题跨集合。拒绝原因按长度、正确性、观测、重复问题顺序记第一个原因。

后续筛选运行 `11_select.sh`，读取父目录的 `normalized.jsonl`，无需再次分词。即使父目录默认训练集只有 16k 以下，也可以从完整 normalized 文件筛选 64k、128k 等长样本。所有筛选结果写入新目录，不修改父数据。可组合以下条件：

- `MIN_LENGTH=32768`、`MAX_LENGTH=33792`：取闭区间样本。
- `EXACT_LENGTH=32768`：仅保留恰好 32768 token；MIN/MAX 仍须包含该值。
- `TOKENIZER`：明确采用哪个测量模型；不指定则所有已测模型都必须满足区间。
- `CORRECTNESS="Incorrect"`：单独导出错误轨迹；`any` 包含未知标签。
- `OBSERVATIONS="with"` 或 `"without"`：按非空 observation 正文筛选，不把 assistant 里的引用当工具返回。
- `SPLIT_MODE="grouped"`：沿用父预处理的问题哈希和官方划分规则；`"train"`：全部写入训练集，供显存探测。

`13_filter_observations.sh` 是保留的独立 observation 过滤器：对已有 train/validation/test 逐字节保留含 observation 的记录，保留划分，并将剔除记录放到 `removed/` 和审计 JSONL。空正文不计观测，纯空白正文按可能产生 token 的有效正文处理。选择无观测数据请用 11 的 `without`。

历史证据库通过 `15_build_corpus.sh` 构建，只读取 visit 调用配对的网页证据，去重并保留溯源，不采集参考答案或 assistant 思考；这种证据库属于历史轨迹语料，不等同于独立在线搜索。

## 本次实际生成的数据

已完成 11,677 条原始轨迹的全量规范化与 Qwen3.5 tokenizer 实测：Correct 4,949 条、Incorrect 6,728 条，结构无效记录为 0。以下均为新代码生成，原始数据保持不变：

| 目录（均在 outputs/data/openseeker/ 下） | train | validation | test |
|---|---:|---:|---:|
| polar_correct_qwen35_16k | 644 | 9 | 11 |
| polar_correct_qwen35_16k_with_observations | 594 | 8 | 8 |
| polar_correct_qwen35_32k_33k_with_obs | 58 | 3 | 0 |
| polar_memory_probe（64512—65536，Correct，有观测，仅训练） | 43 | 0 | 0 |

32k 区间示例测试集为 0，源于小集合的固定哈希划分；用于正式评测时应调整区间或划分比例，不能从空测试集报告能力分数。显存测试集合不用于能力评分。原始全量的训练序列长度范围为 1,193—369,679，p50=68,219；默认 16k 训练集实际最大长度为 16,338。详见各自 manifest 与 `outputs/reports/verification/datasets.json`。

`outputs/reports/dataset.html` 已从默认训练集实际导出 644 条记录，可离线打开。上述输出目录已存在，再执行同名生成脚本会按设计拒绝覆盖；重新实验时修改 OUTPUT 变量。

另已实际生成 `polar_memory_exact_65117`：1 条恰好 65,117 token 的 Correct、有 observation 训练样本，并验证长度精确相等；仅用于显存测试。

## 独立离线 Benchmark 数据

新增 BrowseComp-Plus 与 HotpotQA 的数据适配独立于 Polar 训练准备。56 使用官方 BCP 加密问题、独立 corpus 和 qrels；57 使用有标注 dev 与官方 2017 Wikipedia 摘要库，丢弃 dev 候选 context。结果、明文问题、规范化 corpus 与持久化索引都放在 outputs，不混入训练 split。首次下载与离线导入、字段映射和版本校验见 [43](43_offline_benchmarks.md)，来源和许可证见 [44](44_offline_sources.md)。
