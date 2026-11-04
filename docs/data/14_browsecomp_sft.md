# 14 BrowseComp Qwen3.5 SFT 数据、长度与训练入口

本次接入的是项目目录内已有的 `outputs/data/browsecomp-qwen35-sft/train.jsonl`。Windows 绝对路径为 `E:\HZJ\研究生\学习资料\SearchAgent\代码\sa-openseeker-v2\outputs\data\browsecomp-qwen35-sft\train.jsonl`。Linux 上保留项目内的相对路径即可，不把 Windows 盘符写入配置。本次未安装依赖、未加载模型权重、未启动训练。

## 14.1 数据实际结构与适配范围

源文件包含 372 条 JSONL 轨迹；顶层为 `messages` 和 `tools`，没有独立的 `answer` 或正确性标签。共有 system 372 条、初始 user 372 条、assistant 13,393 条、tool observation 13,021 条。源文件只读，输出进入独立 prepared 目录。

新增 `search_comp/data/browsecomp.py`，由原有 `prepare_openseeker.py --source browsecomp` 显式调用。其他数据源仍执行原来的规范化规则。训练仍为 `data_mode: openseeker`，复用当前 Dataset、collator、Plain/Beacon 训练器及实验归档，不增加第三套训练协议。

适配步骤如下：

1. 深拷贝源记录，保留 system、问题、工具 schema 和全部 assistant 输出，包括思考、调用和最终回答。
2. 将结构化 `tool_calls` 转为现有 `<tool_calls_begin>` / `<tool_call>` / `</tool_calls_end>` 协议，校验参数 JSON、调用 ID 和 observation 对应顺序。
3. 将 tool 消息变为带 `<tool_response>...</tool_response>` 的 user observation。
4. 对原始工具正文中的字面 `<tool_response>` 和 `</tool_response>` 转义为 HTML 实体，防止网页内容提前关闭协议边界。实际文件中发现 1 条轨迹含此情况，共转义 4 个标签；每条记录以 `escaped_observation_delimiters` 记录数量。其他正文不截断、不摘要、不按文档切碎。
5. 源文件无参考答案时，取最后一条 assistant 消息中最后一个 `</think>` 后的最终回答作为训练元数据 `answer`，标记 `answer_source=demonstration_final_response` 和 `training_only=true`。如果已有非空参考答案则保留，标记 `reference`。不会向 assistant 内容追加该字段，也不会将示范回答称为金标准。
6. 无最终回答、未闭合思考、孤立 tool ID、调用后无 observation 等记录写入拒绝记录，不静默修复为成功样本。

这些数据没有独立正确性标签，因此此来源强制关闭 `require_correct` 筛选，接受 `--correctness any`。不能用 `Correct` 假装其全部正确。默认只保留至少有一个非空 observation 正文的轨迹。

## 14.2 长度、监督与压缩口径

长度复用训练实际调用的 `openseeker.encode()`：system、工具 schema、角色边界、问题、assistant 思考与回答、工具调用及 observation 全部计入。每个片段使用目标 tokenizer 的 `encode(add_special_tokens=False)`；不拿字符数、源文件长度、压缩后的长度或模型声明的 context 上限代替训练 token 数。

`MAX_LENGTH=16384` 是默认完整轨迹上限。等于上限的样本保留，超过上限写入 `rejected.jsonl`，原因 `over_length`；不从尾部切掉最终答案，不剪掉中间工具交互。训练 Dataset 会再次实测并拒绝超过 `max_length` 的轨迹。Beacon 的压缩比例不改变入库长度判断，确保 Plain 和 Beacon 读取同一份过滤结果。

监督和压缩沿用现有规则：

| 区域 | assistant CE loss | Beacon 可压缩 |
|---|---|---|
| system、schema、初始问题、角色前缀 | 否 | 否 |
| assistant 思考、工具调用、最终回答 | 是，沿用分段 loss 权重 | 否 |
| observation 正文 | 否，label=-100 | 是 |
| observation 开闭标签 | 否 | 否 |

Plain 使用完整序列；Beacon 只对 `regions` 指定的 observation 正文压缩。`beacon_doc_region_split` 应保持 false。Beacon 的 CE、输出蒸馏、续写损失等由 `39_beacon_browsecomp_train.sh` 的内联 YAML 配置，未因增加数据源而改变默认训练算法。

真实 Qwen3.5 tokenizer 尚未在当前环境执行，因此不能预报 16K 过滤后保留多少条，也不能把字节级测试统计当作 Qwen token 分布。运行预处理后以 manifest 的 `counts`、`distributions` 和 `measurements.jsonl` 为准。

## 14.3 Linux 上运行

使用已有训练环境；不需要新增环境。默认 `OFFLINE=1`，要求本地 tokenizer 可读取。

```bash
# 只做数据处理：不加载模型权重，不训练。
MODEL_PATH=/data/models/Qwen3.5-4B \
  bash scripts/sh/data/14_prepare_browsecomp.sh

# 只归档 Plain 实验配置、数据/代码指纹和启动命令，不训练。
MODEL_PATH=/data/models/Qwen3.5-4B ACTION=plan \
  bash scripts/sh/plain/25_plain_browsecomp_train.sh

# 对同一份 prepared 数据生成 Beacon 计划，不训练。
MODEL_PATH=/data/models/Qwen3.5-4B ACTION=plan \
  bash scripts/sh/beacon/39_beacon_browsecomp_train.sh
```

25/39 在 prepared 目录缺少 manifest 时直接执行预处理 Python 命令；存在时复用，不自动覆盖。两个入口均包含完整训练配置和启动命令，不转调其他训练脚本。首次 plan 可能进行 CPU 分词，但不会启动训练。单独执行 14 时，输出目录必须不存在。

准备正式训练时，将上述 `ACTION=plan` 改成 `ACTION=train`。多卡示例为 `CUDA_VISIBLE_DEVICES=0,1,2,3 DISTRIBUTED_BACKEND=zero2 ACTION=train ...`；进程数默认按可见卡数推断，可用 `NPROC_PER_NODE` 覆盖。本次没有执行这些训练命令。

## 14.4 全部新增入口变量

环境变量应使用 `VAR=value bash ...` 或提前 `export`；仅在 shell 中赋值但不 export 的变量不会传给子脚本。

| 变量 | 默认 | 作用与修改提示 |
|---|---|---|
| INPUT | outputs/data/browsecomp-qwen35-sft/train.jsonl | 原始输入文件，仅 14 或首次自动预处理读取 |
| DATASET_DIR | outputs/data/browsecomp-qwen35-sft/prepared_16k | 独立输出目录；25/39 将其写入实验 dataset |
| MODEL_PATH | Qwen/Qwen3.5-4B | 同时用于长度测量与训练模型；推荐本地绝对路径；两阶段字符串必须相同 |
| MAX_LENGTH | 16384 | 正整数，完整序列 token 上限；同步覆盖训练 max_length |
| WORKERS | 4 | CPU tokenizer 进程数，正整数；内存紧张可设 1 |
| SEED | 42 | 数据流程种子；不自动覆盖原训练入口的训练 seed |
| OBSERVATIONS | with | with 至少一个非空正文；any 不筛；without 无正文对照 |
| OFFLINE | 1 | 1 仅本地 tokenizer；0 允许现有库下载 tokenizer，不安装软件 |
| ACTION | plan | 25/39 接受 plan 或 train；14 本身总是处理数据 |
| PYTHON | 空 | 现有 Python 可执行文件；具体选择规则见 common/00_common.sh |
| CONDA_ENV | search-comp-qwen35-cu126 | Python 选择所用已有环境名，不创建环境 |
| CUDA_VISIBLE_DEVICES | 原训练脚本默认 0 | 训练可见卡列表；预处理不用 GPU |
| NPROC_PER_NODE | 可见卡数 | 训练进程数；单进程直接启动，多进程走原 launcher |
| DISTRIBUTED_BACKEND | zero2 | 原训练分布式参数；ddp/zero2，Plain 还支持 zero3 |

修改长度应同时换目录，例如：

```bash
export MODEL_PATH=/data/models/Qwen3.5-4B
export MAX_LENGTH=32768
export DATASET_DIR=outputs/data/browsecomp-qwen35-sft/prepared_32k
ACTION=plan bash scripts/sh/plain/25_plain_browsecomp_train.sh
```

若复用旧目录而更换 tokenizer，实验计划会因 manifest tokenizer 不匹配报错；更小训练上限也会被拒绝。改变 observation 筛选、输入文件或 tokenizer 后应使用新目录重新处理，不手动修改 manifest 绕过验证。

学习率、LoRA、累积步数、保存频率、分布式配置及 Beacon 压缩参数，直接在对应的 `25_plain_browsecomp_train.sh` / `39_beacon_browsecomp_train.sh` 内联 YAML 中修改。两个脚本各自包含数据预处理、环境变量覆盖、参数校验和 `51_experiment.py` 启动命令；仅加载 `common/00_common.sh` 提供项目路径和 Python 选择等基础函数。20/30 继续用于 OpenSeeker 训练。25/39 不解析额外训练 CLI 参数；应使用这里的变量或修改内联 YAML。

## 14.5 产物、追溯与评测边界

| 文件 | 内容 |
|---|---|
| normalized.jsonl | 成功规范化、尚未筛选的记录 |
| train.jsonl | 去重且满足长度/observation 条件的训练记录 |
| validation.jsonl / test.jsonl | 本来源保留为空文件；不是独立金标准评测集 |
| rejected.jsonl | 结构错误、重复问题、长度/observation 筛选原因 |
| measurements.jsonl | 每条有效轨迹的总长、observation/assistant token 数和拒绝原因 |
| manifest.json | 输入 SHA256、tokenizer 指纹、参数、计数、长度分布和 training_only 标记 |

此来源强制 `--val-fraction 0 --test-fraction 0`，避免把示范答案当作独立参考答案。现有 experiment 入口会将空 validation 转成 null，训练不计算验证 loss；空 test 只用于归档，不应通过旧 replay eval 汇报数据集成绩。实际测试使用独立 benchmark 入口，见 [59 三种方法与 LLM-Judge](../benchmark/59_llm_judge_matrix.md)。

工具能力需要区分：本文件原始 schema 有 web_search/open_url/find_text、paper_search、pubmed_search，轨迹中还出现 browser.* 等调用；既有 benchmark 后端实际提供 search/visit。训练保留原始调用和 observation，不虚构论文检索服务。评测通过 RUN_DIR 自动读取 prepared/train.jsonl 时，`resolve_system_prompt` 识别 browsecomp 来源和 training_only 标记，只将评测提示中的 schema 换成实际可执行的 search/visit，并明确声明历史工具不可用；离线再按已有规则限制为本地工具。研究任务说明仍保留，训练记录不改写。

原始模型做同提示对照时，可设置 `PROMPT_DATA=outputs/data/browsecomp-qwen35-sft/prepared_16k/train.jsonl`。应指向 prepared 文件；直接指定未适配的原始 JSONL 没有来源标记，不会自动执行此能力切换。这里的提示适配不代表模型已学会新工具接口，实际遵循程度需要真实生成验证。

文件名不能证明与 BrowseComp 测试题独立。报告 BrowseComp 成绩前，应依据问题文本/来源核查与此训练文件的重叠，并注明训练来源、测试题版本和重叠处理；本次没有确认其为独立 held-out 划分。标准化 question_id 已按问题空白归一化、casefold 后计算 SHA256，可用于后续重叠审计。

## 14.6 本次验证及复查命令

标准库回归入口，不依赖 pytest 或训练环境：

```bash
python -m unittest discover -s tests -p test_browsecomp_sft_and_hotpot_judge.py -v
```

全量源文件已完成规范化和字节 tokenizer 编码边界检查：372/372 成功，13,021 个 observation 区域均被 loss 屏蔽，4 个字面边界标签转义。输入指纹及审计结果保存在 [14 结构审计 JSON](14_browsecomp_structure_audit.json)。字节 tokenizer 只证明边界和监督规则，不证明真实模型 token 长度、GPU 行为或训练效果。完整检查范围见 [99 验证记录](../project/99_validation.md)。
