# 14 BrowseComp-Plus SFT 数据预处理、参数与运行说明

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

## 14.3 分阶段一键运行（Linux / Bash）

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

14 是唯一的数据预处理入口，只加载 tokenizer，在 CPU 上规范化和计长；25/39 只读取已经准备好的数据，不调用 14，也不调用 20/30。每个阶段均可通过一条命令运行。无需添加“先预处理再训练”的包装训练脚本。

首次使用按顺序运行上面三条命令中的预处理和所需方法的计划命令。预处理输出目录必须不存在；已有数据可以直接运行 25/39。缺少 manifest 时，训练入口报错并提示先运行 14，不会创建 prepared 数据。`ACTION=plan` 也不会预处理、加载模型权重或训练，但会读取已有数据、校验配置并创建实验归档。

正式训练设置 `ACTION=train`：

```bash
# 下面两条按实验需要选择运行，每次创建独立实验。
MODEL_PATH=/data/models/Qwen3.5-4B ACTION=train \
  bash scripts/sh/plain/25_plain_browsecomp_train.sh
MODEL_PATH=/data/models/Qwen3.5-4B ACTION=train \
  bash scripts/sh/beacon/39_beacon_browsecomp_train.sh
```

多卡设置 `CUDA_VISIBLE_DEVICES=0,1,2,3 DISTRIBUTED_BACKEND=zero2 ACTION=train`；进程数默认按可见卡数推断，可用 `NPROC_PER_NODE=4` 显式指定。完整训练参数和调参示例分别见 [25 Plain](../plain/25_plain_browsecomp_train.md)、[39 Beacon](../beacon/39_beacon_browsecomp_train.md)，共有训练及分布式参数见 [53](../tools/53_self_contained_training.md)。

## 14.4 脚本变量、优先级与修改提示

参数有三个位置：14 顶部的数据变量、25/39 的可选训练覆盖变量、25/39 各自完整的内联 YAML。数据变量不会自动跨两次脚本调用传递；跨阶段要么提前 `export`，要么在每条命令上重复传入。脚本均自动切换到项目根目录，数据相对路径按项目根目录解释；在项目外启动时须使用脚本的绝对路径。

### 14.4.1 预处理入口变量（只用于 14）

| 变量 | 默认 | 作用、约束与修改提示 |
|---|---|---|
| `INPUT` | `outputs/data/browsecomp-qwen35-sft/train.jsonl` | 必须存在的原始 JSONL 文件；只读，不接受 benchmark 问答表代替工具轨迹 |
| `DATASET_DIR` | `outputs/data/browsecomp-qwen35-sft/prepared_16k` | 必须不存在的输出目录；运行成功后交给 25/39。重新处理用新目录，脚本不覆盖旧产物 |
| `MODEL_PATH` | `Qwen/Qwen3.5-4B` | tokenizer 来源；建议本地绝对路径。训练时使用完全相同的身份字符串，不能一阶段用 Hub 名、一阶段换本地路径 |
| `MAX_LENGTH` | `16384` | 正整数；完整轨迹 token 上限，超长拒绝，不截断。改变长度同时改输出目录 |
| `WORKERS` | `4` | 正整数；CPU 分词进程数，内存不足设 1；不影响 GPU 数 |
| `SEED` | `42` | 数据分组/划分随机种子；该来源不划分 val/test，也不自动设置训练 seed |
| `OBSERVATIONS` | `with` | `with` 至少一个非空 observation 正文；`any` 不筛；`without` 无正文对照，改变时使用新目录 |
| `OFFLINE` | `1` | 仅接受 0/1；1 仅用本地 tokenizer 缓存，缺失则报错；0 允许库下载 tokenizer，不安装依赖 |
| `PYTHON` | 空 | 已有 Python 可执行文件；留空按 `00_common.sh` 选择，建议通过 `PYTHON=/absolute/path/bin/python` 显式指定 |
| `CONDA_ENV` | `search-comp-qwen35-cu126` | 选择已有 conda 环境，不创建、不安装；有效的显式 `PYTHON` 优先 |

14 不使用 `ACTION`、GPU、LoRA 或训练超参数。脚本拒绝额外位置参数或 CLI 选项，避免变量与命令行重复覆盖；需扩展预处理功能时修改 14 中的命令数组。固定协议参数为 `--source browsecomp --correctness any --val-fraction 0 --test-fraction 0`，不得改成有正确性金标签或 held-out 示范评测。

### 14.4.2 训练入口变量（25/39）

| 变量 | 默认/来源 | 作用与修改提示 |
|---|---|---|
| `DATASET_DIR` | 空，回退 YAML `dataset` | 非空才覆盖；只选择已有 prepared 目录，不生成、不筛选数据 |
| `MODEL_PATH` | 空，回退 YAML `models[model_family]` | 非空覆盖当前模型族路径；不改变 `model_family`，不重新分词 |
| `MAX_LENGTH` | 空，回退 YAML `training.max_length` | 非空正整数覆盖训练上限；不重做过滤，应与 prepared 长度设置一致 |
| `ACTION` | `plan` | 只接受 `plan` / `train`。plan 创建实验归档；train 创建新实验并启动训练 |
| `PYTHON` / `CONDA_ENV` | 与 14 相同 | 使用已安装训练依赖的环境；14 能运行不代表 GPU 训练依赖齐全 |
| `CUDA_VISIBLE_DEVICES` | `0` | 可见 GPU 列表，如 `0,1,2,3` |
| `NPROC_PER_NODE` | 按可见 GPU 数推断 | 可显式设正整数，需与分布式资源一致 |
| `DISTRIBUTED_BACKEND` | `zero2` | 多进程可选 ddp/zero2；Plain 另支持 zero3；单进程配置见 53 |
| `TMPDIR` | `/tmp` | 临时 YAML 所在目录，需存在且可写；文件退出时删除，正式配置保存在实验目录 |

25/39 不读取 `INPUT`、`WORKERS`、`SEED`、`OBSERVATIONS`、`OFFLINE` 来处理数据；训练 seed 直接修改各自 YAML 的 `training.seed`。训练脚本不接受位置参数。`MODEL_PATH` 等环境覆盖仅对当前调用生效，建议先 `ACTION=plan` 检查归档配置。

覆盖优先级为：**非空环境变量 > 对应训练脚本内联 YAML**。三项可选覆盖默认留空，因此直接修改 YAML 的 `dataset`、模型路径或 `training.max_length` 就会生效。如果曾 `export` 同名变量，编辑 YAML 前执行 `unset DATASET_DIR MODEL_PATH MAX_LENGTH`，或明确保留所需覆盖。20/30 的参数修改不会传递到 25/39。

### 14.4.3 修改长度和模型的完整示例

```bash
# 一个 shell 会话中设置共同参数，两个阶段使用完全相同的值。
export MODEL_PATH=/data/models/Qwen3.5-4B
export MAX_LENGTH=32768
export DATASET_DIR=outputs/data/browsecomp-qwen35-sft/prepared_32k

# 只执行一次；该目录必须尚不存在。
WORKERS=4 OFFLINE=1 bash scripts/sh/data/14_prepare_browsecomp.sh

# 选择所需训练方法；均只读取上一步的结果。
ACTION=plan bash scripts/sh/plain/25_plain_browsecomp_train.sh
ACTION=plan bash scripts/sh/beacon/39_beacon_browsecomp_train.sh
# 检查归档配置后正式训练。
ACTION=train bash scripts/sh/plain/25_plain_browsecomp_train.sh
```

不要仅在 25/39 中改 `MAX_LENGTH` 就认为完成了数据重新过滤。改变原始输入、tokenizer、长度或 observation 筛选时，用新目录重新运行 14，再将训练 `DATASET_DIR` 指向它。训练上限小于 manifest 准备上限或 tokenizer 身份不匹配时，实验入口拒绝继续；不要手改 manifest 绕过验证。

### 14.4.4 常见问题

| 提示或现象 | 检查与处理 |
|---|---|
| 原始文件不存在 | 在 14 设置 `INPUT`，相对路径按项目根目录解释 |
| 输出目录已存在 | 成功的 prepared 直接用于训练；失败留下的目录先检查，改用新目录重新处理 |
| 离线 tokenizer 加载失败 | 指定本地 `MODEL_PATH` 或在允许联网的环境设置 `OFFLINE=0` |
| 训练缺少 `manifest.json` | 先独立运行 14，确认成功，再给 25/39 设置相同 `DATASET_DIR` |
| tokenizer 身份不匹配 | 两阶段 `MODEL_PATH` 必须一致；更换模型后重新准备数据 |
| 修改 YAML 未生效 | 检查已 export 的三项覆盖变量，以及是否在查看旧实验归档 |
| 没有验证 loss | 本来源 val/test 为空是既定设计；使用独立 benchmark 评估 |
| 没有可用训练样本 | 查看 manifest 计数及 rejected.jsonl；不要通过截断或修改标签伪造可用数据 |

## 14.5 产物、追溯与评测边界

实验目录名称使用数据集标识，与训练轨迹协议分开：25/39 内联 YAML 设置 `dataset_name: browsecomp-plus`，生成 `qwen35_plain_browsecomp-plus_L16384_...` 或 `qwen35_beacon_browsecomp-plus_L16384_...`。`data_mode: openseeker` 保持为实际使用的编码协议。`dataset_name` 只影响新实验命名与归档，修改时使用字母、数字、下划线、连字符，首字符为字母或数字，不包含路径分隔符。旧配置未指定时，manifest 的 `arguments.source=browsecomp` 自动识别为 `browsecomp-plus`，其他来源保留原协议名称。已有实验目录及其 `--run-dir` 身份不自动改名。

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

2026-11-04 数据与训练分离后的脚本验证及环境限制见 [99 最新记录](../project/99_validation.md)。下述全量审计为此前记录，本次未重新处理真实数据。

此前全量源文件已完成规范化和字节 tokenizer 编码边界检查：372/372 成功，13,021 个 observation 区域均被 loss 屏蔽，4 个字面边界标签转义。输入指纹及审计结果保存在 [14 结构审计 JSON](14_browsecomp_structure_audit.json)。字节 tokenizer 只证明边界和监督规则，不证明真实模型 token 长度、GPU 行为或训练效果。完整检查范围见 [99 验证记录](../project/99_validation.md)。
