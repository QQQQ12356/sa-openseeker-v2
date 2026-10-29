# 45 Plain / Beacon 训练与在线、离线评测的统一流程

## 45.1 目的与适用范围

OpenSeeker 标准 search/visit 任务现在共用提示词构造、工具 schema、调用协议、ChatML 编码、答案提取和 native agent 循环。Plain 和 Beacon 不再各自选择提示词；区别在于模型是否压缩 observation。在线 search/visit、本地固定语料、replay、HTTP 网关，以及官方 v1/v2 评测入口全部保留。

该修改不改变 Beacon 压缩公式、线性状态 writer、LoRA 算法、分段损失、官方评分函数或离线索引。Search-R1 保留独立的 `<search>/<information>` 协议，不把旧数据强行改成 OpenSeeker。自定义工具轨迹保留其原 system/schema；不能将任意工具视为 search/visit。

## 45.2 OpenSeeker 参考流程：哪些可以统一

参考本地 `outputs/vendor/OpenSeeker`，固定提交 `e3dec1f727d87c5eaa93506e4882c67237647098`：

- [官方源码](https://github.com/PolarSeeker/OpenSeeker/tree/e3dec1f727d87c5eaa93506e4882c67237647098) 的 `src/llm_tool_openseeker.py` 定义 v1 developer_prompt、Search、Visit，并在每轮循环中用 `_render_prompt()` 渲染完整历史。
- `src/config/chat_template.jinja` 定义 system、工具 schema、工具响应、assistant 思考和生成前缀。推理时使用同一套工具语法，接收真实工具结果，再接续下一轮。
- 本项目 Polar 训练轨迹的原 system 使用相应 search/visit 提示，工具调用为 `{"name": ..., "arguments": ...}`，工具结果为 `<tool_response>`。这种“轨迹和推理协议一致”是本次统一的依据。
- 固定上游 checkout 提供推理、工具和部署代码，不能据此声称已经验证其未提供的完整训练器。训练侧核对依据为本地公开轨迹和本项目编码器。
- 官方 v2 另有工具和 boxed 答案要求，不能把 v2 当作 v1 的同一提示词。官方 Jinja 还会处理历史 reasoning 并以 `<think>` 开启生成；本项目保留原始 assistant 文本，生成前缀为 `<|im_start|>assistant\n`，由模型输出 think 标签。native 不是官方模板的逐字复现。

本项目选择统一自己的训练和 native 推理；需要原样官方复现时显式选择 `official-v1` 或 `official-v2`，不修改 vendor 文件。

## 45.3 唯一提示词来源与必要扩展

源码为 `search_comp/data/agent_prompt.py`：

| 定义 | 作用 | 修改提示 |
|---|---|---|
| `BASE_INSTRUCTION` | 所有标准任务的基础角色 | 修改后重新 prepare 并训练；不能将新提示词成绩直接归入旧实验 |
| `TOOL_SCHEMAS` | search/visit 的参数结构和通用描述 | 不承诺固定 top-k、互联网访问或摘要模型；这些由实际后端决定 |
| `TOOL_PROTOCOL` | JSON envelope、observation、think、answer 约定 | 与 `openseeker.parse_calls()` 同步修改，更新协议版本和回归 |
| `build_system_prompt()` | 组合基础提示、能力和任务要求 | 按后端参数构造，不按 plain/beacon 分支 |
| `initial_messages(question)` | 构建在线 system/question | 不接收参考答案、qrels 或证据标注 |
| `PROMPT_VERSION` | 提示词身份 | 内容语义变化时更新，并重新计长 |
| `prompt_metadata(system)` | 记录版本和 system 的 SHA256 | 自定义 system 标记为 custom，不冒充共享提示 |

`build_system_prompt()` 的参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `allow_visit` | true | false 时从 schema 中移除 visit，并明确只允许 search |
| `local` | false | true 时限定固定本地语料和 local:// 文档读取，不允许联网 |
| `answer_instruction` | 空字符串 | 附加基准必需的引用或答案格式要求 |

标准训练与 native 在线评测的基础 system 完全相同。离线评测复用同一个构造器，但必须保留真实环境差异：BCP 要求 `[docid]` 引用；Hotpot 要求 answer/sp JSON；search-only 不能假装支持 visit。这些扩展不会因模型类型而变化。比较 Plain/Beacon 时应固定相同后端、能力选项和任务要求。

## 45.4 新数据协议与旧数据迁移

当前协议为 `openseeker-chatml-v4-unified-agent`；旧协议为 `openseeker-chatml-v3-original-answer`。

标准 search/visit 轨迹在 `normalize()` 中替换 system 为共享提示，问题、原始 assistant 思考/工具调用/最终答案、工具 observation 和独立 answer 参考保持不变。工具名包含其他工具时保留自定义 system，不做标准任务替换。原始源文件不会被修改。

修改 system 会改变 token 长度、压缩区域的绝对偏移及长度筛选结果，因此不能只改 manifest 中的版本字符串。`normalize()` 支持读取 v3 并升级，但会清除旧 `lengths`；训练 Dataset 拒绝直接读取 v3；实验入口也拒绝 v3 manifest。必须通过 prepare 重新分词、计长和筛选。

默认新输出目录为 `outputs/data/openseeker/polar_correct_qwen35_16k_unified`，旧目录和旧结果保留。10/20/30 与 `configs/openseeker.yaml` 以及筛选、评测回退路径和数据报告入口已统一到新路径。12 的显存样本输出与 26/37 的输入同步使用 `polar_memory_probe_unified`，避免默认连接旧 v3 数据。

```bash
# 只加载 tokenizer，不加载模型权重，不训练。输出目录必须尚不存在。
bash scripts/sh/10_prepare.sh

# 另选输入和输出，均可通过同名环境变量覆盖。
INPUT_FILE=/data/polar.jsonl OUTPUT_DIR=/data/polar_unified \
  MODEL_NAME=/models/Qwen3.5-4B bash scripts/sh/10_prepare.sh
```

优先从原始数据重建。若只有旧规范化数据，可将其作为 `INPUT_FILE`；输入 train/validation/test 多文件时编辑 INPUTS 数组，保留旧划分。只输入旧 train 不能恢复原 validation/test；只输入旧筛选结果也不能恢复之前被过滤的样本。prepare 的长度/正确性参数仍会重新生效。

10_prepare.sh 参数：

| 变量 | 默认 | 含义与修改提示 |
|---|---|---|
| `PYTHON` | 空 | Python 绝对路径；优先于 CONDA_ENV |
| `CONDA_ENV` | search-comp-qwen3.5 | 已安装依赖的环境；不自动安装 |
| `SOURCE` | polar | polar / zephyr / combined，标记输入来源 |
| `INPUT_FILE` / `INPUTS` | 原 Polar JSONL | 前者适合单文件环境变量覆盖；多文件编辑脚本数组 |
| `OUTPUT_DIR` | 上述 `_unified` 目录 | 必须新建；不会覆盖旧数据 |
| `MODEL_NAME` / `MODELS` | Qwen/Qwen3.5-4B | tokenizer 身份必须与训练模型一致；多模型编辑数组 |
| `MIN_LENGTH` / `MAX_LENGTH` | 1 / 16384 | 完整 ChatML 序列闭区间；超长过滤，不截断 |
| `CORRECTNESS` | Correct | Correct / Incorrect / any；用于质量筛选 |
| `OBSERVATIONS` | any | any / with / without；按 observation 正文判断 |
| `WORKERS` | 4 | tokenizer 预处理进程数；内存不足时减少 |
| `OFFLINE` | 1 | 仅用本地 tokenizer 缓存；0 允许下载 tokenizer |
| `VAL_FRACTION` / `TEST_FRACTION` | 0.02 / 0.02 | 问题分组划分比例；训练为剩余部分；和必须小于 1 |
| `SEED` | 42 | 稳定哈希划分的 seed |

末尾 CLI 标量参数可覆盖脚本默认值，例如 `--max-length 32768`。`--input` 是追加列表，额外传它不会替换 INPUTS；单输入替换应使用 INPUT_FILE。MAX_LENGTH、模型和输出目录变化后，需要同步修改训练脚本中的 dataset/max_length/models。

## 45.5 训练与生成算法的统一边界

```text
原始轨迹 → normalize(共享 system) → encode(完整 ChatML / labels / regions)
                                  ├─ Plain：标准因果 LM，忽略 regions
                                  └─ Beacon：同一 token/labels，仅压缩 regions

新问题 → 同一提示词构造器 → run_agent → 同一 encode(历史 + 生成前缀)
                                  ├─ Plain：model.generate(显式 greedy GenerationConfig)
                                  └─ Beacon：prefill + decode_step，贪心生成
```

训练中真实 assistant token 作为前文；测试中模型自己的输出作为前文。这是监督训练与自回归测试的正常差异，不通过注入正确答案来“统一”。dropout 只在训练启用。native Plain 显式使用新的贪心 GenerationConfig，不继承 checkpoint 中的 beam、采样或 repetition penalty；Beacon 使用相同的 argmax 策略。两路使用相同停止文本和模型 EOS（支持多个 EOS ID）。当前 native 轮间 `reuse_cache=False`，完整历史每轮重新 prefill；轮内逐 token 使用缓存，不跨题复用状态。

Beacon 的监督 forward 和推理 prefill 共用 memory、step、native_forward 和 update_memory，Qwen3.5 全注意力 KV 与线性循环状态采用相同提交规则。新 tiny CPU 测试比较相同前缀的 logits 和缓存，覆盖 append/intersect，以及有/无压缩；不代表所有真实 GPU kernel、精度和 checkpoint 已完成数值验证。

Beacon SFT 布局由 `models/beacon_config.configure_whole_trajectory()` 统一解析，在 experiment 归档前和 trainer 加载前均调用：whole、window_enabled=false、doc_region_split=false、keep_window=0。避免归档写 1024 而实际训练使用 0。模型保存后的 config 才是推理的配置来源；不要只拿 source_config.yaml 判断实际运行值。

随机压缩仍为训练每轨迹采样一次，验证/生成使用 beacon_ratio；这是保留的算法选项。需要严格固定压缩比比较时使用 fixed，并检查结果中的 generation_config.beacon_config。

训练一键入口不变：20 Plain、30 Beacon、28/38 四卡；只审查配置用 `ACTION=plan`。本次没有运行真实训练，尚未生成 v4 checkpoint。

## 45.6 在线、离线与历史诊断入口

| 入口 | 模型 | 工具/数据 | 提示词与评分 |
|---|---|---|---|
| 21 / 31 | Plain / Beacon | replay / corpus / HTTP | 当前规范化轨迹 system，共享 native Agent；历史 EM/F1 保留 |
| 47，默认 native | Plain / Beacon | 官方 v1 在线 Search/Visit | 共享 system/encoder；官方裁判 |
| 47，official-v1 | Plain completion 服务 | 官方 v1 工具 | 原样官方提示/模板/生成/裁判；显式复现模式 |
| 47，official-v2 | Plain completion 服务 | 官方 v2 + E2B | 原样 v2 与 boxed 指令；显式复现模式 |
| 48 | Plain | SGLang completion 服务 | 服务入口保留，配合 official 模式使用 |
| 56 | Plain / Beacon | BCP 固定语料 BM25 | 共享基础提示 + 本地/引用约束；官方评分逻辑 |
| 57 | Plain / Beacon | Hotpot 官方 Wiki 摘要库 | 共享基础提示 + 本地/JSON约束；官方 EM/F1、SP/Joint |

47 的默认值从 official-v2 改为 native，Python 与 shell 一致。原先只设置 OPENSEEKER_BASE_URL/MODEL 的部署脚本，需要显式 `MODE=official-v1` 或 `MODE=official-v2`；原功能未删除。

```bash
# CPU 协议和 tiny 算法回归，不训练、不请求在线 API。
bash scripts/sh/59_check_agent_alignment.sh

# 只显示计划，不加载模型或调用 API。
DRY_RUN=1 bash scripts/sh/47_openseeker_benchmark.sh

# 真实在线运行示例：由使用者在配置好模型/API 后执行，Plain/Beacon 用同一命令。
MODE=native RUN_DIR=/path/to/run ENV_FILE=/private/api.env LIMIT=8 \
  bash scripts/sh/47_openseeker_benchmark.sh

# 原样官方流程继续保留。
MODE=official-v2 ENV_FILE=/private/api.env \
  bash scripts/sh/47_openseeker_benchmark.sh

# 固定语料测试：先按 43 文档完成 download/prepare/index。
ACTION=run RUN_DIR=/path/to/run LIMIT=8 bash scripts/sh/56_browsecomp_plus.sh
ACTION=run RUN_DIR=/path/to/run LIMIT=8 bash scripts/sh/57_hotpotqa_offline.sh
```

统一的是 native 推理协议；离线工具仍返回 JSON 文档，在线工具仍返回上游摘要文本。不伪造互联网搜索结果，不改变固定语料权限，也不把 replay 当作真实检索成绩。若希望研究 observation 格式对模型的影响，应作为独立实验同时适配训练数据和工具渲染，再重新计长训练。

## 45.7 测试参数与修改位置

| 变量 | 入口/默认 | 含义与注意事项 |
|---|---|---|
| `MODE` | 47 / native | native 统一 Plain/Beacon；official-v1/v2 原样复现 |
| `RUN_DIR` | 47/56/57 / 空 | 读取实验与 train_summary 定位已合并模型 |
| `MODEL_PATH` | 同上 / 空 | 显式模型目录；覆盖 RUN_DIR 自动定位；不能是未合并 adapter |
| `KIND` | 同上 / 自动 | plain/beacon；统一比较不修改提示词；Beacon 不能改用普通服务来替代 |
| `ENV_FILE` | 47 / 空 | 私有在线 API shell 配置；不影响严格离线入口 |
| `DATA` / `CORPUS` / `ROOT_DIR` | 56/57 | 问题、独立语料、索引根目录；保留原来源/指纹验证 |
| `ALLOW_VISIT` | 56=0，57=1 | 实际能力和 schema 同步；两个模型必须相同 |
| `TOPK` | 56/57=5 | 每个查询结果数；通过后端控制，不硬编码在提示词 |
| `SNIPPET_TOKENS` | 56/57=512 | 每篇搜索正文上限；按待测 tokenizer 计数 |
| `READ_TOKENS` | 56/57=4096 | 每篇 visit 正文上限；不是整轮工具返回预算 |
| `MAX_TOOL_TOKENS` | 47/56/57=32768 | 单条完整工具响应上限，含包装/JSON；超出则失败，不静默截断 |
| `MAX_CONTEXT` | 47/56/57=131072 | 未压缩完整历史的逻辑 token 上限；不是 Beacon 物理 KV 长度 |
| `MAX_TOKENS` | 47=32768 | native 每轮最大生成 token；official 模式按上游参数处理 |
| `MAX_NEW_TOKENS` | 56/57=8192 | 离线每轮最大生成 token，不是整题总量 |
| `MAX_TURNS` | 47/56/57=200 | native 循环上限，不能解决单轮重复生成 |
| `TOOL_COUNT` | 47/56/57=200 | 离线为实际调用上限；47 该参数沿用官方模式上限，native 仍以 MAX_TURNS 控制循环 |
| `LIMIT` | 47/56/57=-1 | 全量；正数仅前 N 题，不报告为全量成绩 |
| `NUM_WORKERS` | 56/57=1 | 每卡完整模型副本，按题分片；非 tensor parallel |
| `WORKERS` / `JUDGE_WORKERS` | 47=60/20 | 官方在线生成/裁判并发；native 生成逐题，WORKERS 不提供 native 多卡并发 |
| `JUDGE_MODEL` | 56 / 空 | 本地独立裁判；ACTION=run 可仅生成，完整评分需配置 |
| `OUTPUT_DIR` | 测试 / 新时间戳目录 | 不覆盖旧结果；不要在旧结果目录继续写入 |
| `PREDICTIONS` | 评分入口 / 空 | 只评分已有预测时使用；提示词、checkpoint 来源以原预测记录为准 |
| `DRY_RUN` | 47/56/57=0 | 1 只审查计划；56/57 仍有原有的本地输入/参数预检查 |

59 的 PYTHON/CONDA_ENV 与项目相同；OMP_NUM_THREADS 默认 1，仅用于 CPU 回归；末尾参数传给 pytest。21 是设置 KIND=plain 后转入 31 的薄包装；两者的全部参数在 31 顶部说明，支持同名环境变量覆盖。示例：`RUN_DIR=/path/to/run BACKEND=replay DRY_RUN=1 bash scripts/sh/21_plain_eval.sh`。21/31 的专用参数见 40，索引、评分和网络隔离的完整参数见 43，在线密钥与官方部署参数见 42。

## 45.8 结果身份、复现与排查

每条 native 预测新增 prompt_version、system_prompt_sha256、generation_config。实际 messages 仍完整保存，便于直接检查提示词。generation_config 记录 kind、贪心解码、checkpoint、model_config_sha256，Beacon 额外记录实际 beacon_config。自定义生成器测试桩可没有 generation_config 信息；不能把空字典解释为真实模型来源已核验。

比较两个结果应核对：问题/语料指纹、索引设置、能力和答案要求、system hash、模型目录、实际模型/Beacon 配置、预算、裁判。离线扩展会改变完整 system hash，这是预期行为，不应为了让 hash 一样而删除环境约束。

v3 checkpoint 可继续用于诊断和官方复现，但修改代码不会将旧权重变成 v4 训练权重，也不能保证原有重复生成立即消失。需要评估改动效果时，先重建 v4 数据、训练新模型，再按相同环境比较完成率、非法调用率、重复行为、召回和准确率。

如果只想修改模型、压缩比、训练步数或预算，编辑对应脚本顶部变量/YAML；无需复制提示词。若修改提示词/schema/序列化，需同步升级数据协议、重新 prepare、建立新实验，运行 59，并在结果中保留版本与指纹。不要改写旧实验代码指纹来继续训练。
