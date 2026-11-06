# 60 脚本参数与修改手册

## 1. 参数来源与优先级

公共入口是独立 SH 文件。方法名和 benchmark 固定；所有可调参数和环境变量在 SH 顶部修改，PY 仅定义类型和执行逻辑。每次实验保存 SH、PY 副本与最终设置。

- `ENVIRONMENT` 先应用到进程环境，再导入 tokenizer/模型/Hub 客户端。
- `TRAINING`、`BEACON`、`DISTRIBUTED` 为当前文件的完整训练配置，由对应 SA_* 环境变量还原，不继承另一方法配置。
- 测试的模型及训练提示词读取 `TRAINING_RUN`，不接受模型覆盖；baseline 使用本地 `MODEL_PATH`。
- 评分的 benchmark 版本和 `limit` 来自 `EVALUATION_DIR`，避免评分与生成范围不一致。
- 相对路径均以项目根目录解析；模型 tokenizer 身份必须使用预处理时相同的字符串。不要使用 Windows 盘符路径。

本手册保留 TRAINING.max_length 等逻辑字段便于对照算法；实际 SH 名称统一为 SA_TRAINING__MAX_LENGTH。普通顶层字段加 SA_，ENVIRONMENT 下的真实环境变量不加前缀。每份 SH 的完整实际名称见 [61 参数清单](61_script_inventory.md)。布尔值在 SH 中写 true/false，None 在 SH 中用空字符串表示。

benchmark 评分后的 Excel 导出使用 `scripts/tools/sh/82_benchmark_excel.sh`。其参数 INPUTS、OUTPUT_DIR、RUN_NAME、INCLUDE_DETAILS、INCLUDE_INCOMPLETE 的完整作用、示例、输出口径和修改提示集中在 [82 Excel 导出](82_benchmark_excel.md)，不复用其他工具的同名参数默认值。

## 2. 环境与公共路径

| 变量 | 作用与修改提示 |
|---|---|
| ENVIRONMENT.HF_ENDPOINT | Hub 来源；官方 https://huggingface.co 或镜像 https://hf-mirror.com。 |
| ENVIRONMENT.HF_HUB_OFFLINE | 下载使用 0，本地训练/测试/评分使用 1；离线 benchmark 另禁止 socket。 |
| ENVIRONMENT.CUDA_VISIBLE_DEVICES | 可见 GPU 编号列表；离线多副本与训练多进程应有足够的不同卡。 |
| ENVIRONMENT.NPROC_PER_NODE | 训练进程数，默认 1；多卡与可见卡数同步。 |
| ENVIRONMENT.DISTRIBUTED_BACKEND | ddp/zero2；Plain 可选 zero3；单卡不使用 ZeRO。 |
| HF_HOME / TORCH_HOME / XDG_CACHE_HOME | 共享层默认缓存到 artifacts/cache；服务器已设置时保留。 |
| RUN_NAME | 一次实验/测试/评分的语义标签；1–96 字符，仅英文字母、数字、下划线、连字符。 |
| DATASET_DIR | 训练使用的 prepared 数据版本。 |
| MODEL_PATH | 训练基座或 baseline 本地权重；与 tokenizer 计长身份一致。 |
| MODEL_FAMILY | 训练适配族 qwen3/qwen35；baseline 固定原始 Qwen3.5。 |
| PLAN_ONLY | True 只校验/归档，不加载模型；False 正式训练且创建新记录。 |
| TRAINING_RUN | 已成功完成的训练实验根目录；baseline 必须 None。 |
| BENCHMARK_DIR | 准备好的共享 benchmark 版本目录。 |
| PROMPT_DATA | 训练方法必须 None；baseline 可选单一 system 的规范化 train.jsonl。 |
| EVALUATION_DIR | 成功测试目录，含 job.json、evaluation.json、results/；不能填单个预测文件。 |

在线页面读取与总结的 ENVIRONMENT 字段：`SUMMARY_ATTEMPTS` 为有限重试次数（1–10，默认 3）；`SUMMARY_MAX_INPUT_CHARS` 为总结输入字符上限（40000）；`SUMMARY_MAX_TOKENS` 为总结输出 token 上限（2048）；`SUMMARY_TIMEOUT` 为总结超时秒数（120）；`JINA_TIMEOUT` 为页面读取超时秒数（50）。`SUMMARY_API_URL` 和 `SCORER_URLS` 填基础地址如 `http://host:8000/v1`，客户端追加 `/chat/completions`。

## 3. 数据准备与筛选

| 变量 | 作用与修改提示 |
|---|---|
| REPO_ID / MODEL_NAME | 模型下载仓库与本地归档名；改变仓库同步修改归档名。 |
| REVISION | Hub tag/branch/commit；下载时解析并记录不可变 commit。 |
| SOURCE | OpenSeeker: polar/zephyr，联合本地预处理可 combined；Research: research。 |
| SNAPSHOT_DIR | 原始下载 commit 目录，读取下载清单的完整 jsonl_inputs。 |
| INPUTS | 显式本地 JSONL/Parquet 文件列表；与 SNAPSHOT_DIR 互斥。 |
| PROFILE | prepared 数据版本名；改变来源、模型、筛选或划分时新建版本。 |
| MODELS | 预处理 tokenizer 身份列表；所有计量都需满足长度筛选。 |
| OFFLINE | 是否仅使用本地 tokenizer；默认 True，不加载模型权重。 |
| WORKERS | CPU 计长进程数，正整数。 |
| MIN_LENGTH / MAX_LENGTH | 完整轨迹 token 范围，至少 1；不静默截断。 |
| CORRECTNESS | OpenSeeker 与 Research 均支持 Correct/Incorrect/any；前两者精确匹配原始标签，any 包含缺失/未知标签。Research 默认 any；预处理及二次筛选均可修改，变更时使用新 PROFILE。 |
| OBSERVATIONS | any/with/without：工具观测筛选。 |
| SEED | 同题分组哈希划分种子；固定输入与种子便于复现。 |
| VAL_FRACTION / TEST_FRACTION | 非负比例，总和小于 1；公开划分优先，实际比例近似。 |
| PREPARED | 重新筛选的父 prepared 版本，需保留 normalized 与原始输入。 |
| TOKENIZER | 重新筛选时选用的已计量 tokenizer 身份。 |
| EXACT_LENGTH | 可选精确 token 长度，None 关闭；必须位于 MIN/MAX 范围内。 |
| VERSION | benchmark 资产版本名；改变来源、语料或索引参数时新建版本。 |
| PHASE | 离线依次 download/prepare/index，每次编辑后运行；在线 BrowseComp 只需 download。 |
| CSV | BrowseComp 官方加密本地 CSV；None 从官方地址下载。 |
| DATA / CORPUS | 离线官方题目和独立语料路径；None 使用下载阶段的默认资产。 |
| CORPUS_REVISION | BrowseComp-Plus 语料 revision；None 首次解析并写入 assets.json。 |
| DECRYPTED | BrowseComp-Plus 题目是否已解密；HotpotQA 保持 False。 |
| TITLE_WEIGHT | BM25 标题权重，默认 1.0；建索引与测试必须相同。 |

## 4. GENERATION 推理参数

| 字段 | 适用范围 | 默认值 | 作用与修改提示 |
|---|---|---|---|
| limit | 全部 | -1 | -1 全量，正数取源顺序前 N 条；小样本不能当全量成绩。 |
| max_context | 全部 | 131072 | 单问题完整序列上下文预算；应符合模型能力和显存。 |
| max_new_tokens | 全部 | 8192 | 每轮 assistant 生成上限，实际还受剩余上下文限制。 |
| max_turns | 全部 | 200 | Agent 最大生成轮数。 |
| max_tool_tokens | 全部 | 32768 | 单条工具返回编码长度上限；超限终止，不截断观测。 |
| tool_count | 全部 | 200 | 每题工具调用对象数量上限；批量 query 仍属于一个 search 调用。 |
| tool_call_retries | 全部 | 1 | 工具调用格式非法时允许的修正次数，不重试真实工具失败。 |
| decoding | 全部 | greedy | greedy/sample；Beacon 只支持 greedy。 |
| repetition_penalty | 全部 | 1.0 | 重复惩罚，正有限值；1.0 不施加额外惩罚。 |
| temperature | 全部 | 1.0 | sample 时的正温度；greedy 不采样。 |
| top_p | 全部 | 1.0 | sample 的 nucleus 概率阈值，范围 (0,1]。 |
| generation_seed | 全部 | 42 | 推理随机种子；对 sample 有意义。 |
| num_workers | 离线 | 1 | 每 GPU 一个完整模型副本；不要与训练 NPROC 混淆。 |
| title_weight | 离线 | 1.0 | 必须与已建索引的标题权重一致。 |
| verify_index | 离线 | True | 启动时扫描索引摘要，增加大索引读取时间。 |
| topk | 离线 | 5 | 每条查询返回的最高排名文档数，正整数。 |
| snippet_tokens | 离线 | 512 | 每篇搜索结果的正文摘录 token 上限。 |
| read_tokens | 离线 | 4096 | visit 读取本地文档的正文 token 上限。 |
| allow_visit | 离线 | True | 是否允许本地 visit；False 采用 search-only 提示词。 |

BrowseComp 使用固定版工具与 stable 页面读取策略，原生生成逐题执行；在线工具服务的检索数量由其自身协议决定，不暴露离线 topk 参数。

## 5. JUDGE 裁判参数

| 字段 | 适用范围 | 作用与修改提示 |
|---|---|---|
| judge_model | 离线 | 本地裁判目录，默认占位 /models/judge，执行前必改；不能填在线端点。 |
| llm_judge | HotpotQA | 默认 True，加语义裁判；需要本地 judge_model。仅算官方 EM/F1 时同时设 False 和 judge_model=''。 |
| judge_workers | BrowseComp | 远程裁判并发数，默认 1。 |
| judge_attempts | 全部 | 最多尝试次数，默认 3；解析失败仍保留错误状态和分母。 |
| judge_max_tokens | 全部 | 每次裁判输出预算；在线默认 1024，离线默认 4096。 |
| judge_timeout | BrowseComp | 单次在线请求超时秒数，默认 120。 |
| judge_enable_thinking | 全部 | 是否允许 thinking，默认 False；与裁判兼容性一致。 |
| judge_max_context | 离线 | 本地裁判上下文上限，默认 32768。 |
| judge_temperature | 离线 | 裁判采样温度，默认 0。 |
| judge_top_p / judge_top_k | 离线 | 本地裁判采样参数，默认 0.8 / 20。 |
| seed | 离线 | 本地裁判随机种子，默认 42。 |
| SCORER_URLS / SCORER_API_KEY / SCORER_MODEL_NAME | BrowseComp | 远程裁判端点、密钥和模型名；在服务器环境配置。 |

## 6. TRAINING 完整字段

表中默认值直接对应脚本；兼容字段明确标注，修改后用 PLAN_ONLY 检查归档 train.yaml。

| 字段 | Plain 默认 | Beacon 默认 | 作用与修改提示 |
|---|---|---|---|
| deepspeed | `空字符串` | `空字符串` | 单卡保持 None；多卡 ZeRO 由本文件 DISTRIBUTED 生成完整配置。 |
| max_length | `16384` | `16384` | 完整轨迹 token 上限；超长过滤或报错，不静默截断 |
| per_device_batch_size | `1` | `1` | 轨迹 collator 限制为 1，使用梯度累积增大 batch |
| grad_accum_steps | `4` | `4` | 有效 batch = 进程数 × 单卡 batch × 累积步数 |
| learning_rate | `5e-05` | `5e-05` | 学习率；修改后重新 plan |
| weight_decay | `0.01` | `0.01` | 权重衰减 |
| lr_scheduler_type | `'constant'` | `'constant'` | 学习率调度，如 constant / cosine / linear |
| warmup_ratio | `0.0` | `'—'` | Plain 预热比例；范围 0–1 |
| max_grad_norm | `1.0` | `1.0` | 梯度裁剪阈值 |
| num_epochs | `1` | `1` | max_train_steps 为 null 时按 epoch 训练 |
| max_train_steps | `空字符串` | `空字符串` | null 按 epoch；正整数覆盖 epoch，单位 optimizer step |
| use_bf16 | `true` | `true` | 是否启用 BF16，硬件须支持 |
| optim | `'adamw_bnb_8bit'` | `'adamw_bnb_8bit'` | 单卡/DDP 优化器；DeepSpeed 自带 optimizer 时由 distributed 配置决定 |
| gradient_checkpointing | `true` | `false` | Plain 普通层重算；Beacon 使用其专用 checkpoint_blocks，不使用普通层重算。 |
| plain_attention_backend | `'sdpa'` | `'—'` | Plain 注意力实现；sdpa 或环境支持的其他实现 |
| plain_loss_chunk_size | `32` | `'—'` | Plain 分块计算监督 token loss 的块大小 |
| plain_checkpoint_loss | `true` | `'—'` | 是否重算分块输出头 loss，以计算换显存。 |
| plain_cpu_offload_activations | `true` | `'—'` | 用 CPU 内存换 GPU 激活显存 |
| plain_cpu_offload_threshold | `2048` | `'—'` | 达到该 token 长度才 offload；0 表示全部 |
| save_freq_steps | `200` | `200` | 周期保存间隔，单位 optimizer step |
| log_freq_steps | `5` | `5` | 兼容字段；logging_steps 已设置时以 logging_steps 为准 |
| eval_steps | `200` | `200` | 验证间隔，单位 optimizer step |
| seed | `42` | `42` | 训练随机种子 |
| shuffle_train_data | `true` | `true` | Beacon 是否打乱训练数据 |
| use_lora | `true` | `true` | true 训练 LoRA；false 为全参数，资源需求不同 |
| lora_r | `16` | `16` | LoRA 秩 |
| lora_alpha | `32` | `32` | LoRA 缩放参数 |
| lora_dropout | `0.05` | `0.05` | LoRA dropout |
| lora_target_modules | 投影模块列表 | 投影模块列表 | LoRA 投影模块名列表；按实际模型结构核对。 |
| logging_steps | `5` | `5` | 日志间隔，优先于 log_freq_steps |
| loss_segments | 见下表 | `'—'` | 逐 token 监督权重映射，字段说明见损失分段表。 |
| per_device_eval_batch_size | `1` | `'—'` | 验证 collator 仅支持 1 |
| save_total_limit | `2` | `'—'` | Plain 周期 checkpoint 最大保留数；null 表示不限制 |
| beacon_layout_policy | `'—'` | `'configured'` | configured 采用 BEACON 中的布局；whole 强制整段兼容布局并规范化相关字段。 |
| distill_teacher_model_name_or_path | `'—'` | `空字符串` | 蒸馏 teacher；null 使用 student 基座，启用蒸馏需额外资源 |
| distill_teacher_device | `'—'` | `'auto'` | teacher 设备；auto 或显式设备 |
| warmup_steps | `'—'` | `0` | Beacon 预热 optimizer 步数 |
| length_diagnostics_samples | `'—'` | `128` | 训练前统计样本数；0 可关闭 |

## 7. BEACON 完整字段

| 字段 | 默认值 | 作用与修改提示 |
|---|---|---|
| enable_beacon | `true` | 启用 Beacon 模型配置；Beacon 路线保持 True。 |
| beacon_window | `1024` | window 布局文档窗口 token 数 |
| beacon_stride | `1024` | 兼容校验字段，当前推进按 window；不得大于 window |
| beacon_window_enabled | `false` | configured/window 模式需 true；whole 使用 false |
| beacon_ratio | `16` | 固定比例及推理比例；越小 Beacon 越多 |
| beacon_compression_mode | `'fixed'` | fixed 或 random；random 按训练样本采样 |
| beacon_compression_ratios | `[8, 16, 32, 64]` | random 训练的正整数候选列表，要求非空且无重复。 |
| beacon_attn | `'full-coverage'` | 当前仅支持 full-coverage |
| beacon_attention_backend | `'sdpa'` | Beacon 注意力后端；默认 sdpa，切换需验证模型与内核兼容性。 |
| beacon_param | `'q k v'` | 独立投影的空格分隔组合，可选 q/k/v/o；改变后 checkpoint 结构随之改变。 |
| beacon_embed_init | `'eos'` | Beacon token embedding 初始化来源，默认 eos。 |
| beacon_sink_size | `0` | 兼容字段；当前不使用 |
| beacon_attend_prev | `true` | 兼容字段；当前固定保留历史 Beacon |
| beacon_pos | `'append'` | append 或 intersect |
| eval_beacon_ratio | `空字符串` | 兼容字段；推理使用 beacon_ratio |
| beacon_linear_writer_rank | `128` | Qwen3.5 原生线性状态 writer 的低秩维度。 |
| beacon_preserve_native_linear_state | `false` | 保留原生线性注意力状态的消融开关；改变前评估记忆行为。 |
| beacon_question_memory_v1 | `false` | 启用问题条件记忆实验，默认关闭。 |
| beacon_question_max_tokens | `128` | 问题条件记忆最多读取的 question token 数。 |
| beacon_readout_distill_weight | `0.0` | readout 蒸馏权重，0 表示关闭 |
| beacon_output_distill_weight | `0.0` | 输出蒸馏权重；>0 加载 teacher |
| beacon_output_distill_temperature | `1.0` | teacher/student 输出蒸馏温度，必须为正。 |
| beacon_output_distill_strategy | `'all'` | 输出蒸馏 token 选取策略，默认 all；实验前核对 BeaconConfig 校验。 |
| beacon_output_distill_tokens | `8` | 蒸馏采样策略的 token 数。 |
| beacon_output_distill_doc_weighting | `false` | 是否按文档相关性调整输出蒸馏权重。 |
| beacon_output_distill_doc_epsilon | `0.1` | 文档权重平滑量，防止完全零权重。 |
| beacon_doc_region_split | `false` | 按子文档拆分；OpenSeeker 不支持 true |
| beacon_information_mode | `'whole'` | whole 对完整观测压缩；window 对窗口观测压缩，与窗口开关保持一致。 |
| beacon_information_loss | `false` | 开启续写监督需正 continuation 权重 |
| beacon_ce_loss_weight | `1.0` | assistant 交叉熵权重 |
| beacon_continuation_loss_weight | `0.1` | 续写 loss 权重；实际生效还需 information_loss 开启 |
| beacon_continuation_tokens | `4` | 启用 continuation 监督时的后续目标 token 数。 |
| beacon_continuation_per_beacon | `false` | 是否为每个 Beacon 建立 continuation 监督。 |
| beacon_keep_window | `0` | keep 段上限；0 整段；policy=configured 时生效 |
| beacon_loss_chunk_size | `32` | Beacon 监督 loss 分块大小 |
| beacon_checkpoint_loss | `true` | Beacon 输出头 loss 重算 |
| beacon_checkpoint_blocks | `true` | 训练时重算无状态 attention/MLP/writer，避免重放状态更新。 |
| beacon_cpu_offload_activations | `true` | 训练时将符合长度阈值的激活暂存 CPU，以主机内存换 GPU 显存。 |
| beacon_cpu_offload_threshold | `2048` | 达到该 token 长度才 offload |
| loss_segments | 见损失分段表 | 逐 token 监督权重映射，字段说明见损失分段表。 |

## 8. 损失分段权重

Plain 位于 TRAINING.loss_segments；Beacon 位于 BEACON.loss_segments。默认全部 1.0，0 表示该段不计监督 loss；观测体按协议不作 assistant 监督。

| 字段 | 监督区域 |
|---|---|
| think_tag | think 起止标签 |
| think_content | think 标签内部文本 |
| tool_call_tag | tool call 起止标签 |
| tool_call_content | tool call JSON 内容 |
| answer_tag | answer 起止标签 |
| answer_content | answer 内部文本 |
| other | 其他 assistant 文本 |
| search_tag | 兼容 Search-R1 的 search 标签 |
| search_content | 兼容 Search-R1 的 search 内容 |

## 9. DISTRIBUTED 多卡字段

DISTRIBUTED.zero2 / zero3 分别提供完整 DeepSpeed 字典；脚本不加载外部 ds JSON 文件。单卡保持 TRAINING.deepspeed=''。以下为配置键路径，zero3 仅 Plain。

| 字段 | 作用与修改提示 |
|---|---|
| bf16.enabled | auto：沿用 Trainer BF16 设置。 |
| train_micro_batch_size_per_gpu | auto：跟随单卡 batch，当前轨迹必须为 1。 |
| train_batch_size / gradient_accumulation_steps | auto：由卡数、单卡 batch、梯度累积解析。 |
| gradient_clipping | auto：跟随 max_grad_norm。 |
| optimizer.type | AdamW；使用 DeepSpeed 优化器时覆盖单卡/DDP 的 optim 选择。 |
| optimizer.params.lr / betas / eps / weight_decay | auto：与 Trainer 优化器参数保持一致。 |
| zero_optimization.stage | zero2=2，zero3=3，不可与后端名称冲突。 |
| zero_optimization.offload_optimizer.device / pin_memory | 默认 cpu / True，将优化器状态放到主机内存。 |
| zero_optimization.offload_param.device / pin_memory | ZeRO-3 默认 cpu / True，卸载模型参数。 |
| zero_optimization.overlap_comm | 默认 False，控制通信计算重叠。 |
| zero_optimization.contiguous_gradients | 默认 True，连续梯度存储。 |
| zero_optimization.reduce_bucket_size / allgather_bucket_size | 通信桶元素数，默认 10000000。 |
| zero_optimization.ignore_unused_parameters | ZeRO-2 默认 True，允许动态路径未参与梯度的参数。 |
| zero_optimization.stage3_prefetch_bucket_size | ZeRO-3 预取桶元素数，默认 10000000。 |
| zero_optimization.stage3_param_persistence_threshold | ZeRO-3 小参数常驻阈值，默认 10000。 |
| zero_optimization.stage3_max_live_parameters / stage3_max_reuse_distance | ZeRO-3 活跃参数/复用距离上限，默认各 100000000。 |
| zero_optimization.stage3_gather_16bit_weights_on_model_save | ZeRO-3 保存时聚合 16 位权重，默认 True。 |
| zero_allow_untested_optimizer | 默认 True，允许非内置兼容优化器组合；修改前在 Linux 验证。 |

## 10. 安装、报告与检查变量

| 脚本 | 变量 | 作用 |
|---|---|---|
| 00_setup_environment.py | ENVIRONMENT_DIR / TORCH_SPEC / TORCH_INDEX / PIP_INDEX / HF_ENDPOINT | 环境位置、PyTorch wheel、包源与 Hub 源；见 10 环境文档。 |
| 00_setup_environment.py | INSTALL_DEEPSPEED / DEEPSPEED_SPEC | 是否安装 DeepSpeed，以及待验证的版本约束。 |
| 80_dataset_report.py | DATASET_DIR / SPLIT / LIMIT / OFFSET / TITLE | 数据版本、展示划分、条数、偏移、标题；LIMIT=0 全量。 |
| 81_prediction_report.py | METHOD / BENCHMARK / EVALUATION_DIR / TITLE | 匹配测试记录并生成 HTML。 |
| 90_validate.py | RUN_UNIT_TESTS / TEST_PATTERNS | 是否运行无模型标准库测试，以及测试文件模式。 |
| 91_test_algorithms.py | TEST_PATH / VERBOSE / TOKENIZER_PATH | Linux 算法回归范围、输出详细程度和可选本地 tokenizer；未缓存的 tokenizer 兼容测试会明确跳过。 |

## 11. 修改联动检查

- 更换模型：下载/准备本地模型 → 重新按相同 tokenizer 身份预处理 → 修改训练 MODEL_PATH/MODEL_FAMILY → 新实验。
- 更换长度或筛选条件：新 PROFILE → 修改训练 DATASET_DIR/max_length → 新 RUN_NAME。
- 更换多卡配置：同步 CUDA_VISIBLE_DEVICES、NPROC_PER_NODE、DISTRIBUTED_BACKEND 和对应 DISTRIBUTED。
- 更换 benchmark 来源或 BM25 权重：新 VERSION → 重新 prepare/index → 修改各测试 BENCHMARK_DIR/标题权重。
- 更换裁判：只修改 JUDGE/服务器裁判变量，继续指向原 EVALUATION_DIR，生成新的 scores 记录。
