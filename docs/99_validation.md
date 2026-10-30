# 99 本次验证记录

本次在现有 `search-comp-qwen3.5` 环境下完成代码迁移与离线验证，没有启动真实模型训练、真实多卡 GPU 训练或真实模型生成。

| 项目 | 结果 | 证据 |
|---|---|---|
| 全量回归 | 465 passed、2 skipped、1 warning | `outputs/reports/verification/random_compression_pytest.log` |
| 跳过项 | 两项显式 GPU / 预训练模型 smoke | pytest 日志中列出具体位置 |
| 警告 | tiny PEFT 模型保存时缺少基座 config 的既有提示 | pytest 日志 |
| Python / shell | compileall 与全部 bash -n 通过 | 本次终端检查 |
| Python 薄入口 | 13 个非 GPU 入口的 --help 导入和参数检查通过 | 本次终端检查；不执行 03 GPU 探针 |
| 四卡计划 | Plain 与 Beacon 均生成 torchrun 4 进程、ZeRO-2 命令，支持仅训练集与空 validation | `outputs/reports/verification/plans.json`；使用临时数据，只执行 plan |
| 核心保留 | 按用户要求精简重复配置后，保留源 search_comp/configs/tests 中 109 个文件，77 个逐字节一致；保留全部核心算法与测试 | `docs/90_core_inventory.json` |
| 原始数据迁移 | 4 个文件、6,763,061,538 bytes，源/目标 SHA256 全部一致 | `outputs/reports/verification/raw_copy.json` |
| 快照语义校验 | 官方快照与可读 JSONL 的 11,677 个 JSON 对象逐条一致 | `outputs/reports/snapshot_verification.json` |
| 协议移除 | 核心代码、脚本、配置、测试和维护文档无旧追加答案标签/配置/提取逻辑 | 本次全文检索 |

回归覆盖模型 tiny 前向/反向、Beacon 压缩和线性记忆、蒸馏、适配器保存、多卡 launcher/CPU Gloo、数据/长度/损失边界、评测分片与失败分母，以及 HTML 的 Node 客户端执行。新筛选回归验证闭区间、精确长度、tokenizer 身份、缺失正确性排除、observation 角色边界、训练集单独导出和 token 不变性。新汇总回归验证 OpenSeeker 严格 EM、F1、失败样本和 Excel 字段。

CUDA/NCCL、实际四卡训练、目标服务器 CPUAdam 编译、4B 模型显存与吞吐尚未实测。显存回调仅提供后续采集机制，不代表已获得显存上限。数据和启动计划可在当前目录直接审查。

全量预处理已完成：11,677 条，Correct/Incorrect = 4,949/6,728，无结构无效记录；默认 train/validation/test = 644/9/11。已实际执行 11 的 32k 区间筛选、12 的 64k 显存样本筛选、13 的 observation 过滤及 40 的 HTML 导出，并验证所选记录符合长度/正确性规则与问题跨 split 不重复。43 条 64k 显存样本的实测长度范围为 64,541—65,528。处理日志保存在 `outputs/reports/verification/`。

另用真实默认数据生成了 Plain/Beacon 各一份四卡 ZeRO-2 配置计划，目录索引见 `outputs/reports/verification/actual_plans.json`。保留配置供审查，未执行其中的训练命令。临时 train-only 计划验证与实际数据计划在报告中分别记录。

另已实际生成 `polar_memory_exact_65117`：1 条恰好 65,117 token 的 Correct、有 observation 训练样本，并验证长度精确相等；仅用于显存测试。

## 随机压缩比训练验证

新增 22 项随机压缩回归，完整测试现为 465 passed、2 skipped、1 warning。覆盖两种模型与 append/intersect 的随机/固定损失和梯度等价、实际缓存长度、候选校验、种子、推理固定、LoRA 日志和配置兼容。

双进程 CPU/Gloo tiny 模型完成 2 个 optimizer steps、每步累积 2 个 micro-batch，实际采样到 2 和 4 两种比例；各 rank 结束后的 LoRA 参数逐位同步，checkpoint/final adapter 保存成功。日志及逐 rank 比例在 `outputs/reports/verification/random_compression_ddp.log` 与 `random_compression_ddp/ratios_rank*.json`。测试显式设置 ACCELERATE_USE_CPU=true 及空 GPU 可见列表；不加载真实模型权重，不在 GPU 上训练。

30/37 两个一键入口均实际以 ACTION=plan、COMPRESSION_MODE=random、候选 [4,8,16] 生成四卡 ZeRO-2 命令与配置快照，未执行训练。索引见 `outputs/reports/verification/random_compression_plans.json`。当前默认仍为 fixed；正式切换只需修改脚本顶部变量。新增说明见 [31 随机压缩](31_random_compression.md)。

## 2026-10-28 官方 Benchmark 增量验证

本次保留训练器、Beacon 算法与旧 EM/F1 实现，新增固定 OpenSeeker 提交 `e3dec1f727d87c5eaa93506e4882c67237647098` 的官方脚本调度、BrowseComp 导入、native 工具桥接与覆盖审计。

- 下载并解密官方 BrowseComp CSV，1,266 题；JSONL 与来源指纹保存在 `outputs/data/benchmarks/browsecomp.manifest.json`。不展示测试题，不将测试集加入训练数据。
- 新增 18 项 CPU 回归通过，覆盖加密数据导入、禁止覆盖、重复/缺参考校验、参考轨迹隔离、重试去重、v2 boxed 后缀、漏生成题保留、评分覆盖率、正确 CLI 参数、模型定位、Beacon 服务拒绝、dry-run 无写入，以及生成→裁判真实子进程调度。
- 全量回归 **483 passed、2 skipped、1 warning**。日志：`outputs/reports/verification/official_benchmark_pytest.log`；warning 为既有 PEFT tiny 模型缺基座 config 提示。
- 单独建立 `/tmp` 的最小评测虚拟环境，不更改训练环境，实测新增 API 入口可独立导入，无需加载 torch、Transformers 或 BM25。
- 用未经修改的官方 `eval/eval.py` 和本地 OpenAI 兼容模拟裁判跑通真实评分请求，验证 temperature=0、skip_special_tokens=False、A/B 解析、v2 boxed 问题匹配与结果审计。两条合成题得到 1 对/1 错，仅为协议测试，**不是 BrowseComp 模型成绩**。记录：`outputs/reports/verification/official_benchmark_smoke/`。
- 官方 v1/v2 生成脚本的 `--help`、新增入口的 dry-run、Python 编译、shell 语法和 diff 空白检查通过。

未执行真实训练、真实 checkpoint 在线生成、GPU SGLang 部署、真实外部搜索/visit/E2B 或付费裁判。实际成绩需要训练后的 checkpoint、支持目标模型的服务环境、搜索/摘要/裁判 API 配置。native 的完整模型和真实工具联调仍待用户在目标环境执行；它的序列化、解码与预算差异在 [42](42_openseeker_benchmark.md) 中明确说明。

## 2026-10-28 固定语料离线 Benchmark 验证

新增 56/57 两条完整流程：独立数据导入、持久化 SQLite FTS5 BM25、Plain/Beacon native 工具循环、本地评分与版本/覆盖审计。没有删除原有训练或在线评测实现。

- 新增 25 项离线回归通过，覆盖官方参考文件指纹、加密 BCP Parquet 解密、独立 corpus 与 dev context 隔离、tar.bz2 内 bz2 分片不解包读取、HTML 与句号/句子编号保留、索引跨进程复用/只读/拒绝重复 ID、预算与 local URL 限制、完整 search→visit→answer 的 Plain/Beacon 路径和 observation 压缩区域。
- 直接执行随代码附带的固定 Hotpot evaluator，覆盖归一化、yes/no/noanswer 特例、漏题/失败分母、模型预测 supporting facts、SP/Joint，以及无效 sp 不改变答案 EM/F1。
- 固定 BCP 官方 prompt/parser/citation/qrels 驱动模拟裁判回归，覆盖累计召回、漏题、未知标签与 confidence 样本不足标记；另在真实子进程中跑通 retrieval-only CLI，不加载任何裁判模型。
- 全量回归 **508 passed、2 skipped、3 warnings**。日志：`outputs/reports/verification/offline_benchmarks_pytest.log`。一项为既有 PEFT tiny 模型提示；两项为固定 BCP 上游 prompt 的 `\%` 转义警告，未改写上游源文件。
- 真实 IP socket 与 DNS 连接在子进程中被离线 audit hook 拒绝；默认 dry-run 不创建目录。shell 语法、Python 编译、文档链接与 diff 空白检查通过。
- 已生成可查看的小样本持久化索引与评分产物：`outputs/reports/verification/offline_benchmarks_smoke/`。00_scope.json 明确标注其为**合成协议样本**；其中 Hotpot 执行真实官方指标，BCP 使用真实 qrels/prompt/parser 和模拟裁判。任何其中的正确率都不是实际模型基准成绩。
- reference 文件与对应固定提交的官方源逐字节一致，许可证和 00_provenance.json 随代码保留；未提交问题明文、corpus、权重、索引或日志。

未下载完整 GB 级 corpus、未完成 fullwiki 全量索引、未加载真实训练 checkpoint 或 Qwen3-32B 裁判，也未测量真实 GPU 显存/吞吐。历史 CMU dev 下载主机在本环境解析失败，代码提供官方 HF fullwiki validation 备选和显式镜像地址；HF 与 Stanford 全量资源下载需在目标联网准备环境执行。完整参数和复现步骤见 [43](43_offline_benchmarks.md)，固定来源和授权见 [44](44_offline_sources.md)。

补充使用本地保存的 tiny 随机 Qwen3 checkpoint，真实运行 LocalJudge 的 HF tokenizer/chat template/权重加载与 generate，并验证裁判上下文超限处理；该测试不下载权重，也不代表真实裁判的判题质量。Hotpot 独立 score 另在仅标准库的 Python 路径下完成小样本验证，无需导入 torch、Transformers 或 pyarrow。

## 四卡 A800 长轨迹显存修复（2026-10-29）

本次仅 CPU 小模型验证，没有启动真实训练或下载生产模型。环境：PyTorch 2.7.0+cu126、Transformers 5.9.0；测试显式隐藏 GPU。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_beacon_checkpoint_blocks.py tests/test_distributed_training.py \
  tests/test_beacon_qwen35_linear_memory.py tests/test_beacon_question_memory.py \
  tests/test_qwen3_whole_information.py
```

结果：101 passed、6 skipped；PEFT 小模型保存测试有一条缺少基础模型配置的预期 warning。新测试比对重计算开关前后的 loss、全部梯度和最终记忆，覆盖两类模型、whole/window、offload开关和冻结embedding。未运行两进程DDP更新或DeepSpeed训练；本次多卡验证限于启动参数、设备配置逻辑及相关单元测试。

`bash -n` 检查30/37/38脚本通过，`git diff --check` 通过。38的 `ACTION=plan` 验证仅归档配置并输出四进程ZeRO-2启动命令，不加载模型、不执行该命令。目标服务器的真实80 GB显存峰值、CPU内存、NCCL和融合算子组合尚未实测；按 [52](52_beacon_oom.md) 和 [51](51_memory_probe.md) 验收。


## 2026-10-30 保留原始 system，测试复用训练提示

本节记录保留原始 system 的验收结果。数据协议为 `openseeker-chatml-v3-original-answer`，保留每条训练记录 system；独立 tools 只在输入渲染时补充。native 使用 PROMPT_DATA > RUN_DIR 训练文件 > 内置原始 Polar profile；离线只调整必要的工具描述、能力和格式。详见 [45](45_unified_agent_flow.md)。

完整回归：**564 passed、2 skipped、9 warnings，85.90 秒**。环境仍为项目 search-comp-qwen3.5；日志为 `outputs/reports/verification/original_system_pytest.log`，没有覆盖上一版日志。命令：

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/huangzj/miniconda3/envs/search-comp-qwen3.5/bin/python \
  -m pytest tests -q --disable-warnings
```

新增/调整的检查包括：

- normalize 保留原始 system，原有 v3 数据和长度记录可复用，不替换训练 system。
- 独立 tools 元数据不写入保存的 system；预处理计长、Plain/Beacon SFT 编码、native schema 渲染使用相同逻辑；每轮前缀逐 token 对齐。
- 显式提示源、RUN_DIR 自动定位、来源覆盖、多个 system 拒绝、缺失文件与空源处理。
- 在线 Plain/Beacon 直接读取测试指定的训练原提示，不使用 upstream developer_prompt 替代；训练问题和答案不泄漏到测试消息。
- 离线 Plain/Beacon 保留原始提示主体与自定义指令，schema 不再承诺联网/固定 top 10/LLM 摘要，search-only 移除 visit。
- 所有原有模型算法、LoRA/蒸馏、在线/离线、官方 v1/v2、replay/corpus/HTTP、多卡 worker 和计分回归继续通过；真实工具与模型由测试桩或 tiny 模型覆盖。

只读审计当前 644 条 v3 训练记录：`normalize(row) == row` 对全部记录成立。解析得到的唯一 system 与内置 `openseeker_v1_system.txt` 完全相等；原始 Polar 第一条 system 也与之相等。system SHA256 为 `53e65173df52e70802665c69f9aae1efb476773b6db8071aa5ea601967421840`。审计产物为 `outputs/reports/verification/original_system_audit.json`，没有写回原始或预处理数据。

47 native 和 56 的 PROMPT_DATA dry-run 参数映射通过，全部 shell 的 bash -n、Python compileall、文档本地链接和 git diff --check 通过。dry-run 仅打印计划，不声称真实 checkpoint/API 已联调。47/56/57 的 PROMPT_DATA 参数说明已写入脚本及 42/43/45；离线 worker 保留 RUN_DIR 和显式提示源。

未运行真实训练、真实 checkpoint 在线或离线生成、在线搜索/摘要/裁判服务，也未重建全量数据或索引。现有数据、模型、tmp 的结果均未删除或改写。已有原始 v3 数据无需为本次修复重新计长。

## 2026-10-30 Beacon 完整实验指南与流程整理

新增 [32 Beacon 完整实验指南](32_beacon_experiment_guide.md)，按环境、数据准备/筛选、配置与 plan、训练、合并、少量诊断、离线/在线基准、结果审查的顺序给出命令、参数、阅读路径和每步完成条件。README、文档索引、30、45 已接入该指南。已删除不再使用的数据协议专项分支、迁移历史段落和对应专用测试断言；现行数据仍保留原始 system，使用原有数据格式。

验证结果：

- 全量 CPU/离线回归 **564 passed、2 skipped、9 warnings，85.72 秒**，日志为 `outputs/reports/verification/beacon_experiment_guide_pytest.log`。
- 指南中 23 个 bash 命令块通过 bash -n；内嵌 Python 示例通过 AST 语法检查；所有脚本路径和本地 Markdown 链接存在。
- 全部 shell 语法、Python compileall 和 git diff --check 通过；废弃数据协议标识及其专用兼容常量在维护代码/文档中无残留。
- 使用现有真实 prepared 数据分别运行 30 的单卡 ACTION=plan 和 38 的四卡 ACTION=plan：生成配置与启动命令成功，四卡命令包含 nproc_per_node=4 和 ds_zero2.json。日志分别为 `beacon_guide_single_plan.log`、`beacon_guide_four_gpu_plan.log`，均位于 `outputs/reports/verification/`。

plan 仅创建新的配置目录，没有加载真实训练模型或执行日志中的 next 训练命令。没有执行本指南中的真实训练、显存训练探针、模型合并、在线 API 请求或真实 checkpoint 评测，也未重建数据或索引。已有数据、模型和评测结果保持不变。
