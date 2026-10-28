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
