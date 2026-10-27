# 99 本次验证记录

本次在现有 `search-comp-qwen3.5` 环境下完成代码迁移与离线验证，没有启动真实模型训练、真实多卡 GPU 训练或真实模型生成。

| 项目 | 结果 | 证据 |
|---|---|---|
| 全量回归 | 443 passed、2 skipped、1 warning | `outputs/reports/verification/pytest.log` |
| 跳过项 | 两项显式 GPU / 预训练模型 smoke | pytest 日志中列出具体位置 |
| 警告 | tiny PEFT 模型保存时缺少基座 config 的既有提示 | pytest 日志 |
| Python / shell | compileall 与全部 bash -n 通过 | 本次终端检查 |
| Python 薄入口 | 13 个非 GPU 入口的 --help 导入和参数检查通过 | 本次终端检查；不执行 03 GPU 探针 |
| 四卡计划 | Plain 与 Beacon 均生成 torchrun 4 进程、ZeRO-2 命令，支持仅训练集与空 validation | `outputs/reports/verification/plans.json`；使用临时数据，只执行 plan |
| 核心保留 | 按用户要求精简重复配置后，保留源 search_comp/configs/tests 中 109 个文件，84 个逐字节一致；保留全部核心算法与测试 | `docs/90_core_inventory.json` |
| 原始数据迁移 | 4 个文件、6,763,061,538 bytes，源/目标 SHA256 全部一致 | `outputs/reports/verification/raw_copy.json` |
| 快照语义校验 | 官方快照与可读 JSONL 的 11,677 个 JSON 对象逐条一致 | `outputs/reports/snapshot_verification.json` |
| 协议移除 | 核心代码、脚本、配置、测试和维护文档无旧追加答案标签/配置/提取逻辑 | 本次全文检索 |

回归覆盖模型 tiny 前向/反向、Beacon 压缩和线性记忆、蒸馏、适配器保存、多卡 launcher/CPU Gloo、数据/长度/损失边界、评测分片与失败分母，以及 HTML 的 Node 客户端执行。新筛选回归验证闭区间、精确长度、tokenizer 身份、缺失正确性排除、observation 角色边界、训练集单独导出和 token 不变性。新汇总回归验证 OpenSeeker 严格 EM、F1、失败样本和 Excel 字段。

CUDA/NCCL、实际四卡训练、目标服务器 CPUAdam 编译、4B 模型显存与吞吐尚未实测。显存回调仅提供后续采集机制，不代表已获得显存上限。数据和启动计划可在当前目录直接审查。

全量预处理已完成：11,677 条，Correct/Incorrect = 4,949/6,728，无结构无效记录；默认 train/validation/test = 644/9/11。已实际执行 11 的 32k 区间筛选、12 的 64k 显存样本筛选、13 的 observation 过滤及 40 的 HTML 导出，并验证所选记录符合长度/正确性规则与问题跨 split 不重复。43 条 64k 显存样本的实测长度范围为 64,541—65,528。处理日志保存在 `outputs/reports/verification/`。

另用真实默认数据生成了 Plain/Beacon 各一份四卡 ZeRO-2 配置计划，目录索引见 `outputs/reports/verification/actual_plans.json`。保留配置供审查，未执行其中的训练命令。临时 train-only 计划验证与实际数据计划在报告中分别记录。

另已实际生成 `polar_memory_exact_65117`：1 条恰好 65,117 token 的 Correct、有 observation 训练样本，并验证长度精确相等；仅用于显存测试。
