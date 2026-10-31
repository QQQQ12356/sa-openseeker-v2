# 90 来源与迁移范围

骨架来源：`/home/huangzj/proj/searchagent/sa-qwen-openseeker`。本次开始时目标目录为空且不是 Git 仓库；源目录保持原样。初次迁移复制全部 search_comp、configs、tests、依赖与环境文件；后续按用户要求删除重复的 Qwen3.5 Beacon 配置，入口统一使用 whole 基础配置；模型算法和多卡框架延续原实现，按新答案协议调整数据、训练分段损失、评测和测试。

数据只复制 `outputs/data/raw/polar/`，不复制旧规范化数据、Search-R1、其他原始数据、模型或训练结果。当前目录的派生数据由新预处理实际生成。原始文件与快照按相对路径及 SHA256 对照，验证结果在 99。

原 scripts 中 Python 实现移入可导入的 search_comp/workflows，编号的 shell 入口按功能分入 scripts/sh 下的 common、tools、data、plain、beacon、report、benchmark、baseline，Python 入口放入 scripts/py，模板单独存放。旧路径到新路径的逐项映射见 [90_script_mapping.json](90_script_mapping.json)，目录说明见 [00 索引](../00_index.md)。核心文件保留清单和来源 SHA256 见 [90_core_inventory.json](90_core_inventory.json)。

原算法/环境文档保留为 61—67；旧工作流与旧结果报告已由本套编号文档替换，不再把旧数据条数当成当前结果。根索引为 `00_README.md`。本次没有初始化 Git 或创建提交。
