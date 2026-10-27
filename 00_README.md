# sa-openseeker-v2

以 `../sa-qwen-openseeker` 为骨架，保留 Qwen3 / Qwen3.5、Plain / Beacon、Search-R1、多工具轨迹处理、LoRA、蒸馏算法和原有测试。OpenSeeker 训练保留原始回答，不追加参考答案；评测使用模型最后一轮回答计算 EM/F1。

只迁移 Polar 原始快照；本目录中的规范化数据由新代码重新生成，没有复制其他数据集、模型权重、旧训练结果或实验日志。本次没有启动真实模型训练。

先阅读 [文档索引](docs/00_index.md)。常用入口如下，参数直接修改各 shell 脚本顶部，然后运行，无需传命令行参数：

```bash
bash scripts/sh/01_test.sh                    # CPU 回归测试
bash scripts/sh/10_prepare.sh                 # 全量规范化和分词；输出存在时拒绝覆盖
bash scripts/sh/11_select.sh                  # 长度区间/精确长度/正确性/观测筛选
bash scripts/sh/12_select_memory_probe.sh     # 生成显存测试训练集
bash scripts/sh/40_dataset_html.sh            # 离线查看数据
# 以下由使用者之后在训练服务器执行，本次没有执行：
# bash scripts/sh/20_plain_train.sh
# bash scripts/sh/30_beacon_train.sh
# bash scripts/sh/26_plain_memory_probe.sh
# bash scripts/sh/37_beacon_memory_probe.sh
# bash scripts/sh/21_plain_eval.sh
# bash scripts/sh/31_beacon_eval.sh
```

训练脚本支持将 `ACTION="train"` 改成 `ACTION="plan"`，只创建可审查的配置和启动命令。多卡参数集中在训练脚本顶部：`CUDA_VISIBLE_DEVICES`、`DISTRIBUTED_BACKEND`、`NPROC_PER_NODE`。配置脚本后直接运行即可。

目录分工：`search_comp/` 为核心实现，`search_comp/workflows/` 为可导入的流程实现，`scripts/sh/` 为编号的一键入口，`scripts/py/` 为编号的 Python 薄入口，`scripts/templates/` 为编号的 HTML 模板，`tests/` 为回归测试，`configs/` 为训练配置，`docs/` 为编号文档。

本次已生成可用的默认数据（train/validation/test=644/9/11）、43 条约 64k 的显存测试数据和数据 HTML。已有输出无需重复生成，详情见 [数据说明](docs/10_data.md) 与 [验证记录](docs/99_validation.md)。全量回归为 **443 通过、2 项 GPU 专用测试跳过**。
