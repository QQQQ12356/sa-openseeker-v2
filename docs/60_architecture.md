# 60 核心架构与损失

`search_comp/models/` 保留 Plain/Beacon 的 Qwen3、Qwen3.5 实现、共享压缩配置、Qwen3.5 线性状态、question memory 与模型加载；`trainer/` 保留普通 SFT、Beacon、native、continuation、输出/读出/信息蒸馏、适配器合并和多卡 launcher。`data/` 保留 Search-R1、文档规范化、BM25、语料构建、OpenSeeker 编码与数据工厂。`evaluation/` 保留原交互式、native、直接回答和统一指标实现。

OpenSeeker 原始文本保留，结构化工具调用统一为 tool_calls_begin/tool_call JSON；仅 observation 正文可压缩，不截断轨迹。数据读取使用 JSONL offset 懒加载，减少多 GB 数据常驻内存。训练和 agent 推理共享 encode()/pieces()，测试验证每个生成前缀与训练序列对齐。

主 CE 的分段类别：think_tag/content、search_tag/content、answer_tag/content、tool_call_tag/content、other。`beacon.loss_segments` 同时用于 Plain 和 Beacon，默认全部启用、权重 1。每 token 权重乘交叉熵，分母为有效权重和。关闭类别等价于权重 0，不参与主 CE 梯度与分母；全部关闭得到可反传的有限零损失。token/labels/压缩区域不受权重改变影响。

continuation、information、output KL、readout 蒸馏仍各有独立总权重；输出蒸馏会避开关闭监督的片段，stratified 采样按 answer、思考、search、tool_call 等队列分配预算。当前类别 ID 已删除弃用类别，需使用本项目重新生成的分段数据，不能直接复用旧类别数组。

流程实现在 `search_comp/workflows/`，`scripts/py/` 仅负责调用 main，便于测试直接导入实现、命令入口维持编号。原回归文件整体保留，修改涉及协议、路径和类别的断言，新增长度/质量/观测筛选测试；没有通过删除核心测试来绕过迁移问题。
